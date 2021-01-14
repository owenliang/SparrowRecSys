import os
from pyspark import SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.feature import Word2Vec
from pyspark.ml.linalg import Vectors
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F


class UdfFunction:
    @staticmethod
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


def processItemSequence(spark, rawSampleDataPath):
    # rating data
    # 加载评分表
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)

    # ratingSamples.show(5)
    # ratingSamples.printSchema()
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))

    # 取>=3.5分的记录, 按userId拉链movieId，返回从古至今的movieId序列
    userSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias('movieIds'))

    return userSeq.select('movieIds')

    # userSeq.select("userId", "movieIdStr").show(10, truncate = False)
    # return userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' ')).toDF(['movieIdStr'])


# 训练LSH模型，用于将vector分桶
def embeddingLSH(spark, movieEmbDf):
    # movieEmbSeq = []
    # for key, embedding_list in movieEmbMap.items():
    #     embedding_list = [np.float64(embedding) for embedding in embedding_list]
    #     movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    #
    #
    # movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")

    # 实际就是训练出一个vector，用于将emb向量内积为数字，再经过多个hash函数后取模分多个桶
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="vector", outputCol="bucketId", bucketLength=0.1,
                                                      numHashTables=3)
    # 训练并生成分桶
    bucketModel = bucketProjectionLSH.fit(movieEmbDf)
    embBucketResult = bucketModel.transform(movieEmbDf)

    print("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    print("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate=False)

    # 给定任意emb vector，可以计算出其分桶，并在桶内计算余弦距离最近的emb
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDf, sampleEmb, 5).show(truncate=False)


def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    # word2vec模型，词（物品）向量长度embLength，滑动窗口5个词（生成训练样本目的），训练10轮
    word2vec = Word2Vec(vectorSize=embLength, windowSize=5, numPartitions=10, inputCol='movieIds')
    model = word2vec.fit(samples)

    # 查找电影158的相似电影20部
    synonyms = model.findSynonyms("158", 20)
    synonyms.show(20, False)

    # 保存所有movie的emb向量
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)

    # 获取所有word的embedding向量
    vectors = model.getVectors()
    vectors.show(20, False)

    def writer(iterator):
        with open(embOutputPath, 'w') as f:
            for row in iterator:
                movie_id = row['word']
                vector = " ".join([str(col) for col in row['vector']])
                f.write(movie_id + ":" + vector + "\n")
        return iter([])

    # 将df转row rdd，写入文件中
    vectors.rdd.repartition(1).mapPartitions(writer).count()

    # 训练LSH（只是演示用）
    embeddingLSH(spark, vectors)
    return model


def generate_pair(x):
    # eg:
    # watch sequence:['858', '50', '593', '457']
    # return:[['858', '50'],['50', '593'],['593', '457']]
    pairSeq = []
    previousItem = ''
    for item in x:
        if previousItem:
            pairSeq.append((previousItem, item))
        previousItem = item
    return pairSeq

# 生成转移矩阵
def generateTransitionMatrix(samples):
    # 根据用户访问序列，生成成对的电影转移元祖(from movie, to movie)
    pairSamples = samples.rdd.flatMap(lambda x: generate_pair(x['movieIds']))

    # 统计(from movie,to movie)的出现次数，取回本地内存
    pairCountMap = pairSamples.countByValue()

    # 1）建索引：from movie -> to movie边的出现次数
    # 2）统计from movie出发的所有边数
    # 3）总边数
    pairTotalCount = 0
    transitionCountMatrix = defaultdict(dict)
    itemCountMap = defaultdict(int)
    for key, cnt in pairCountMap.items():
        key1, key2 = key
        transitionCountMatrix[key1][key2] = cnt
        itemCountMap[key1] += cnt
        pairTotalCount += cnt

    # 计算每条from movie -> to movie边的转移概率
    transitionMatrix = defaultdict(dict)
    itemDistribution = defaultdict(dict)
    for key1, transitionMap in transitionCountMatrix.items():
        for key2, cnt in transitionMap.items():
            transitionMatrix[key1][key2] = transitionCountMatrix[key1][key2] / itemCountMap[key1]

    # 计算每个from movie出发的边数占总边数的比例
    for itemid, cnt in itemCountMap.items():
        itemDistribution[itemid] = cnt / pairTotalCount
    return transitionMatrix, itemDistribution


def oneRandomWalk(transitionMatrix, itemDistribution, sampleLength):
    sample = []

    # pick the first element
    # 选出随机游走的起点movie
    randomDouble = random.random()
    firstItem = ""
    accumulateProb = 0.0
    for item, prob in itemDistribution.items(): # itemDistribution是电影分布比例
        accumulateProb += prob
        if accumulateProb >= randomDouble:
            firstItem = item
            break

    # 起点加入序列
    sample.append(firstItem)

    # 开始随机游走
    curElement = firstItem
    i = 1
    while i < sampleLength:
        # 没有从该movie转移走的边
        if (curElement not in itemDistribution) or (curElement not in transitionMatrix):
            break

        # 取出from movie的所有边
        probDistribution = transitionMatrix[curElement]

        # 根据概率选择一个边进行移动
        randomDouble = random.random()
        accumulateProb = 0.0
        for item, prob in probDistribution.items():
            accumulateProb += prob
            if accumulateProb >= randomDouble:
                curElement = item
                break
        sample.append(curElement)
        i += 1
    return sample

# 随机游走一轮，生成1个训练样本（1个访问序列）
def randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength):
    samples = []
    for i in range(sampleCount):
        samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    return samples

# 随机游走生成word2vec训练样本
def graphEmb(samples, spark, embLength, embOutputFilename, saveToRedis, redisKeyPrefix):
    # 生成转移矩阵
    transitionMatrix, itemDistribution = generateTransitionMatrix(samples)

    # 采样20000组movie id序列, 每一组包含10个movie id
    sampleCount = 20000
    sampleLength = 10

    newSamples = randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength)

    # 利用样本进行训练
    rddSamplesDf = spark.sparkContext.parallelize(newSamples).map(lambda x: Row(x)).toDF(['movieIds'])
    rddSamplesDf.show(10, False)
    trainItem2vec(spark, rddSamplesDf, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)

# 生成用户emb向量
def generateUserEmb(spark, rawSampleDataPath, model, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    # 加载评分数据
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)

    # Vectors_list = []
    # for key, value in model.getVectors().items():
    #     Vectors_list.append((key, list(value)))
    # fields = [
    #     StructField('movieId', StringType(), False),
    #     StructField('emb', ArrayType(FloatType()), False)
    # ]
    # schema = StructType(fields)
    # Vectors_df = spark.createDataFrame(Vectors_list, schema=schema)

    # 生成dataframe，包含两列： movieId和emb
    Vectors_df = model.getVectors().withColumn('movieId', F.col('word')).withColumn('emb', F.col('vector')).drop('word', 'vector')

    # 评分表inner join向量表，用movieId列
    ratingSamples = ratingSamples.join(Vectors_df, on='movieId', how='inner')

    # 按userId做groupby，将其物品embeeding向量求和，然后写入到文件中
    def writer(iterator):
        with open(embOutputPath, 'w') as f:
            for row in iterator:
                userId = row[0]
                emb = " ".join([str(col) for col in row[1]])
                f.write(userId + ":" + emb + "\n")
        return iter([])
    result = ratingSamples.select('userId', 'emb').rdd.map(lambda x: (x[0], x[1])) \
        .reduceByKey(lambda a, b: [a[i] + b[i] for i in range(len(a))]).mapPartitions(writer).count()

if __name__ == '__main__':
    # 连接spark
    conf = SparkConf().setAppName('ctrModel').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Change to your own filepath
    file_path = 'file:///Users/smzdm/IdeaProjects/SparrowRecSys/src/main/resources'
    rawSampleDataPath = file_path + "/webroot/sampledata/ratings.csv"

    embLength = 10

    print("==============processItemSequence==============")
    # 生成若干movie的label=1打分序列
    samples = processItemSequence(spark, rawSampleDataPath)
    samples.show(10, False)

    print("==============trainItem2vec==============")
    # word2vec训练电影emb
    model = trainItem2vec(spark, samples, embLength,
                          embOutputPath=file_path[7:] + "/webroot/modeldata2/item2vecEmb.csv", saveToRedis=False,
                          redisKeyPrefix="i2vEmb")

    print("==============generateUserEmb==============")
    # 根据电影emb生成用户emb
    generateUserEmb(spark, rawSampleDataPath, model, embLength,
                    embOutputPath=file_path[7:] + "/webroot/modeldata2/userEmb.csv", saveToRedis=False,
                    redisKeyPrefix="uEmb")

    print("==============graphEmb==============")
    # 随机游走生成电影序列
    graphEmb(samples, spark, embLength, embOutputFilename=file_path[7:] + "/webroot/modeldata2/itemGraphEmb.csv",
             saveToRedis=True, redisKeyPrefix="graphEmb")

