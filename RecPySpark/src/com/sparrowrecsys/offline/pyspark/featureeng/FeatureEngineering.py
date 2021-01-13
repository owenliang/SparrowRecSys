from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F

# 演示one-hot编码
def oneHotEncoderExample(movieSamples):
    # 增加一列movieIdNumber整数电影ID列
    samplesWithIdNumber = movieSamples.withColumn("movieIdNumber", F.col("movieId").cast(IntegerType()))

    # pyspark.ml.feature 特征工程，movieIdNumber列做OneHot编码到movieIdVector
    # dropLast=True会导致使用全0的onehot向量表示最后一个category，这不适合我们
    # 一共1000部电影，之所以有1001列onehot是因为没有ID为0的电影
    encoder = OneHotEncoder(inputCols=["movieIdNumber"], outputCols=['movieIdVector'], dropLast=False)

    # 执行转换
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)

    # 打印schema
    oneHotEncoderSamples.printSchema()
    # 打印数据
    oneHotEncoderSamples.show(10, False)

# UDF函数，
def array2vec(genreIndexes, indexSize):
    genreIndexes.sort()
    fill_list = [1.0 for _ in range(len(genreIndexes))]
    return Vectors.sparse(indexSize, genreIndexes, fill_list) # 稀疏向量(总长度,有效位置,填充值）

# 电影分类的multi-hot编码
def multiHotEncoderExample(movieSamples):
    # 将genres按|切分成数组，然后用explode拆成多行结构
    samplesWithGenre = movieSamples.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))

    samplesWithGenre.show(10, False)

    # 为genre列的值编码0~N
    genreIndexer = StringIndexer(inputCol="genre", outputCol="genreIndex")  # 输出genreIndex是float类型，做multihot之前需要转int
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn("genreIndexInt",
                                                                                  F.col("genreIndex").cast(IntegerType()))

    # 统计一共有多少个genre分类
    indexSize = genreIndexSamples.agg(F.max(F.col("genreIndexInt"))).head()[0] + 1

    # 按电影ID聚合，拉链genreIndexInt得到genreIndexes数组，补上一列indexSize
    processedSamples = genreIndexSamples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes'))\
        .withColumn("indexSize", F.lit(indexSize))
    processedSamples.show(10, False)

    # 将genreIndexes数组做multihot编码，保存到vector
    finalSample = processedSamples.withColumn("vector", # 使用udf自定义函数，udf第一个参数是回调函数，第二个参数是返回值类型
                                              udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    finalSample.printSchema()
    finalSample.show(10, False)


def ratingFeatures(ratingSamples):
    ratingSamples.printSchema()
    ratingSamples.show()

    # calculate average movie rating score and rating count
    # 按movieId做聚合，统计电影点击次数count(1) as ratingCount
    # avg(rating) as avgRating
    # variance(rating) as ratingVar   -- 这个是方差
    movieFeatures = ratingSamples.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("rating").alias("avgRating"),
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))  # 把平均得分转成只有1列的向量存储，后续做标准化要求的
    movieFeatures.show(10)

    ######## 走pipeline特征处理 ########
    # bucketing
    # 连续值分桶：对ratingCount按分布划分成100个大小一样的桶
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")
    # Normalization
    # 标准化：将平均得分向量进行标准化
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")

    # 创建pipeline
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    featurePipeline = Pipeline(stages=pipelineStage)
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)

    # 把分桶转成整数类型, 把标准化的向量提取为非向量
    movieProcessedFeatures = movieProcessedFeatures.withColumn('ratingCountBucket', F.col('ratingCountBucket').cast(IntegerType()))\
        .withColumn('scaleAvgRating', udf(lambda v: float(v[0]), FloatType())(F.col('scaleAvgRating'))).drop(F.col('avgRatingVec'))
    movieProcessedFeatures.show(10)

if __name__ == '__main__':
    # 连接spark
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # 读movie基础信息到dataframe
    file_path = 'file:///Users/smzdm/IdeaProjects/SparrowRecSys/src/main/resources'
    movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)

    # 显示表格
    print("==============Raw Movie Samples:==============")
    movieSamples.show(10, False)
    # 打印schema（CSV加载后所有列都是字符串）
    movieSamples.printSchema()

    print("==============OneHotEncoder Example:==============")
    oneHotEncoderExample(movieSamples)

    print("==============MultiHotEncoder Example:==============")
    multiHotEncoderExample(movieSamples)

    print("==============Numerical features Example:==============")
    ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingFeatures(ratingSamples)
