from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as F

NUMBER_PRECISION = 2


def addSampleLabel(ratingSamples):
    # 记录总数
    sampleCount = ratingSamples.count()

    # 统计各个分值的出现比例
    ratingSamples.groupBy('rating').count().orderBy('rating').withColumn('percentage',
                                                                         F.col('count') / sampleCount).show()
    # 将>=3.5的作为label=1
    ratingSamples = ratingSamples.withColumn('label', when(F.col('rating') >= 3.5, 1).otherwise(0))
    return ratingSamples

# 从电影标题中解析出发布年份
def extractReleaseYearUdf(title):
    # add realease year
    if not title or len(title.strip()) < 6:
        return 1990
    else:
        yearStr = title.strip()[-5:-1]
    return int(yearStr)

# 添加电影侧特征
def addMovieFeatures(movieSamples, ratingSamplesWithLabel):
    ############### 电影侧特征 ##################
    # add movie basic features
    # 评分表左连接电影基础信息表（TODO: 怎么丢弃left join没有匹配到的行？)
    samplesWithMovies1 = ratingSamplesWithLabel.join(movieSamples, on=['movieId'], how='left')
    samplesWithMovies1.show(10, False)

    # add releaseYear,title
    # 走udf从title中分离出年份
    samplesWithMovies2 = samplesWithMovies1.withColumn('releaseYear',
                                                       udf(extractReleaseYearUdf, IntegerType())('title')) \
        .withColumn('title', udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
        .drop('title')
    samplesWithMovies2.show(10, False)

    # split genres
    # 将电影分类列按|切开，生成3列特征
    samplesWithMovies3 = samplesWithMovies2.withColumn('movieGenre1', split(F.col('genres'), "\\|")[0]) \
        .withColumn('movieGenre2', split(F.col('genres'), "\\|")[1]) \
        .withColumn('movieGenre3', split(F.col('genres'), "\\|")[2])
    samplesWithMovies3.show(10, False)

    ##################### 电影历史统计信息 ################
    # add rating features
    # 1）groupby 电影，统计电影的被评分次数，平均评分（保留2位小数），评分标准差
    # 2）填充空值为0（todo: 哪一列有null呢？）
    # 3）将标准差保留2位小数
    movieRatingFeatures = samplesWithMovies3.groupBy('movieId').agg(
        F.count(F.lit(1)).alias('movieRatingCount'),
        format_number(F.avg(F.col('rating')),NUMBER_PRECISION).alias('movieAvgRating'),
        format_number(F.stddev(F.col('rating')), NUMBER_PRECISION).alias('movieRatingStddev')
    )
    movieRatingFeatures.show(10,False)

    # join movie rating features
    # 把电影历史评分特征join到样本上
    samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, on=['movieId'], how='left')
    samplesWithMovies4.printSchema()
    samplesWithMovies4.show(5, truncate=False)
    return samplesWithMovies4

# 统计N部电影的分类出现次数，从大到小排序返回分类
def extractGenres(genres_list):
    '''
    pass in a list which format like ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    count by each genre，return genre_list in reverse order
    eg:
    (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
    return:['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    '''
    genres_dict = defaultdict(int)
    for genres in genres_list:
        for genre in genres.split('|'):
            genres_dict[genre] += 1
    sortedGenres = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sortedGenres]


# 追加用户侧特征
def addUserFeatures(samplesWithMovieFeatures):
    extractGenresUdf = udf(extractGenres, ArrayType(StringType()))

    '''
     userPositiveHistory： 样本出现之前该用户喜欢的电影ID列表，按最近到最远排序
     userRatedMovie1~5：基于userPositiveHistory生成，最近访问的5部电影
     userRatingCount：最近评分的次数
     userAvgReleaseYear：最近电影平均年份
     userReleaseYearStddev：最近电影年份标准差
     userAvgRating：最近电影平均评分
     userRatingStddev：最近电影评分标准差
     userGenres：最近喜欢电影的分类出现次数统计，返回从高到低频率的分类
     userGenre1~5：从userGenres取top 5分类
     
     保留userRatingCount大于1的样本
    '''
    samplesWithUserFeatures = samplesWithMovieFeatures \
        .withColumn('userPositiveHistory',
                    F.collect_list(when(F.col('label') == 1, F.col('movieId')).otherwise(F.lit(None))).over(
                        sql.Window.partitionBy("userId").orderBy(F.col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userPositiveHistory", reverse(F.col("userPositiveHistory"))) \
        .withColumn('userRatedMovie1', F.col('userPositiveHistory')[0]) \
        .withColumn('userRatedMovie2', F.col('userPositiveHistory')[1]) \
        .withColumn('userRatedMovie3', F.col('userPositiveHistory')[2]) \
        .withColumn('userRatedMovie4', F.col('userPositiveHistory')[3]) \
        .withColumn('userRatedMovie5', F.col('userPositiveHistory')[4]) \
        .withColumn('userRatingCount',
                    F.count(F.lit(1)).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn('userAvgReleaseYear', F.avg(F.col('releaseYear')).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)).cast(IntegerType())) \
        .withColumn('userReleaseYearStddev', F.stddev(F.col("releaseYear")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userReleaseYearStddev", format_number(F.col("userReleaseYearStddev"), NUMBER_PRECISION)) \
        .withColumn("userAvgRating", format_number(F.avg(F.col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)), NUMBER_PRECISION)) \
        .withColumn("userRatingStddev", F.stddev(F.col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userRatingStddev", format_number(F.col("userRatingStddev"), NUMBER_PRECISION)) \
        .withColumn("userGenres", extractGenresUdf(F.collect_list(when(F.col('label') == 1, F.col('genres')).otherwise(F.lit(None))).over(
            sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)))) \
        .withColumn("userGenre1", F.col("userGenres")[0]) \
        .withColumn("userGenre2", F.col("userGenres")[1]) \
        .withColumn("userGenre3", F.col("userGenres")[2]) \
        .withColumn("userGenre4", F.col("userGenres")[3]) \
        .withColumn("userGenre5", F.col("userGenres")[4]) \
        .drop("genres", "userGenres", "userPositiveHistory") \
        .filter(F.col("userRatingCount") > 1)

    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(10)
    # 挑选1个用户看一下
    samplesWithUserFeatures.filter(samplesWithMovieFeatures['userId'] == 1).orderBy(F.col('timestamp').asc()).show(
        truncate=False)
    return samplesWithUserFeatures

# 随机拆分训练集和测试集
def splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1)
    training, test = smallSamples.randomSplit((0.8, 0.2))
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)

# 按时间切训练集和测试集，避免穿越问题
def splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1).withColumn("timestampLong", F.col("timestamp").cast(LongType()))

    # 采80%分位的时间戳
    quantile = smallSamples.stat.approxQuantile("timestampLong", [0.8], 0.05)
    splitTimestamp = quantile[0]

    # 取小于该时间的样本作为训练
    training = smallSamples.where(F.col("timestampLong") <= splitTimestamp).drop("timestampLong")
    # 取大于该时间的为验证
    test = smallSamples.where(F.col("timestampLong") > splitTimestamp).drop("timestampLong")

    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)


if __name__ == '__main__':
    # spark会话
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    file_path = 'C:/Users/home/IdeaProjects/SparrowRecSys/src/main/resources'
    movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"
    ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"

    # 加载movie基础表
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    # 加载rating评分表
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)

    # 为rating打分样本计算标签（0：不喜欢，1：喜欢），阈值是人眼观察的
    print("==============addSampleLabel==============")
    ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate=False)

    # 为样本丰富movie侧特征
    print("==============addMovieFeatures==============")
    samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)

    # 为样本丰富user侧特征
    print("==============addUserFeatures==============")
    samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)

    # save samples as csv format
    # 保存数据集
    print("==============splitAndSaveTrainingTestSamples==============")
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path + "/webroot/sampledata")
    # splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path + "/webroot/sampledata")
