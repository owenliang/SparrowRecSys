from pyspark import SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F

# 矩阵分解：得到用户emb和电影emb
if __name__ == '__main__':
    # 连接spark
    conf = SparkConf().setAppName('collaborativeFiltering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    #/Users/zhewang/Workspace/SparrowRecSys/src/main/resources/webroot/modeldata

    # 加载csv为dataframe
    file_path = 'file:///Users/smzdm/IdeaProjects/SparrowRecSys/src/main/resources'
    ratingResourcesPath = file_path + '/webroot/sampledata/ratings.csv'
    # 在原dataframe增加了3列转换类型后的列
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingResourcesPath) \
        .withColumn("userIdInt", F.col("userId").cast(IntegerType())) \
        .withColumn("movieIdInt", F.col("movieId").cast(IntegerType())) \
        .withColumn("ratingFloat", F.col("rating").cast(FloatType()))

    print('==============ratingSamples==============')
    ratingSamples.show()

    # 切分成80%的训练集和20%的测试集
    training, test = ratingSamples.randomSplit((0.8, 0.2),seed=24)

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

    # 矩阵分解model，需要user*item共现矩阵
    als = ALS(regParam=0.01, maxIter=5, userCol='userIdInt', itemCol='movieIdInt', ratingCol='ratingFloat',
              numUserBlocks=10, numItemBlocks=10, coldStartStrategy='drop') # drop表示在预测时，对于没见过的user id和movie id直接跳过
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    # 预测一组user*item的评分
    predictions = model.transform(test)

    print('==============predictions（{}/{}）=============='.format(predictions.filter(predictions.prediction.isNotNull()).count(), test.count()))
    predictions.show()

    # 输出所有物品emb向量
    print('==============model.itemFactors==============')
    model.itemFactors.show(10, truncate=False)
    # 输出所有用户emb向量
    print('==============model.userFactors==============')
    model.userFactors.show(10, truncate=False)

    # 计算预测误差（预测打分-真实打分，属于回归的误差计算方式）
    print('==============RegressionEvaluator.evaluate==============')
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='ratingFloat', metricName='rmse')
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = {}".format(rmse))

    # Generate top 10 movie recommendations for each user
    # 为每个用户计算10个最高分的movie，经过用户矩阵*物品矩阵可以得到共现矩阵，直接基于共现矩阵打分就可以为每个用户挑出TOP N
    print('==============recommendForAllUsers==============')
    userRecs = model.recommendForAllUsers(10)
    userRecs.show(5, False)
    # 为每个电影计算最适合推荐给的用户，和上面一样道理
    # Generate top 10 user recommendations for each movie
    print('==============recommendForAllItems==============')
    movieRecs = model.recommendForAllItems(10)
    movieRecs.show(5, False)

    # Generate top 10 movie recommendations for a specified set of users
    # 取3个用户，保留userIdInt列
    print('==============recommendForUserSubset==============')
    users = ratingSamples.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10) # 仅为这三个用户推荐电影
    userSubsetRecs.show(5, False)

    # Generate top 10 user recommendations for a specified set of movies
    # 取3个电影，保留movieIdInt列
    print('==============recommendForUserSubset==============')
    movies = ratingSamples.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10) # 仅为3个电影推荐用户
    movieSubSetRecs.show(5, False)

    # 模型超参数搜索，利用K折交叉验证
    print('==============CrossValidator==============')
    paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.01]).build()
    cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(ratingSamples)

    # 所有超参数枚举的平均误差
    avgMetrics = cvModel.avgMetrics
    print('avgMetrics:', avgMetrics)
    # 最佳模型
    print('bestModel:', cvModel.bestModel)
    cvModel.bestModel.recommendForAllUsers(10).show(10, False)

    spark.stop()