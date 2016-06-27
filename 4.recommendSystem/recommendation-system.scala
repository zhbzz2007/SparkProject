import org.apache.spark.{SparkConf, SparkContext}

val conf = new SparkConf().setAppName("recommendation")
val sc = new SparkContext(conf)

val userFileName = "/home/zhb/Desktop/work/SparkData/ml-100k/u.data"
val rawData = sc.textFile(userFileName)
rawData.first()

val rawRatings = rawData.map(_.split("\t").take(3))
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

val ratings = rawRatings.map{case Array(user,moive,rating) => Rating(user.toInt,moive.toInt,rating.toDouble)}
ratings.first()

// 模型训练
val model = ALS.train(ratings,50,10,0.01)
//预测用户789与商品123的评级
val predictedRating = model.predict(789,123)

//为某个用户生成前K个推荐商品
val userId = 789
val K = 10
val topKRecs = model.recommendProducts(userId,K)
println(topKRecs.mkString("\n"))

val itemFileName = "/home/zhb/Desktop/work/SparkData/ml-100k/u.item"
val moives = sc.textFile(itemFileName)
val titles = moives.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt,array(1))).collectAsMap()
titles(123)

// 用keyBy函数来从ratings RDD来创建一个键值对RDD，主键为用户ID，然后利用lookup函数来只返回给定键值对对应的那些评级数据到驱动程序
val moivesForUser = ratings.keyBy(_.user).lookup(789)
println(moivesForUser.size)

moivesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product),rating.rating)).foreach(println)

topKRecs.map(rating => (titles(rating.product),rating.rating)).foreach(println)
