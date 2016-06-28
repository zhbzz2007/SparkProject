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


// 物品推荐
import org.jblas.DoubleMatrix
val aMatrix = new DoubleMatrix(Array(1.0,2.0,3.0))

// 余弦相似度
def cosineSimilarity(vec1:DoubleMatrix, vec2:DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm() * vec2.norm())
}

val itemId = 567
val itemFactor = model.productFeature.lookup(itemId).head
val itemVector = new DoubleMatrix(itemFactor)

// 计算各个商品与指定商品的相似度
val sims = model.productFeature.map{case (id,factor) =>
    val factorVector = new DoubleMatrix(factor)
    val sim = cosineSimilarity(factorVector,itemVector)
    (id,sim)
}

// 对物品按照相似度排序，然后取出最相似的前10个物品
val sortedSims = sims.top(K)(Ordering.by[(Int,Double),Double] {case (id,similarity) => similarity})

// 打印出这K个与给定物品最相似的物品
println(sortedSims.take(10).mkString("\n"))

// 打印出给定的商品名称
println(titles(itemId))

// 取前K+1部最相似电影，以排除给定的那部
val sortedSims2 = sims.topK(K+1)(Ordering.by[(Int,Double),Double] {case (id,similarity) => similarity})
sortedSims2.slice(1,11).map{case (id,sim) => (titles(id),sim)}.mkString("\n")


// 推荐模型效果的评估
// 均方差
// 商品评级实际值
val actualRating = moivesForUser.take(1)(0)
// 商品评级预测值
val predictedRating = model.predict(789,actualRating.product)
// 误差平方和
val squareError = math.pow(predictedRating - actualRating.rating,2)

// 提取用户和商品ID
val usersProducts = ratings.map{ case Rating(user,product,rating) => (user,product) }
// 对各个“用户-商品”对做预测
val predictions = model.predict(usersProducts).map {
    case Rating(user,product,rating) => ((user,product),rating)
}

//提取真实的评级，主键为“用户-商品”对，键值为相应的实际评级和预计评级
val ratingsAndPredictions = ratings.map{
    case Rating(user,product,rating) => ((user,product),rating)
}.join(predictions)

// 先用reduce求和，然后再处以count函数所求得的总记录数
val MSE = ratingsAndPredictions.map{
    case ((user,product),(actual,predicted)) => math.pow((actual - predicted),2)
}.reduce(_ + _) / ratingsAndPredictions.count
println("Mean Squared Error = " + MSE)

// 均方根误差
val RMSE = math.sqrt(MSE)
println("Root Mean Squared Error = " + RMSE)

// K值平均准确率
def avgPrecisionK(actual:Seq[Int],predicted:Seq[Int],k:Int):Double = {
    val predK = predicted.take(K)
    var score = 0.0
    var numHits = 0.0
    for((p,i) <- predK.zipWithIndex){
        if(actual.contains(p)){
            numHits += 1.0
            score += numHits / (i.toDouble + 1.0)
        }
    }
    if (actual.isEmpty)
    {
        1.0
    }else{
        score / scala.math.min(actual.size,k).toDouble
    }
}

// 提取出用户实际评级过的电影的ID
val actualMoives = moivesForUser.map(_.product)

// 提取出推荐的物品列表，K设置为10
val predictedMoives = topKRecs.map(_.product)

// 计算平均准确率
val apk10 = avgPrecisionK(actualMoives,predictedMoives,10)

val itemFactors = model.productFeatures.map{case(id,factor) =>factor}.collect()
val itemMatrix = new DoubleMatrix(itemFactors)
println(itemMatrix.rows,itemMatrix.columns)

val imBroadcast = sc.broadcast(itemMatrix)

val allRecs = model.userFeatures.map{ case (userId,array) =>
    val userVector = new DoubleMatrix(array)
    val scores = imBroadcast.value.mmul(userVector)
    val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
    val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
    (userId,recommendedIds)
}

val userMoives = ratings.map{case Rating(user,product,rating) =>
(user,product)}.groupBy(_._1)

val K = 10
val MAPK = allRecs.join(userMoives).map{case (userId,(predicted,
actualWithIds)) =>
    val actual = actualWithIds.map(_._2).toSeq
    avgPrecisionK(actual,predicted,K)
}.reduce(_ + _) / allRecs.count
println("Mean Average Precision at K = " + MAPK)

// 使用MLlib内置的评估函数
// RMSE和MSE
import org.apache.spark.mllib.evaluation.RegressionMetrics
val predictedAndTrue = ratingsAndPredictions.map{ case ((user,product),(predicted,actual)) => (predicted,actual)}
val regressionMetrics = new RegressionMetrics(predictedAndTrue)
println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
println("Root Mean Squared Error"  + regressionMetrics.rootMeanSquaredError)

// MAP
// 使用RankingMetrics来计算MAP
import org.apache.spark.mllib.evaluation.RankingMetrics
val predictedAndTrueForRanking = allRecs.join(userMoives).map{case
(userId,(predicted,actualWithIds)) =>
    val actual = actualWithIds.map(_._2)
    (predicted.toArray,actual.toArray)
}
val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
println("Mean Average Precision = " + rankingMetrics.meanAveragePrecison)

// 用和之前完全相同的方法来计算MAP，但是将K值设置到很高，比如2000
val MAPK2000 = allRecs.join(userMoives).map{case (userId,(predicted,actualWithIds)) =>
    val actual = actualWithIds.map(_._2).toSeq
    avgPrecisionK(actual,predicted,2000)
}.reduce(_ + _) / allRecs.count
println("Mean average Precison = " + MAPK2000)
