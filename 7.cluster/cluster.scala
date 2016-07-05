// 1.提取数据
// 获取电影数据集
val itemFilePath = "/home/zhb/Desktop/work/SparkData/ml-100k/u.item"
val movies = sc.textFile(itemFilePath)
println(movies.first)

// 获取电影的题材标签
val genreFilePath = "/home/zhb/Desktop/work/SparkData/ml-100k/u.genre"
val genres = sc.textFile(genreFilePath)
genres.take(5).foreach(println)

// 提取题材的映射关系，对每一行进行分割，得到具体的<题材，索引>键值对，需要处理最后的空行，不然会抛出异常
val genreMap = genres.filter(!_.isEmpty).map(line => line.split("\\|")).map(array => (array(1),array(0))).collectAsMap
println(genreMap)

val titlesAndGenres = movies.map(_.split("\\|")).map{array =>
    val genres = array.toSeq.slice(5,array.size)
    val genresAssigned = genres.zipWithIndex.filter{ case (g,idx)
        => g == "1"
    }.map{case (g,idx) => genreMap(idx.toString)}
    (array(0).toInt,(array(1),genresAssigned))
}
println(titlesAndGenres.first)

// 训练推荐模型
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
val userFilePath = "/home/zhb/Desktop/work/SparkData/ml-100k/u.data"
val rawData = sc.textFile(userFilePath)
val rawRatings = rawData.map(_.split("\t").take(3))
val ratings = rawRatings.map{case Array(user,movie,rating) =>
    Rating(user.toInt,movie.toInt,rating.toDouble) }
ratings.cache
val alsModel = ALS.train(ratings,50,10,0.1)

// 对用户和电影分别进行处理
import org.apache.spark.mllib.linalg.Vectors
val movieFactors = alsModel.productFeatures.map{case (id,factor) => (id,Vectors.dense(factor))}
val movieVectors = movieFactors.map(_._2)
val userFactors = alsModel.userFeatures.map{case (id,factor) => (id,Vectors.dense(factor))}
val userVectors = userFactors.map(_._2)

// 归一化
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val movieMatrix = new RowMatrix(movieVectors)
val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
val userMatrix = new RowMatrix(userVectors)
val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
println("movie factor mean: " + movieMatrixSummary.mean)
println("movie factor variance " + movieMatrixSummary.variance)
println("User factor mean: " + userMatrixSummary.mean)
println("User factor variance " + userMatrixSummary.variance)

// 2.训练聚类模型
import org.apache.spark.mllib.clustering.KMeans
val numClusters = 5
val numIterations = 10
val numRuns = 3

// 对电影的系数向量运行K均值算法
val movieClusterModel = KMeans.train(movieVectors,numClusters,numIterations,numRuns)
// 设置更大的迭代次数使得K均值模型收敛
val movieClusterModelConverged = KMeans.train(movieVectors,numClusters,numIterations,100)
// 在用户相关因素的特征向量上训练K均值模型
val userClusterModel = KMeans.train(userVectors,numClusters,numIterations,numRuns)

// 3.使用聚类模型进行预测
val movie1 = movieVectors.first
val movieCluster = movieClusterModel.predict(movie1)
println(movieCluster)

val predictions = movieClusterModel.predict(movieVectors)
println(predictions.take(10).mkString(","))

import breeze.linalg._
import breeze.numerics.pow
def computerDistance(v1:DenseVector[Double],v2:DenseVector[Double])=pow(v1-v2,2).sum

val titlesWithFactors = titlesAndGenres.join(movieFactors)
def moviesAssigned = titlesWithFactors.map{ case (id,((title,genres),vector)) =>
    val pred = movieClusterModel.predict(vector)
    val clusterCentre = movieClusterModel.clusterCenters(pred)
    val dist = computerDistance(DenseVector(clusterCentre.toArray),DenseVector(vector.toArray))
    (id,title,genres.mkString(" "),pred,dist)
}
val clusterAssignments = moviesAssigned.groupBy{ case (id,title,genres,cluster,dist) => cluster}.collectAsMap

for ((k,v) <- clusterAssignments.toSeq.sortBy(_._1)){
    println(s"Cluster $k:")
    val m = v.toSeq.sortBy(_._5)
    println(m.take(20).map{case (_,title,genres,_,d) => (title,genres,d)}.mkString("\n") )
    println("=====\n")
}

val movieCost = movieClusterModel.computeCost(movieVectors)
val userCost = userClusterModel.computeCost(userVectors)
println("WCSS for movies: " + movieCost)
println("WCSS for users: " + userCost)

// 4.聚类模型参数调优
val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6,0.4),123)
val trainMovies = trainTestSplitMovies(0)
val testMovies = trainTestSplitMovies(1)
val costsMovies = Seq(2,3,4,5,10,20).map{ k => (k,KMeans.train(trainMovies,numIterations,k,numRuns).computeCost(testMovies))}
println("Movie clustering cross-validation:")
costsMovies.foreach{ case (k,cost) => println(f"WCSS for K=$k id $cost%2.2f")}

// 用户聚类在交叉验证下的性能
val trainTestSplitUsers = userVectors.randomSplit(Array(0.6,0.4),123)
val trainUsers = trainTestSplitUsers(0)
val testUsers = trainTestSplitUsers(1)
val costsUsers = Seq(2,3,4,5,10,20).map{ k => (k,KMeans.train(trainUsers,numIterations,k,numRuns).computeCost(testUsers))}
println("User clustering cross-validation:")
costsUsers.foreach{ case (k,cost) => println(f"WCSS for K=$k id $cost%2.2f")}
