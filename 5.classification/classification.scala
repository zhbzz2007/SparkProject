// 1.从原始数据中抽取特征
val trainData = "/home/zhb/Desktop/work/SparkData/classification/train.tsv"
val rawData = sc.textFile(trainData)
val records = rawData.map(line => line.split("\t"))
records.first()

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors


val data = records.map{ r =>
    // 将额为的(")去掉
    val trimmed = r.map(_.replaceAll("\"",""))
    // 提取标签数据
    val label = trimmed(r.size - 1).toInt
    // 数据集中还有一些用"?"代替缺失数据，直接用0替换那些缺失数据
    // 第5列到倒数第二列作为特征
    val features = trimmed.slice(4,r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
    LabeledPoint(label,Vectors.dense(features))
}
// 对数据进行缓存
data.cache
// 统计数据样本的数目
val numData = data.count

// 朴素贝叶斯要求特征值非负，否则碰到负的特征值程序会抛出错误，因此需要为朴素贝叶斯模型
// 构建一份输入特征向量的数据，将负特征值设置为0
val nbData = records.map{ r =>
    // 将额为的(")去掉
    val trimmed = r.map(_.replaceAll("\"",""))
    // 提取标签数据
    val label = trimmed(r.size - 1).toInt
    // 数据集中还有一些用"?"代替缺失数据，直接用0替换那些缺失数据，并将负特征值设置为0
    // 第5列到倒数第二列作为特征
    val features = trimmed.slice(4,r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if(d < 0) 0.0 else d)
    LabeledPoint(label,Vectors.dense(features))
}


// 2.训练分类模型
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
val numIterations = 10
val maxTreeDepth = 5

// 训练逻辑回归模型
val lrModel = LogisticRegressionWithSGD.train(data,numIterations)

// 训练SVM模型
val svmModel = SVMWithSGD.train(data,numIterations)

// 训练朴素贝叶斯模型
val nbModel = NaiveBayes.train(nbData)

// 训练决策树模型
val dtModel = DecisionTree.train(data,Algo.Classification,Entropy,maxTreeDepth)


// 3.使用分类模型
val dataPoint = data.first
// 使用分类模型进行预测
val prediction = lrModel.predict(dataPoint.features)
// 样本真正的标签
val trueLabel = dataPoint.label
// 将整个RDD[Vector]作为输入做预测
val predictions = lrModel.predict(data.map(lp => lp.features))
predictions.take(5)


// 4.评估分类模型的性能
// 4.1预测的正确率和错误率
// 逻辑回归模型
val lrTotalCorrect = data.map{point =>
if (lrModel.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracy = lrTotalCorrect / numData

// svm模型
val svmTotalCorrect = data.map{point =>
if (svmModel.predict(point.features) == point.label) 1 else 0
}.sum
val svmAccuracy = svmTotalCorrect / numData

// 朴素贝叶斯模型
val nbTotalCorrect = data.map{point =>
if (nbModel.predict(point.features) == point.label) 1 else 0
}.sum
val nbAccuracy = nbTotalCorrect / numData

// 决策树模型
val dtTotalCorrect = data.map{point =>
    val score = dtModel.predict(point.features)
    val predicted = if (score > 0.5) 1 else 0
    if (predicted == point.label) 1 else 0
}.sum
val dtAccuracy = dtTotalCorrect / numData

// 4.2准确率和召回率
// 4.3ROC曲线和AUC
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
val metrics = Seq(lrModel,svmModel).map{ model =>
    val scoreAndLabels = data.map{ point =>
    (model.predict(point.features),point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName,metrics.areaUnderPR,metrics.areaUnderROC)
}

val nbmetrics = Seq(nbModel).map{ model =>
    val scoreAndLabels = nbData.map{ point =>
    val score = model.predict(point.features)
    (if (score > 0.5) 1.0 else 0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName,metrics.areaUnderPR,metrics.areaUnderROC)
}

val dtmetrics = Seq(dtModel).map{ model =>
   val scoreAndLabels = data.map{ point =>
   val score = model.predict(point.features)
   (if (score > 0.5) 1.0 else 0, point.label)
   }
   val metrics = new BinaryClassificationMetrics(scoreAndLabels)
   (model.getClass.getSimpleName,metrics.areaUnderPR,metrics.areaUnderROC)
}

val allMetrics = metrics ++ nbmetrics ++ dtmetrics
allMetrics.foreach{case (m,pr,roc) =>
println(f"$m, Area under PR : ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
}

// 5.改进模型性能以及参数调优
// 5.1特征标准化
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val vectors = data.map(lp => lp.features)
val matrix = new RowMatrix(vectors)
val matrixSummary = matrix.computeColumnSummaryStatistics()
// 输出矩阵每列的均值
println(matrixSummary.mean)
// 输出矩阵每列的最小值
println(matrixSummary.min)
// 输出矩阵每列的最大值
println(matrixSummary.max)
// 输出矩阵每列的方差
println(matrixSummary.variance)
// 输出矩阵每列的非零项的数目
println(matrixSummary.numNonzeros)

import org.apache.spark.mllib.feature.StandardScaler
// 传入两个参数，一个表示是否从数据中减去均值，一个表示是否应用标准差缩放
val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
val scaledData = data.map(lp => LabeledPoint(lp.label,scaler.transform(lp.features)))
println(data.first.features)
println(scaledData.first.features)

// 使用标准化的数据重新训练模型，这里只训练逻辑回归模型，因为决策树和朴素贝叶斯不受特征标准化的影响
val lrModelScaled = LogisticRegressionWithSGD.train(scaledData,numIterations)
val lrTotalCorrectScaled = scaledData.map{ point =>
    if(lrModelScaled.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaled = lrTotalCorrectScaled / numData
val lrPredictionsVsTrue = scaledData.map{ point =>
    (lrModelScaled.predict(point.features), point.label)
}
val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
val lrPr = lrMetricsScaled.areaUnderPR
val lrRoc = lrMetricsScaled.areaUnderROC
println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy:${lrAccuracyScaled * 100}%2.4f%%\nArea under PR : ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")


// 5.2评估类别特征对性能的影响
val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
val numCategories = categories.size
println(categories)
println(numCategories)

// 创建一个长度为14的向量表示类别特征，根据每个样本所属类别索引，对相应的维度赋值为1，其他为0
val dataCategories = records.map{ r =>
    val trimmed = r.map(_.replace("\"",""))
    val label = trimmed(r.size - 1).toInt
    val categoryIdx = categories(r(3))
    val categoryFeatures = Array.ofDim[Double](numCategories)
    categoryFeatures(categoryIdx) = 1.0
    val otherFeatures = trimmed.slice(4,r.size - 1).map(d =>
    if (d == "?") 0.0 else d.toDouble)
    val features = categoryFeatures ++ otherFeatures
    LabeledPoint(label,Vectors.dense(features))
}
println(dataCategories.first)

// 对数据集进行标准化转换
val scalerCats = new StandardScaler(withMean = true,withStd = true).fit(dataCategories.map(lp => lp.features))
val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label,scalerCats.transform(lp.features)))
println(dataCategories.first.features)
println(scaledDataCats.first.features)

// 用扩展后的特征来训练新的逻辑回归模型，然后评估其性能
val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats,numIterations)
val lrTotalCorrectScaledCats = scaledDataCats.map{point =>
    if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
val lrPredictionsVsTrueCats = scaledDataCats.map{point =>
    (lrModelScaledCats.predict(point.features),point.label)
}
val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
val lrPrCats = lrMetricsScaledCats.areaUnderPR
val lrRocCats = lrMetricsScaledCats.areaUnderROC
println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy:${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")

// 5.3使用正确的数据格式
val dataNB = records.map{ r=>
    val trimmed = r.map(_.replaceAll("\"",""))
    val label = trimmed(r.size - 1).toInt
    val categoryIdx = categories(r(3))
    val categoryFeatures = Array.ofDim[Double](numCategories)
    categoryFeatures(categoryIdx) = 1.0
    LabeledPoint(label,Vectors.dense(categoryFeatures))
}

val nbModelCats = NaiveBayes.train(dataNB)
val nbTotalCorrectCats = dataNB.map{ point =>
    if (nbModelCats.predict(point.features) == point.label) 1 else 0
}.sum
val nbAccuracyCats = nbTotalCorrectCats / numData
val nbPredictionsVsTrueCats = dataNB.map{point =>
    (nbModelCats.predict(point.features),point.label)
}
val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
val nbPrCats = nbMetricsCats.areaUnderPR
val nbRocCats = nbMetricsCats.areaUnderROC
println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy:${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%")

// 5.4 模型参数调优
// 1.线性模型
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.classification.ClassificationModel

// 定义辅助函数，根据给定输入训练模型
def trainWithParams(input: RDD[LabeledPoint],regParam:Double,numIterations:Int,updater:Updater,stepSize:Double) = {
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    lr.run(input)
}

// 根据输入数据和分类模型，计算相关的AUC
def createMetrics(label:String,data:RDD[LabeledPoint],model:ClassificationModel) = {
    val scoreAndLabels = data.map{ point =>
    (model.predict(point.features),point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label,metrics.areaUnderROC)
}

// 为了加快多次模型的训练，可以缓存标准化的数据，包括类别信息
scaledDataCats.cache

// (1)迭代次数
val iterResults = Seq(1,5,10,50).map{ param =>
    val model = trainWithParams(scaledDataCats,0.0,param,new SimpleUpdater,1.0)
    createMetrics(s"$param iterations",scaledDataCats,model)
}

iterResults.foreach{ case(param,auc) => println(f"$param,AUC=${auc * 100}%2.2f%%")}

// (2)步长
val stepResults = Seq(0.001,0.01,0.1,1.0,10.0).map{ param =>
    val model = trainWithParams(scaledDataCats,0.0,numIterations,new SimpleUpdater,param)
    createMetrics(s"$param step size",scaledDataCats,model)
}

stepResults.foreach{ case(param,auc) => println(f"$param,AUC=${auc * 100}%2.2f%%")}

// (3)正则化
val regResults = Seq(0.001,0.01,0.1,1.0,10.0).map{ param =>
    val model = trainWithParams(scaledDataCats,param,numIterations,new SquaredL2Updater,1.0)
    createMetrics(s"$param L2 regularization parameter",scaledDataCats,model)
}

regResults.foreach{ case(param,auc) => println(f"$param,AUC=${auc * 100}%2.2f%%")}


// 2.决策树
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Gini

// 创建辅助函数
def trainDTWithParams(input:RDD[LabeledPoint],maxDepth:Int,impurity:Impurity) = {
    DecisionTree.train(input,Algo.Classification,impurity,maxDepth)
}

// Entropy
val dtResultEntropy = Seq(1,2,3,4,5,10,20).map {param =>
    val model = trainDTWithParams(data,param,Entropy)
    val scoreAndLabels = data.map{point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$Entropy param tree depth", metrics.areaUnderROC)
}
dtResultEntropy.foreach{ case(param,auc) => println(f"$param,AUC = ${auc * 100}%2.2f%%")}

// Gini
val dtResultGini = Seq(1,2,3,4,5,10,20).map {param =>
    val model = trainDTWithParams(data,param,Gini)
    val scoreAndLabels = data.map{point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$Gini param tree depth", metrics.areaUnderROC)
}
dtResultGini.foreach{ case(param,auc) => println(f"$param,AUC = ${auc * 100}%2.2f%%")}


// 3.朴素贝叶斯
// 创建辅助函数，用来训练不同lambda级别下的模型
def trainNBWithParams(input:RDD[LabeledPoint],lambda:Double) = {
    val nb = new NaiveBayes
    nb.setLambda(lambda)
    nb.run(input)
}

//
val nbResults = Seq(0.001,0.01,0.1,1.0,10.0).map {param =>
    val model = trainNBWithParams(dataNB,param)
    val scoreAndLabels = dataNB.map{point =>
        (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param tree depth", metrics.areaUnderROC)
}
nbResults.foreach{ case(param,auc) => println(f"$param,AUC = ${auc * 100}%2.2f%%")}


// 4.交叉验证
val trainTestSplit = scaledDataCats.randomSplit(Array(0.6,0.4),123)
val train = trainTestSplit(0)
val test = trainTestSplit(1)

val regResultsTest = Seq(0.0,0.001,0.0025,0.005,0.01).map{ param =>
    val model = trainWithParams(train,param,numIterations,new SquaredL2Updater,1.0)
    createMetrics(s"$param L2 regularization parameter",test,model)
}

regResultsTest.foreach{ case(param,auc) => println(f"$param,AUC=${auc * 100}%2.6f%%")}
