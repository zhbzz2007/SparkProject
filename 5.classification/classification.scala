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
