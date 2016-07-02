#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: GBK -*-

# 1.加载和查看数据集
from pyspark import SparkContext,SparkConf
conf = SparkConf().setAppName("regression")
sc = SparkContext(conf=conf)

path = "/home/zhb/Desktop/work/SparkData/regression/Bike-Sharing-Dataset/hour_noheader.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x:x.split(","))
first = records.first()
print first
print num_data

# 对数据进行缓存
records.cache()

# 将特征值映射到二元向量中非0的位置，首先将第idx列的特征值蛆虫，然后对每个值使用zipWithIndex函数映射
# 到一个唯一的索引，就组成了RDD键值对，键是变量，值是索引，索引就是特征在二元变量中对应的非0的位置，最后将
# RDD表示为Python的字典内型
def get_mapping(rdd,idx):
    return rdd.map(lambda fields:fields[idx]).distinct().zipWithIndex().collectAsMap()

print "Mapping of first categorical feature column: %s" % get_mapping(records,2)

# 对类型变量的列（2-9列）应用该函数
mappings = [get_mapping(records,i) for i in range(2,10)]
cat_len = sum(map(len,mappings))
num_len = len(records.first()[11:15])
total_len = cat_len + num_len

print "Feature vector length for categorical features: %d" % cat_len
print "Feature vector length for numerical features: %d" % num_len
print "Total feature vector length: %d" % total_len

# 为线性模型创建特征向量
from pyspark.mllib.regression import LabeledPoint
import numpy as np

# 遍历数据的每一行每一列，根据已创建的映射对每个特征进行二元编码，
# step变量用来确保非0特征在整个特征向量中位于正确的位置，数值向量直接对之前
# 已经被转换成浮点数的数据用numpy的array进行封装，最后将二元向量和数值向量拼接起来
def extract_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[2:9]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)
    num_vec = np.array( [float(field) for field in record[10:14]] )
    return np.concatenate((cat_vec,num_vec))

# 将数据中的最后一列cnt的数据转换成浮点数
def extract_label(record):
    return float(record[-1])

data = records.map(lambda r:LabeledPoint(extract_label(r), extract_features(r)))
first_point = data.first()
print "Raw data: " + str(first[2:])
print "Label: " + str(first_point.label)
print "Linear Model feature vector:\n" + str(first_point.features)
print "Linear Model feature vectors length: " + str(len(first_point.features))

# 为决策树创建特征向量
def extract_features_dt(record):
	return np.array(map(float,record[2:14]))

data_dt = records.map(lambda r:LabeledPoint(extract_label(r),extract_features_dt(r)))
first_point_dt = data_dt.first()
print "Raw data: " + str(first[2:])
print "Decision Tree feature vector:\n" + str(first_point_dt.features)
print "Decision Tree feature vectors length: " + str(len(first_point_dt.features))

# 2.回归模型的训练和使用
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
help(LinearRegressionWithSGD.train)
help(DecisionTree.trainRegressor)

# 训练线性模型
linear_model = LinearRegressionWithSGD.train(data,iterations = 10, step = 0.1,intercept = False)
true_vs_predicted = data.map(lambda p:(p.label,linear_model.predict(p.features)))
print "Linear Model predictions: " + str(true_vs_predicted.take(5))

# 训练决策树
dt_model = DecisionTree.trainRegressor(data_dt,{})
preds = dt_model.predict(data_dt.map(lambda p:p.features))
actual = data.map(lambda p:p.label)
true_vs_predicted_dt = actual.zip(preds)
print "Decision Tree predictions: " + str(true_vs_predicted_dt.take(5))
print "Decision Tree depth: " + str(dt_model.depth())
print "Decision Tree number of nodes: " + str(dt_model.numNodes())

# 3.评估回归模型的性能
# 计算MSE
def squared_error(actual,pred):
    return (pred - actual)**2

# 计算MAE
def abs_error(actual,pred):
    return np.abs(pred - actual)

# 计算RMSLE
def squared_log_error(pred,actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2

# 线性模型
mse = true_vs_predicted.map(lambda (t,p) : squared_error(t,p)).mean()
mae = true_vs_predicted.map(lambda (t,p) : abs_error(t,p)).mean()
rmsle = true_vs_predicted.map(lambda (t,p) : squared_log_error(t,p)).mean()

print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Error: %2.4f" % rmsle

# 决策树

mse_dt = true_vs_predicted_dt.map(lambda (t,p) : squared_error(t,p)).mean()
mae_dt = true_vs_predicted_dt.map(lambda (t,p) : abs_error(t,p)).mean()
rmsle_dt = true_vs_predicted_dt.map(lambda (t,p) : squared_log_error(t,p)).mean()

print "Decision Tree - Mean Squared Error: %2.4f" % mse_dt
print "Decision Tree - Mean Absolute Error: %2.4f" % mae_dt
print "Decision Tree - Root Mean Squared Error: %2.4f" % rmsle_dt

# 4.改进模型性能和参数调优
