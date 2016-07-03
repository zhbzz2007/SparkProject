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
rmsle = np.sqrt(true_vs_predicted.map(lambda (t,p) : squared_log_error(t,p)).mean())

print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Error: %2.4f" % rmsle

# 决策树

mse_dt = true_vs_predicted_dt.map(lambda (t,p) : squared_error(t,p)).mean()
mae_dt = true_vs_predicted_dt.map(lambda (t,p) : abs_error(t,p)).mean()
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (t,p) : squared_log_error(t,p)).mean())

print "Decision Tree - Mean Squared Error: %2.4f" % mse_dt
print "Decision Tree - Mean Absolute Error: %2.4f" % mae_dt
print "Decision Tree - Root Mean Squared Error: %2.4f" % rmsle_dt

# 4.改进模型性能和参数调优
import matplotlib.pyplot as plt
import pylab

# 4.1 变换目标变量
targets = records.map(lambda r:float(r[-1])).collect()
pylab.hist(targets,bins=40,color = 'lightblue',normed = True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16,10)

log_targets = records.map(lambda r:np.log(float(r[-1]))).collect()
pylab.hist(log_targets,bins = 40,color = 'lightblue',normed = True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16,10)

sqrt_targets = records.map(lambda r:np.sqrt(float(r[-1]))).collect()
pylab.hist(sqrt_targets,bins = 40,color = 'lightblue',normed = True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16,10)

# 考察线性模型中对数变换的影响
data_log = data.map(lambda lp:LabeledPoint(np.log(lp.label),lp.features))
model_log = LinearRegressionWithSGD.train(data_log,iterations = 10,step = 0.1)
true_vs_predicted_log = data_log.map(lambda p: (np.exp(p.label),np.exp(model_log.predict(p.features))))
# 计算模型的MSE、MAE、RMSLE
mse_log = true_vs_predicted_log.map(lambda (t,p) : squared_error(t,p)).mean()
mae_log = true_vs_predicted_log.map(lambda (t,p) : abs_error(t,p)).mean()
rmsle_log = np.sqrt(true_vs_predicted_log.map(lambda (t,p) : squared_log_error(t,p)).mean())

print "Mean Squared Error: %2.4f" % mse_log
print "Mean Absolute Error: %2.4f" % mae_log
print "Root Mean Squared Error: %2.4f" % rmsle_log
print "Non log-transformed predictions:\n" + str(true_vs_predicted.take(3))
print "Log-transformed predictions:\n" + str(true_vs_predicted_log.take(3))

# 决策树
data_dt_log = data_dt.map(lambda lp:LabeledPoint(np.log(lp.label),lp.features))
dt_model_log = DecisionTree.trainRegressor(data_dt_log,{})

preds_log = dt_model_log.predict(data_dt_log.map(lambda p:p.features))
actual_log = data_dt_log.map(lambda p:p.label)
true_vs_predicted_dt_log = actual_log.zip(preds_log).map(lambda (t,p):(np.exp(t),np.exp(p)))

# 计算模型的MSE、MAE、RMSLE
mse_log_dt = true_vs_predicted_dt_log.map(lambda (t,p) : squared_error(t,p)).mean()
mae_log_dt = true_vs_predicted_dt_log.map(lambda (t,p) : abs_error(t,p)).mean()
rmsle_log_dt = np.sqrt(true_vs_predicted_dt_log.map(lambda (t,p) : squared_log_error(t,p)).mean())

print "Mean Squared Error: %2.4f" % mse_log_dt
print "Mean Absolute Error: %2.4f" % mae_log_dt
print "Root Mean Squared Error: %2.4f" % rmsle_log_dt
print "Non log-transformed predictions:\n" + str(true_vs_predicted_dt.take(3))
print "Log-transformed predictions:\n" + str(true_vs_predicted_dt_log.take(3))

# 4.2模型参数调优
data_with_idx = data.zipWithIndex().map(lambda (k,v):(v,k))
test = data_with_idx.sample(False,0.2,42)
train = data_with_idx.subtractByKey(test)

train_data = train.map(lambda (idx,p):p)
test_data = test.map(lambda (idx,p):p)
num_size = data.count()
train_size = train_data.count()
test_size = test_data.count()
print "Training data size: %d" % train_size
print "Test data size : %d" % test_size
print "Total data size : %d" % num_size
print "Train + Test size : %d" % (train_size + test_size)

# 提取决策树所需特征
data_with_idx_dt = data_dt.zipWithIndex().map(lambda (k,v):(v,k))
test_dt = data_with_idx_dt.sample(False,0.2,42)
train_dt = data_with_idx_dt.subtractByKey(test_dt)
train_data_dt = train_dt.map(lambda (idx,p):p)
test_data_dt = test_dt.map(lambda (idx,p):p)

# 评估函数
def evaluate(train,test,iterations,step,regParam,regType,intercept) :
    model = LinearRegressionWithSGD.train(train,iterations,step,regParam = regParam,regType = regType,intercept = intercept)
    tp = test.map(lambda p:(p.label,model.predict(p.features)))
    rmsle = np.sqrt(tp.map(lambda (t,p):squared_log_error(t,p)).mean())
    return rmsle

# 线性模型
# 迭代
params = [1,5,10,20,50,100]
metrics = [evaluate(train_data,test_data,param,0.01,0.0,'l2',False) for param in params]
print params
print metrics
pylab.plot(params,metrics)
fig = matplotlib.pyplot.gcf()
matplotlib.pyplot.xscale('log')

# 步长
params = [0.01,0.025,0.05,0.1,1.0]
metrics = [evaluate(train_data,test_data,10,param,0.0,'l2',False) for param in params]
print params
print metrics

# L2正则化
params = [0.0,0.01,0.1,1.0,5.0,10.0,20.0]
metrics = [evaluate(train_data,test_data,10,0.1,param,'l2',False) for param in params]
print params
print metrics
pylab.plot(params,metrics)
fig = matplotlib.pyplot.gcf()
matplotlib.pyplot.xscale('log')

# L1正则化
params = [0.0,0.01,0.1,1.0,10.0,100.0,1000.0]
metrics = [evaluate(train_data,test_data,10,0.1,param,'l1',False) for param in params]
print params
print metrics
pylab.plot(params,metrics)
fig = matplotlib.pyplot.gcf()
matplotlib.pyplot.xscale('log')

model_l1 = LinearRegressionWithSGD.train(train_data,10,0.1,regParam = 1.0,regType = 'l1',intercept = False)
model_l10 = LinearRegressionWithSGD.train(train_data,10,0.1,regParam = 10.0,regType = 'l1',intercept = False)
model_l100 = LinearRegressionWithSGD.train(train_data,10,0.1,regParam = 100.0,regType = 'l1',intercept = False)
print "L1 (1.0) number of zeros weights: " + str(sum(model_l1.weights.array == 0))
print "L1 (10.0) number of zeros weights: " + str(sum(model_l10.weights.array == 0))
print "L1 (100.0) number of zeros weights: " + str(sum(model_l100.weights.array == 0))

# 截距
params = [False,True]
metrics = [evaluate(train_data,test_data,10,0.1,1.0,'l2',param) for param in params]
print params
print metrics
matplotlib.pyplot.bar(params,metrics,color = "lightblue")
fig = matplotlib.pyplot.gcf()

# 决策树
# 评估函数
def evaluate_dt(train,test,maxDepth,maxBins):
    model = DecisionTree.trainRegressor(train,{},impurity = 'variance',maxDepth = maxDepth,maxBins = maxBins)
    preds = model.predict(test.map(lambda p:p.features))
    actual = test.map(lambda p:p.label)
    tp = actual.zip(preds)
    rmsle = np.sqrt(tp.map(lambda (t,p):squared_log_error(t,p)).mean())
    return rmsle

# 深度
params = [1,2,3,4,5,10,20]
metrics = [evaluate_dt(train_data_dt,test_data_dt,param,32) for param in params]
print params
print metrics
plt.plot(params,metrics)
fig = matplotlib.pyplot.gcf()
plt.show()

# 最大划分数
params = [2,4,8,16,32,64,100]
metrics = [evaluate_dt(train_data_dt,test_data_dt,5,param) for param in params]
print params
print metrics
plt.plot(params,metrics)
fig = matplotlib.pyplot.gcf()
plt.show()
