#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: GBK -*-

# 1.加载和查看数据集
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

# 为决策树创建特征向量

# 2.回归模型的训练和使用

# 3.评估回归模型的性能

# 4.改进模型性能和参数调优
