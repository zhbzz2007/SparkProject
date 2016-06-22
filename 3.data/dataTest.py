#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: GBK -*-

from pyspark import SparkContext,SparkConf
import matplotlib
import pylab
import numpy as np

def datatest():
    conf = SparkConf().setAppName("test")
    sc = SparkContext(conf=conf)
    userFileName = "/home/zhb/Desktop/work/SparkData/ml-100k/u.user"
    user_data = sc.textFile(userFileName)
    print "user_data first col:",user_data.first()

    user_fields = user_data.map(lambda line : line.split("|"))
    num_users = user_fields.map(lambda fields : fields[0]).count()
    num_genders = user_fields.map(lambda fields : fields[2]).distinct().count()
    num_occupations = user_fields.map(lambda fields : fields[3]).distinct().count()
    num_zipcodes = user_fields.map(lambda fields : fields[4]).distinct().count()
    print "User:%d,gender:%d,occupations:%d,zip code:%d"%(num_users,num_genders,num_occupations,num_zipcodes)

    # 直方图分析用户年龄的分布
    ages = user_fields.map(lambda x : int(x[1])).collect()
    pylab.hist(ages,bins = 20, color = "lightblue",normed = True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16,10)

    # 分析用户的职业分布
    count_by_occupation = user_fields.map(lambda fields : (fields[3],1)).reduceByKey(lambda x,y : x+y).collect()
    x_axis1 = np.array([c[0] for c in count_by_occupation])
    y_axis1 = np.array([c[1] for c in count_by_occupation])

    x_axis = x_axis1[np.argsort(x_axis1)]
    y_axis = y_axis1[np.argsort(y_axis1)]

    pos = np.arange(len(x_axis))
    width = 1.0

    ax = plt.axes()
    ax.set_xticks(pos + (width/2))
    ax.set_xticklabels(x_axis)

    plt.bar(pos,y_axis,width,color = "lightblue")

    plt.xticks(rotation = 30)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16,10)

    # 利用countByValue统计不同值分别出现的次数
    count_by_occupation2 = user_fields.map(lambda fields : fields[3]).countByValue()
    print "Map-Reduce Approach:"
    print dict(count_by_occupation)
    print ""
    print "countByValue Approach:"
    print dict(count_by_occupation2)

if __name__ == "__main__":
    datatest()
