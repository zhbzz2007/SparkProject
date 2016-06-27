#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: GBK -*-

from pyspark import SparkContext,SparkConf
import matplotlib
import pylab
import numpy as np

def userDatatest():
    # 3.2.1探索用户数据
    conf = SparkConf().setAppName("userTest")
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


    # 3.4从数值中提取有用特征
    # 3.4.2 类别特征
    # 取回occupations的所有可能值
    all_occupations = user_fields.map(lambda fields:fields[3]).distinct().collect()
    all_occupations.sort()
    idx = 0
    # 依次对各可能的职业分配序号
    all_occupations_dict = {}
    for o in all_occupations:
        all_occupations_dict[o] = idx
        idx += 1
    print "Encoding of 'doctor' : %d" % all_occupations_dict['doctor']
    print "Encoding of 'programmer' : %d" % all_occupations_dict['programmer']

    K = len(all_occupations_dict)
    binary_x = np.zeros(K)
    k_programmer = all_occupations_dict['programmer']
    binary_x[k_programmer] = 1
    print "Binary feature vector: %s" % binary_x
    print "Length of binary vector : %d" %K

def moiveDataTest():
    # 3.2.2探索电影数据 #
    conf = SparkConf().setAppName("moiveTest")
    sc = SparkContext(conf=conf)
    moiveFileName = "/home/zhb/Desktop/work/SparkData/ml-100k/u.item"
    moive_data = sc.textFile(moiveFileName)
    print moive_data.first()
    num_moives = moive_data.count()
    print "Moives:%d" % num_moives

    # 数据转换
    def convert_year(x):
        try:
            return int(x[-4:])
        except:
            return 1900#若数据缺失年份，将其年份设置为1900,在后续处理中会过滤掉这类数据

    moive_fields = moive_data.map(lambda lines:lines.split("|"))
    years = moive_fields.map(lambda fields:fields[2]).map(lambda x:convert_year(x))
    years_filtered = years.filter(lambda x:x != 1900)

    moive_ages = years_filtered.map(lambda yr:1998-yr).countByValue()
    values = moive_ages.values()
    bins = moive_ages.keys()
    pylab.hist(values, bins = bins, color = 'lightblue', normed = True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16,10)

    # 3.3非规整数据和缺失数据的填充
    years_pre_processed = moive_fields.map(lambda fields:fields[2]).map(lambda x:convert_year(x)).collect()
    years_pre_processed_array = np.array(years_pre_processed)

    # 计算发行年份的平均数，不包括非规整数据
    mean_year = np.mean(years_pre_processed_array[years_pre_processed_array != 1900])
    # 计算发行年份的中位数，不包括非规整数据
    median_year = np.median(years_pre_processed_array[years_pre_processed_array != 1900])
    # 找到非规整数据点的序号
    index_bad_data = np.where(years_pre_processed_array == 1900)[0][0]
    # 通过序号将中位数作为非规整数据的发行年份
    years_pre_processed_array[index_bad_data] = median_year
    print "Mean year of release : %d" % mean_year
    print "Median year of release : %d" % median_year
    print "Index of '1900' after assigning median : %s" % np.where(years_pre_processed_array == 1900)[0]


    # 3.4.4 文本特征
    def extrat_title(raw):
        import re
        grps = re.search("\((\w+)\)",raw)
        if grps:
            return raw[:grps.start()].strip()
        else:
            return raw

    raw_titles = moive_fields.map(lambda fields:fields[1])
    b = [extrat_title(raw_title) for raw_title in raw_titles.take(5)]
    print b

    moive_titles = raw_titles.map(lambda m:extrat_title(m))
    # 用简单空白分词法将标题分词为词
    title_terms = moive_titles.map(lambda t:t.split(" "))
    print title_terms.take(5)

    # 使用flatMap来扩展title_terms RDD中每个记录的字符串列表，以得到一个新的字符串RDD
    # 下面取回所有可能的词，以便构建一个词到序号的映射词典
    all_terms = title_terms.flatMap(lambda x:x).distinct().collect()
    # 创建一个新的字典来保存词，并分配k之1序号
    idx = 0
    all_terms_dict = {}
    for term in all_terms:
        all_terms_dict[term] = idx
        idx += 1
    print "Total number of terms:%d" % len(all_terms_dict)
    print "Index of term 'Dead':%d" % all_terms_dict['Dead']
    print "Index of term 'Rooms': %d" % all_terms_dict['Rooms']

    # 通过Spark的zipWithIndex函数可以更高效的得到相同结果
    all_terms_dict2 = title_terms.flatMap(lambda x:x).distinct().zipWithIndex().collectAsMap()
    print "Index of term 'Dead':%d" % all_terms_dict['Dead']
    print "Index of term 'Rooms': %d" % all_terms_dict['Rooms']

    # 将一个词集合转换为一个稀疏向量的表示
    def create_vector(terms,term_dict):
        from scipy import sparse as sp
        num_terms = len(term_dict)
        x = sp.csc_matrix((1,num_terms))
        for t in terms:
            if t in term_dict:
                idx = term_dict[t]
                x[0,idx] = 1
        return x

    all_terms_bcast = sc.broadcast(all_terms_dict)
    term_vector = title_terms.map(lambda terms:create_vector(terms,all_terms_bcast.value))
    print term_vector.take(5)


    # 正则化特征
    np.random.seed(42)
    x = np.random.randn(10)
    norm_x_2 = np.linalg.norm(x)
    normalized_x = x / norm_x_2
    print "x:\n%s" % x
    print "2-Norm of x: %2.4f" %norm_x_2
    print "Normalized x:\n%s" %normalized_x
    print "2-Norm of normalized_x: %2.4f" % np.linalg.norm(normalized_x)

    from pyspark.mllib.feature import Normalizer
    normalizer = Normalizer()
    vector = sc.parallelize([x])
    normalized_x_mllib = normalizer.transform(vector).first().toArray()
    print "x:\n%s" % x
    print "2-Norm of x: %2.4f" %norm_x_2
    print "Normalized x MlLib:\n%s" %normalized_x_mllib
    print "2-Norm of normalized_x_mllib: %2.4f" % np.linalg.norm(normalized_x_mllib)


def rateDataTest():
    # 3.2.3探索评级数据
    conf = SparkConf().setAppName("rateTest")
    sc = SparkContext(conf=conf)
    rateFileName = "/home/zhb/Desktop/work/SparkData/ml-100k/u.data"
    rating_data = sc.textFile(rateFileName)
    print rating_data.first()
    num_ratings = rating_data.count()
    print "Ratings:%d" % num_ratings

    rating_data = rating_data.map(lambda line:line.split("\t"))
    ratings = rating_data.map(lambda fields :int(fields[2]))
    max_rating = ratings.reduce(lambda x,y:max(x,y))
    min_rating = ratings.reduce(lambda x,y:min(x,y))
    mean_rating = ratings.reduce(lambda x,y:x+y) / num_ratings
    median_rating = np.median(ratings.collect())
    num_users = 943
    ratings_per_user = num_ratings / num_users
    num_moives = 1682
    ratings_per_moive = num_ratings / num_moives
    print "Min rating : %d" % min_rating
    print "Max rating : %d" % max_rating
    print "Average rating : %2.2f" % mean_rating
    print "Median rating : %d" % median_rating
    print "Average # of ratings per user : %2.2f" % ratings_per_user
    print "Average # of ratings per moive : %2.2f" % ratings_per_moive

    print ratings.stats()

    count_by_rating = ratings.countByValue()
    x_axis = np.array(count_by_rating.keys())
    y_axis = np.array([float(c) for c in count_by_rating.values()])
    # 对y轴正则化，使它表示百分比
    y_axis_normed = y_axis / y_axis.sum()
    pos = np.arange(len(x_axis))
    width = 1.0

    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(x_axis)

    plt.bar(pos,y_axis_normed,width,color = "lightblue")
    plt.xticks(rotation = 30)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16,10)

    user_ratings_grouped = rating_data.map(lambda fields :(int(fields[0]),int(fields[2]))).groupByKey()
    user_ratings_byuser = user_ratings_grouped.map(lambda (k,v):(k,len(v)))
    print user_ratings_byuser.take(5)

    user_ratings_byuser_local = user_ratings_byuser.map(lambda (k,v):v).collect()
    pylab.hist(user_ratings_byuser_local,bins = 200,color = "lightblue",normed = True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16,10)

    # 3.4从数值中提取有用特征
    # 3.4.3 派生特征
    # 将评级时间戳提取为datetime的格式
    def extract_datetime(ts):
        import datetime
        return datetime.datetime.fromtimestamp(ts)

    # 将时间戳属性转换为int类型，然后将各时间戳转为datetime类型的对象，进而提取其点钟数
    timestamps = rating_data.map(lambda fields:int(fields[3]))
    hour_of_day = timestamps.map(lambda ts:extract_datetime(ts).hour)
    print hour_of_day.take(5)

    # 根据点钟数划分不同时段
    def assign_tod(hr):
        times_of_day = {
        "morning":range(7,12),
        "lunch":range(12,14),
        "afternoon":range(14,18),
        "evening":range(18,23),
        "night":range(23,25) + range(0,7)
        }
        for k,v in times_of_day.iteritems():
            if hr in v:
                return k
     time_of_day = hour_of_day.map(lambda hr:assign_tod(hr))
     print time_of_day.take(5)

if __name__ == "__main__":
    moiveDataTest()
