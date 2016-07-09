// 1.抽取特征
val path = "/home/zhb/Desktop/work/SparkData/text/20news-bydate-train/*"
val rdd = sc.wholeTextFiles(path)
val text = rdd.map{case (file,text) => text}
println(text.count)

val newsgroups = rdd.map {case (file,text) =>
    file.split("/").takeRight(2).head }
val countByGroup = newsgroups.map( n=> (n,1)).reduceByKey(_+_).collect.sortBy(-_._2).mkString("\n")
println(countByGroup)

// 应用基本的分词方法
val text = rdd.map {case (file,text) => text}
val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
println(whiteSpaceSplit.distinct.count)

println(whiteSpaceSplit.sample(true,0.3,42).take(100).mkString(","))

// 改进分词效果
val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))
println(nonWordSplit.distinct.count)

println(nonWordSplit.distinct.sample(true,0.3,42).take(100).mkString(","))

// 使用正则模式可以过滤掉和这个模式不匹配的单词
val regex = """[^0-9]*""".r
val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)
println(filterNumbers.distinct.count)

println(filterNumbers.distinct.sample(true,0.3,42).take(100).mkString(","))

// 移除停用词
val tokenCounts = filterNumbers.map(t => (t,1)).reduceByKey(_ + _)
// 按照键值对的第二个元素排序，也即按照次数排序
val oreringDesc = Ordering.by[(String,Int),Int](_._2)
println(tokenCounts.top(20)(oreringDesc).mkString("\n"))

// 过滤停用词
val stopWords = Set("the","a","an","of","or","in","for","by","on","but","is","not",
    "with","as","was","if","they","are","this","and","it","have","from","at","my",
    "be","that","to")
val tokenCountsFilteredStopwords = tokenCounts.filter{case (k,v) => !stopWords.contains(k)}
println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))

// 删除仅仅含有一个字符的单词
val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter{case (k,v) => k.size >=2}
println(tokenCountsFilteredSize.top(20)(oreringDesc).mkString("\n"))

// 基于频率去除单词
val oreringAsc = Ordering.by[(String,Int),Int](-_._2)
println(tokenCountsFilteredSize.top(20)(oreringAsc).mkString("\n"))

val rareTokens = tokenCounts.filter{case (k,v) => v < 2}.map{
    case (k,v) => k}.collect.toSet
val tokenCountsFilteredAll = tokenCountsFilteredSize.filter{case (k,v) => !rareTokens.contains(k)}
println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))

println(tokenCountsFilteredAll.distinct.count)

// 把过滤逻辑组合到一个函数中，并应用到RDD中的每个文档
def tokenize(line:String) : Seq[String] = {
    line.split(""""\W+""")
    .map(_.toLowerCase)
    .filter(token => regex.pattern.matcher(token).matches)
    .filterNot(token => stopWords.contains(token))
    .filterNot(token => rareTokens.contains(token))
    .filter(token => token.size >= 2)
    .toSeq
}
println(text.flatMap(doc => tokenize(doc)).distinct.count)
val tokens = text.map(doc => tokenize(doc))
println(tokens.first.take(20))

// 训练TF-IDF模型
import org.apache.spark.mllib.linalg.{SparseVector => SV}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
val dim = math.pow(2,18).toInt
val hashingTF = new HashingTF(dim)
val tf = hashingTF.transform(tokens)
tf.cache

val v = tf.first.asInstanceOf[SV]
println(v.size)
println(v.values.size)
println(v.values.take(10).toSeq)
println(v.indices.take(10).toSeq)

val idf = new IDF().fit(tf)
val tfidf = idf.transform(tf)
val v2 = tfidf.first.asInstanceOf[SV]
println(v2.values.size)
println(v2.values.take(10).toSeq)
println(v2.indices.take(10).toSeq)

// 分析TF-IDF权重
val minMaxVals = tfidf.map{ v =>
    val sv = v.asInstanceOf[SV]
    (sv.values.min,sv.values.max)
}
val globalMinMax = minMaxVals.reduce{case ((min1,max1),(min2,max2)) =>
    (math.min(min1,min2), math.max(max1,max2))
}
println(globalMinMax)

// 对之前计算得到的频率最高的几个词的TF-IDF表示进行计算
val common = sc.parallelize(Seq(Seq("you","do","we")))
val tfCommon = hashingTF.transform(common)
val tfidfCommon = idf.transform(tfCommon)
val commonVector = tfidfCommon.first.asInstanceOf[SV]
println(commonVector.values.toSeq)

// 不常用单词应用相同的转换
val uncommon = sc.parallelize(Seq(Seq("telescope","legislation","investment")))
val tfUncommon = hashingTF.transform(uncommon)
val tfidfUncommon = idf.transform(tfUncommon)
val uncommonVector = tfidfUncommon.first.asInstanceOf[SV]
println(uncommonVector.values.toSeq)

// 2.使用TF-IDF模型
// 计算文本相似度和TF-IDF特征

// 使用TF-IDF训练文本分类器

// 3.评估文本处理技术的作用

// 4.word2vec模型
