// 1.抽取特征
// 使用通配符的路径标识，来告诉Spark在lfw文件夹中访问每个文件夹以获取文件
val path = "/home/zhb/Desktop/work/SparkData/dimension/lfw/*"
// 返回一个键值对，键是文件位置，值是整个文件内容
val rdd = sc.wholeTextFiles(path)
val first = rdd.first
println(first)

// rdd中文件路径格式是以"file:"开始，因此需要将前面部分删掉
// "file://"是本地文件系统，"hdfs://"是hdfs，"s3n://"是Amazon S3文件系统
val files = rdd.map {case (fileName,content) =>
    fileName.replace("file:","") }
println(files.first)
println(files.count)

// (1)载入图片
import java.awt.image.BufferedImage
def loadImageFromFile(path:String):BufferedImage = {
    import javax.imageio.ImageIO
    import java.io.File
    ImageIO.read(new File(path))
}

val aePath = "/home/zhb/Desktop/work/SparkData/dimension/lfw/Adrien_Brody/Adrien_Brody_0001.jpg"
val aeImage = loadImageFromFile(aePath)

// (2)转换灰度图片并改变图片尺寸
def processImage(image:BufferedImage,width:Int,height:Int):BufferedImage = {
    // 创建一个指定宽、高和灰度模型的新图片
    val bwImage = new BufferedImage(width,height,BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics()
    // 从原始图片绘制除灰度图片，drawImage方法负责颜色转换和尺寸变化
    g.drawImage(image,0,0,width,height,null)
    g.dispose()
    bwImage
}

// 测试示例图片的输出，转换灰度图片并改变尺寸到100*100像素
val grayImage = processImage(aeImage,100,100)

// 存储处理过的图片文件到临时路径
import javax.imageio.ImageIO
import java.io.File
ImageIO.write(grayImage,"jpg",new File("/home/zhb/aeGrapy.jpg"))

// (3)提取特征向量
// 打平二维的像素矩阵来构造一维的向量
def getPixelsFromImage(image:BufferedImage):Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0,0,width,height,pixels)
}

// 接受一个图片位置和需要处理的宽和高，返回一个包含像素数据的Array
def extractPixels(path:String, width:Int, height:Int):Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw,width,height)
    getPixelsFromImage(processed)
}

// 把上述函数应用到包含图片路径的RDD的每一个元素上将产生一个新的RDD，新的RDD包含每张图片的像素数据
val pixels = files.map(f => extractPixels(f,50,50))
println(pixels.take(10).map(_.take(10).mkString("",",",",...")).mkString("\n"))

import org.apache.spark.mllib.linalg.Vectors
// 为每一张图片创建MLlib向量对象，并将RDD缓存
val vectors = pixels.map(p => Vectors.dense(p))
vectors.setName("image-vectors")
vectors.cache

// (4)正则化
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler
// 只从数据中提取平均值，不提取方差
val scaler = new StandardScaler(withMean = true,withStd = false).fit(vectors)
// 使用返回的scaler来转换原始的图像向量，让所有向量减去当前列的平均值
val scaledVectors = vectors.map(v => scaler.transform(v))

// 2.训练降维模型
// LFW数据集上运行PCA
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val matrix = new RowMatrix(scaledVectors)
val K = 10
val pc = matrix.computePrincipalComponents(K)

// 可视化特征脸
val rows = pc.numRows
val cols = pc.numCols
println(rows,cols)

// 使用变量(一个MLlib)创建一个Breeze DenseMatrix
import breeze.linalg.DenseMatrix
val pcBreeze = new DenseMatrix(rows,cols,pc.toArray)

import breeze.linalg.csvwrite
csvwrite(new File("/home/zhb/pc.csv"),pcBreeze)

// 3.使用降维模型
// LFW数据集上使用PCA投影数据
val projected = matrix.multiply(pc)
println(projected.numRows,projected.numCols)
println(projected.rows.take(5).mkString("\n"))

// PCA和SVD的关系
// SVD计算产生的右奇异值向量等同于计算得到的主城分
val svd = matrix.computeSVD(10,computeU = true)
println(s"U dimension : (${svd.U.numRows}, ${svd.U.numCols})")
println(s"S dimension : (${svd.s.size})")
println(s"V dimension : (${svd.V.numRows}, ${svd.V.numCols})")

// 不考虑正负号和浮点数误差，判断两个数组是否相同
def approxEqual(array1:Array[Double], array2:Array[Double],tolerance:Double = 1e-6) : Boolean = {
    val bools = array1.zip(array2).map{ case (v1,v2) =>
        if (math.abs(math.abs(v1) - math.abs(v2)) > 1e-6) false else true
    }
    bools.fold(true)(_&_)// 对所有的位置是否相等取与运算
}
println(approxEqual(Array(1.0,2.0,3.0), Array(1.0,2.0,3.0)))

println(approxEqual(Array(1.0,2.0,3.0), Array(3.0,2.0,1.0)))

println(approxEqual(svd.V.toArray,pc.toArray))

// 矩阵U和向量S（或者对焦矩阵S）的乘积和PCA中用来把原始图像数据投影到10个主成分构成的空间中的投影矩阵相等
val breezeS = breeze.linalg.DenseVector(svd.s.toArray)
val projectedSVD = svd.U.rows.map {v =>
    val breezeV = breeze.linalg.DenseVector(v.toArray)
    val multV = breezeV :* breezeS
    Vectors.dense(multV.data)
}
projected.rows.zip(projectedSVD).map{case (v1,v2) =>
approxEqual(v1.toArray,v2.toArray)}.filter(b => true).count

// 4.评价降维模型
// LFW数据集上估计SVD的k值
// 奇异值每次运行结果相同，并且按照递减的顺序返回
val sValues = (1 to 5).map{ i => matrix.computeSVD(i,computeU = false).s}
sValues.foreach(println)

// 计算最大的300个奇异值，然后写入到临时CSV文件中
val svd300 = matrix.computeSVD(300,computeU=false)
val sMatrix = new DenseMatrix(1,300,svd300.s.toArray)
csvwrite(new File("/home/zhb/s.csv"),sMatrix)
