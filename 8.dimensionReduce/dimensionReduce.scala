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

// 3.使用降维模型
// LFW数据集上使用PCA投影数据

// PCA和SVD的关系

// 4.评价降维模型
// LFW数据集上估计SVD的k值
