package com.github.sshaddicts.lucrecium.imageProcessing

import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer
import com.github.sshaddicts.lucrecium.util.split
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgcodecs.Imgcodecs.*
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import org.slf4j.LoggerFactory
import java.awt.image.BufferedImage
import java.io.IOException
import java.util.*


/**
 * This class is intended to processing Images
 *
 * @author Alexey Shereshovets
 * @see
 */
class ImageProcessor {

    private var resizeRate = 0.3
    private val log = LoggerFactory.getLogger(this.javaClass)

    private fun Mat.findContours(mode: Int, method: Int): List<MatOfPoint> {
        val contours = LinkedList<MatOfPoint>()

        Imgproc.findContours(this, contours, Mat(), mode, method)

        return contours
    }

    fun Mat.threshold(): Mat {
        val mask = Mat(this.height(), this.width(), this.type())
        Imgproc.threshold(this, mask, 0.0, 255.0,
                Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)

        return mask
    }

    fun Mat.adaptiveThreshold(): Mat{
        val mask = Mat(this.size(), this.type())
        Imgproc.adaptiveThreshold(this, mask, 255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV,
                21,22.0)
        return mask
    }

    fun Mat.resize(): Mat {
        if (this.height() == 0 || this.width() == 0) {
            throw IllegalArgumentException("Image size is illegal" + this.size().toString())
        }

        if (this.height() < RESIZE_THRESHOLD || this.width() < RESIZE_THRESHOLD) {
            resizeRate = 1.0
        }

        val tempMat = Mat(this.height(), this.width(), this.type())

        val height = this.height() * resizeRate
        val width = this.width() * resizeRate

        Imgproc.resize(this, tempMat, Size(width, height), 2.0, 2.0, Imgproc.INTER_AREA)
        return tempMat
    }

    fun Mat.crop(): Mat {
        val tmpImage = this.threshold()

        val contours = tmpImage.findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)

        var result = Mat()

        for (mat in contours) {
            if (Imgproc.contourArea(mat) > tmpImage.rows() * tmpImage.cols() / 4) {
                val rect = Imgproc.boundingRect(mat)
                if (rect.height > 50 && (rect.height != this.height() || rect.width != this.width())) {
                    rect.x -= 1
                    rect.y -= 1
                    rect.width += 1
                    rect.height += 1
                    result = this.submat(rect)
                }
            }
        }

        if (result.height() != 0 && result.width() != 0) {
            return result
        }
        log.warn("Not found a contour suitable for cropping the image. Consider brighter light, less blurred image or higher contrast levels")
        return this
    }

    fun Mat.adjustContrast(alpha: Double, beta: Double): Mat {
        val result = Mat.zeros(this.size(), this.type())
        this.convertTo(result, -1, alpha, beta)

        return result
    }

    /**
     * Main method of [ImageProcessor]. Processed image from a given filepath.
     */
    @JvmOverloads
    fun findTextRegions(filename: String, mergeType: Int = ImageProcessor.NO_MERGE, isRotationNeeded: Boolean = false): SearchResult {
        return findTextRegions(ImageProcessor.loadImage(filename), mergeType, isRotationNeeded)
    }

    /**
     * Main method of [ImageProcessor]. Processed image from a given [Mat].
     */
    @JvmOverloads
    fun findTextRegions(image: Mat, mergeType: Int = ImageProcessor.NO_MERGE, isRotationNeeded: Boolean = false): SearchResult {
        val processed = process(image, isRotationNeeded)

        val chars = detectText(processed, mergeType, isRotationNeeded)

        return constructCharRegions(processed, chars)
    }

    /**
     * Main method of [ImageProcessor]. Processed image from a given [ByteArray].
     */
    @JvmOverloads
    fun findTextRegions(bytes: ByteArray, mergeType: Int = ImageProcessor.NO_MERGE, isRotationNeeded: Boolean = false): SearchResult {
        return findTextRegions(ImageProcessor.loadImage(bytes), mergeType, isRotationNeeded)
    }

    fun constructCharRegions(image: Mat, chars: List<Rect>):SearchResult {

        val charMap = chars.map {
            CharContainer(image.submat(it), it)
        }
        var sorted : List<CharContainer> = charMap
        try {
            sorted = charMap.sorted()
            SearchResult(sorted, makeOverlay(image, chars))
        } catch (e: Throwable) {
            log.error(e.message)
        }

        return SearchResult(sorted, makeOverlay(image, chars))
    }


    private fun deskew(src: Mat, angle: Double) {
        val center = Point((src.width() / 2).toDouble(), (src.height() / 2).toDouble())
        val rotatedImage = Imgproc.getRotationMatrix2D(center, angle, 1.0)

        val size = Size(src.width().toDouble(), src.height().toDouble())
        Imgproc.warpAffine(src, src, rotatedImage, size,
                Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS)
    }

    private fun deskew(src: Mat): Double {
        val tmpMat = src.clone()

        Imgproc.adaptiveThreshold(
                tmpMat,
                tmpMat,
                255.0,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV,
                21,
                22.0
        )

        val pointMat = Mat.zeros(tmpMat.size(), tmpMat.channels())
        Core.findNonZero(tmpMat, pointMat)

        val mat2f = MatOfPoint2f()
        pointMat.convertTo(mat2f, CvType.CV_32FC2)

        val rotated = Imgproc.minAreaRect(mat2f)

        if (rotated.size.width > rotated.size.height)
            rotated.angle += 90.0

        log.debug("Rotation angle: " + rotated.angle + " degrees.")

        deskew(src, rotated.angle)

        return rotated.angle
    }

    fun process(image: Mat, isRotationNeeded: Boolean): Mat {

        if (isRotationNeeded) {
            deskew(image)
        }

        val result = image.resize().adjustContrast(5.0, -780.0).adaptiveThreshold()
        val end = result.crop()

        log.debug("Image size is " + result.size())

        return result
    }

    private fun detectText(processed: Mat, mergeType: Int, isRotationNeeded: Boolean): List<Rect> {

        val contours = processed.findContours(RETR_TREE, CHAIN_APPROX_NONE).toMutableList()

        if (processed.height() > 500) {
            contours.removeIf { processed.height() - it.height() < 200 }
        }

        contours.removeIf { it.size().area() < 6 }

        val chars = processed.mergeInnerRects(contours, mergeType).toMutableList()
        chars.removeIf { it.width < 4 || it.height < 4 }
        chars.reverse()

        log.debug("Contours size: " + contours.size)

        return chars
    }

    private fun Mat.mergeInnerRects(points: List<MatOfPoint>, mergeType: Int): List<Rect> {
        val mask = Mat.zeros(this.size(), this.type())

        for (point in points) {
            val rect = Imgproc.boundingRect(point)
            rect.width -= mergeType
            rectangle(mask, rect.tl(), rect.br(), Scalar(255.0), -10)
        }

        val result = mask
                .findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)
                .map { Imgproc.boundingRect(it) }

        log.debug("ROI number after merging inner rects: " + result.size)

        return result
    }

    fun approximateLineNumberFor(m: Mat): Int {
        var lineNumber = 0

        val sums = calculateLineSums(m)

        val mean = Core.mean(sums).`val`[0].toInt() / 3

        var prev = false

        (0 until sums.height())
                .asSequence()
                .map { sums.get(it, 0)[0].toInt() }
                .forEach {
                    if (prev) {
                        prev = it < mean
                    } else {
                        prev = it < mean
                        lineNumber += if (prev) 1 else 0
                    }
                }

        return lineNumber
    }

    private fun calculateLineSums(m: Mat): Mat {
        if (m.width() < 20) {
            throw IllegalArgumentException("m is too small, and probably is not valid")
        }
        val result = Mat(m.height(), 1, CvType.CV_32S)

        for (i in 0 until m.height()) {
            result.put(i, 0, Core.sumElems(m.row(i)).`val`[0])
        }
        return result
    }

    private fun calculateMean(rects: List<Rect>, horizontal: Boolean): Int {
        val total = rects.sumBy { if (horizontal) it.width else it.height }

        return total / rects.size
    }

    private fun splitForThreshold(rects: List<Rect>, mean: Int, horizontal: Boolean): List<Rect> {
        val result = ArrayList<Rect>(rects.size)

        rects.forEach { rect ->
            if (rect.height > mean * 2) {
                val split = rect.split(Math.floor((rect.height / mean).toDouble()).toInt(), horizontal)
                result.addAll(split)
            } else {
                result.add(rect)
            }
        }

        return result
    }

    private fun makeOverlay(image: Mat, chars: List<Rect>): ByteArray {
        val imageClone = Mat.zeros(image.size(), 16)
        cvtColor(image, imageClone, Imgproc.COLOR_GRAY2RGB)
        log.debug("debug image type is " + imageClone.type())

        drawRects(imageClone, chars, Scalar(0.0, 128.0, 255.0))

        val bytes = MatOfByte()
        imencode(".jpg", imageClone, bytes)

        return bytes.toArray()
    }

    fun drawRects(image: Mat, rects: List<Rect>, color: Scalar) =
            rects.forEach { rect -> rectangle(image, rect.tl(), rect.br(), color, 1) }


    fun drawContours(image: Mat, contours: List<MatOfPoint>, color: Scalar) =
            Imgproc.drawContours(image, contours, -1, color, -1)

    companion object {

        val MERGE_WORDS = -5
        val MERGE_LINES = -20
        val NO_MERGE = 2

        private val RESIZE_THRESHOLD = 1000

        fun toBufferedImage(image: Mat): BufferedImage = Imshow.toBufferedImage(image)

        fun toByteArray(src: Mat): ByteArray {
            val byteData = ByteArray(src.height() * src.width() * src.channels())
            src.get(0, 0, byteData)
            return byteData
        }

        @Throws(IOException::class)
        fun toNdarray(mat: Mat): INDArray {
            val matData = ByteArray(mat.width() * mat.height())
            val retData = DoubleArray(matData.size)

            mat.get(0, 0, matData)

            for (i in matData.indices) {
                retData[i] = matData[i].toDouble()
            }

            return Nd4j.create(retData, intArrayOf(mat.width(), mat.height()))
        }

        @JvmOverloads
        fun loadImage(bytes: ByteArray, flags: Int = Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE): Mat {
            return imdecode(MatOfByte(*bytes), flags)
        }

        fun loadImage(filename: String?): Mat {
            if (filename == null || filename == "") {
                throw IllegalArgumentException("Filename cannot be null or empty. Requested filepath: " + filename!!)
            }

            val image = imread(filename, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE)

            if (image.height() == 0 || image.width() == 0) {
                throw IllegalArgumentException("Image has to be at least 1x1. current dimensions: height = "
                        + image.height() + ", width = " + image.width() + ".\n" +
                        "Requested filepath: " + filename + ", check it once again.")
            }

            return image
        }
    }
}