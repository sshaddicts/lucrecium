package com.github.sshaddicts.lucrecium

import com.github.sshaddicts.lucrecium.datasets.ImageDataSet
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow
import com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet
import com.github.sshaddicts.lucrecium.neuralNetwork.TextRecognizer
import com.github.sshaddicts.lucrecium.util.FileInteractions
import org.apache.commons.io.FileUtils
import org.opencv.core.Core
import org.opencv.imgcodecs.Imgcodecs
import org.slf4j.LoggerFactory
import java.io.File
import java.io.IOException
import java.lang.reflect.InvocationTargetException
import java.util.*

object Runner {

    //private val filenames = FileInteractions.getFileNamesFromDir("dataSources/newData")
    //private val greatFile = FileInteractions.getFileNamesFromDir("dataSources/greatData")
    //private val netDir = "outputData/productionDataSet2"
    private val log = LoggerFactory.getLogger(this.javaClass)

    val labels: List<String> = listOf(
            "0","1","2","3","4","5",
            "6","7","8","9","dot","slash",
            "Є","І","Ї","А","Б","В",
            "Г","Д","Е","Ж","З","И",
            "Й","К","Л","М","Н","О",
            "П","Р","С","Т","У","Ф",
            "Х","Ц","Ч","Ш","Щ","Ь",
            "Ю","Я",
            "а","б","в","г","д","е",
            "ж","з","и","й","к","л",
            "м","н","о","п","р","с",
            "т","у","ф","х","ц","ч",
            "ш","щ","ь","э","ю","я",
            "є","і","ї","ґ"
    )

    init {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    }

    @Throws(IOException::class, NoSuchMethodException::class, InvocationTargetException::class, IllegalAccessException::class)
    @JvmStatic
    fun main(args: Array<String>) {

        val numb = 3
        //trainNetwork(1, outputLabelCount);

        testNetworkOnImage(numb, labels)
        println("done")
    }

    @Throws(IOException::class)
    fun recognizeDigits() {
        val image = ImageProcessor.loadImage("testCase/numbers.jpg")
        val processor = ImageProcessor()

        val recognizer = TextRecognizer("netFileModLetters.lucrecium")

        val (chars) = processor.findTextRegions(image, ImageProcessor.NO_MERGE, false)

        val text = recognizer.recognize(chars, ArrayList())

        println(text)
    }

    @Throws(IOException::class)
    fun tryCatch() {
        val bytes = FileUtils.readFileToByteArray(File("testCase/numbers31.jpg"))

        val image = ImageProcessor.loadImage(bytes)

        Imshow.show(image, "1")

        val processor = ImageProcessor()

        val (chars, overlay) = processor.findTextRegions(image)

        val recognizer = TextRecognizer("netFile")

        val recognize = recognizer.recognize(chars, ArrayList())

        println(recognize)
        Imshow.show(chars[0].mat)
        Imshow.show(ImageProcessor.loadImage(overlay, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED))
    }

    @Throws(IOException::class, NoSuchMethodException::class, InvocationTargetException::class, IllegalAccessException::class)
    fun testNetworkOnImage(num: Int, labels: List<String>) {
        val image = ImageProcessor.loadImage("test.jpg")
        val processor = ImageProcessor()

        val (chars, overlay) = processor.findTextRegions(image, ImageProcessor.NO_MERGE, true)
        println("found " + chars.size + " elements")

        val recognizer = TextRecognizer("netFileModLetters.lucrecium")
        val recognize = recognizer.recognize(chars, labels)

        for (node in recognize) {
            println(node)
        }

    }


}
