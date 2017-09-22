package com.github.sshaddicts.lucrecium.neuralNetwork

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ObjectNode
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor
import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer
import org.datavec.image.loader.Java2DNativeImageLoader
import org.slf4j.LoggerFactory
import java.io.IOException
import java.io.InputStream
import java.util.*

class TextRecognizer {

    private val net: RichNeuralNet

    private val log = LoggerFactory.getLogger(this.javaClass)

    private val mapper = ObjectMapper().registerKotlinModule()

    @Throws(IOException::class)
    constructor(filename: String) {
        net = RichNeuralNet(RichNeuralNet.loadNetwork(filename))
    }

    @Throws(IOException::class)
    constructor(`is`: InputStream) {
        net = RichNeuralNet(RichNeuralNet.loadnetwork(`is`))
    }

    //TODO refactor
    @Throws(IOException::class)
    fun recognize(containers: List<CharContainer>, labels: List<String>): List<ObjectNode> {
        val entries = ArrayList<ObjectNode>(containers.size)

        val loader = Java2DNativeImageLoader(32, 32, 1)

        log.debug("container size " + containers.size)

        val sb = StringBuilder()

        var prevX = containers[0].rect.x
        var prevXWidth = containers[0].rect.width
        var prevY = containers[0].rect.y

        val regex = "((?:\\d\\s?){3,5})\\s".toRegex()

        var prev:String = ""

        for (i in containers.indices) {
            val rect = containers[i].rect

            val currentY = rect.y
            val currentX = rect.x

            if(currentX - prevX > prevXWidth * 1.5){
                sb.append(" ")
            }

            //the line ends at this condition
            if (Math.abs(currentY - prevY) > 10) {

                val row = sb.toString().trim()

                val find = regex.find(row)
                if(find == null){
                    prev+=row
                }else {
                    val value =find.groupValues[0]
                    val result = fixValue(value)

                    val name = row.subSequence(0, find.groups[0]!!.range.start) as String

                    entries.add(mapper.valueToTree<JsonNode>(Occurrence(
                            name.toLowerCase(),
                            result
                    )) as ObjectNode)
                    sb.setLength(0)
                }
            }

            if(sb.length > 70){
                sb.setLength(0)
            }

            prevY = currentY
            prevX = currentX
            prevXWidth = rect.width

            val slice = containers[i].mat
            val bufferedSlice = ImageProcessor.toBufferedImage(slice)
            val array = loader.asMatrix(bufferedSlice)

            val `in` = labels.get(net.predict(array)!![0])
            sb.append(`in`)
        }

        return entries
    }

    fun fixValue(value: String) : Double{

        val tmpString = value.replace(" ", "")

        val first = tmpString.subSequence(0,tmpString.length -2) as String
        val second = tmpString.subSequence(tmpString.length - 2, tmpString.length) as String

        return java.lang.Double.parseDouble(first + "." + second)
    }

}
