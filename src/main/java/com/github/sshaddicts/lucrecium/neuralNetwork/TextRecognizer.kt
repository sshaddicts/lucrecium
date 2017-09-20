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
    fun recognize(containers: List<CharContainer>): List<ObjectNode> {
        val entries = ArrayList<ObjectNode>(containers.size)

        val loader = Java2DNativeImageLoader(32, 32, 1)

        log.debug("container size " + containers.size)

        val sb = StringBuilder()

        var prevY = containers[0].rect.y

        for (i in containers.indices) {
            val rect = containers[i].rect

            val currentY = rect.y

            //the line ends at this condition
            if (Math.abs(currentY - prevY) > 10) {
                entries.add(mapper.valueToTree<JsonNode>(Occurrence(
                        "entry_" + i,
                        java.lang.Double.parseDouble(sb.toString())
                )) as ObjectNode)
                sb.setLength(0)
            }

            prevY = currentY

            val slice = containers[i].mat
            val bufferedSlice = ImageProcessor.toBufferedImage(slice)

            val array = loader.asMatrix(bufferedSlice)

            val `in` = net.predict(array)!![0]
            sb.append(`in`)
        }

        entries.add(mapper.valueToTree<JsonNode>(Occurrence(
                "last_entry",
                java.lang.Double.parseDouble(sb.toString())
        )) as ObjectNode)

        return entries
    }
}
