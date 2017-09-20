package com.github.sshaddicts.lucrecium.neuralNetwork

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory
import java.io.IOException
import java.io.InputStream


class RichNeuralNet {
    private val ITERATIONS = 10
    private val LEARNING_RATE = 0.009

    private val REGULARIZATION = true
    private val L2REGULARIZATION = 0.005
    private val SEED = 123

    private val eval = Evaluation()

    private var iterationNumber = 0

    private var net: MultiLayerNetwork? = null

    private val log = LoggerFactory.getLogger(this.javaClass)

    constructor(net: MultiLayerNetwork) {
        this.net = net
    }

    constructor(outputLabelCount: Int) {
        this.net = initLenetMnist(outputLabelCount)
    }

    private fun initLenetMnist(outputLabelCount: Int): MultiLayerNetwork {
        val conf = NeuralNetConfiguration.Builder()
                .seed(SEED)
                .iterations(ITERATIONS)
                .regularization(true).l2(0.0005)
                .learningRate(LEARNING_RATE).biasLearningRate(0.02)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS) //To configure: .updater(new Nesterovs(0.9))
                .list()
                .layer(0, ConvolutionLayer.Builder(9, 9)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, ConvolutionLayer.Builder(3, 3)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputLabelCount)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(32, 32, 1))
                .build()

        return MultiLayerNetwork(conf)
    }

    fun train(data: DataSetIterator) {
        val before = System.nanoTime()
        iterationNumber++
        log.debug("working on iteration #" + iterationNumber)
        net!!.fit(data)
        val after = System.nanoTime()
        log.debug("done. took " + (after - before) / 100000000 + "seconds.")
    }

    fun train(data: INDArray) {
        net!!.fit(data)
    }

    fun predict(data: INDArray): IntArray? {
        return net?.predict(data)
    }

    fun eval(input: INDArray, actual: INDArray) {
        val output = net?.output(input)
        eval.eval(actual, output)
    }

    val stats: String
        get() = eval.stats()

    fun saveNetwork(filename: String) {
        try {
            ModelSerializer.writeModel(net!!, filename, true)
        } catch (e: IOException) {
            log.error(e.message, e)
        }

    }

    companion object {

        @Throws(IOException::class)
        fun loadNetwork(filename: String): MultiLayerNetwork {
            return ModelSerializer.restoreMultiLayerNetwork(filename)
        }

        @Throws(IOException::class)
        fun loadnetwork(`is`: InputStream): MultiLayerNetwork {
            return ModelSerializer.restoreMultiLayerNetwork(`is`)
        }
    }
}
