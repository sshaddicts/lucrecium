package com.github.sshaddicts.lucrecium.neuralNetwork;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.util.Log;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;


public class RichNeuralNet {
    private final int ITERATIONS = 10;
    private final double LEARNING_RATE = 0.001;

    private final boolean REGULARIZATION = true;
    private final int SEED = 7899;

    private Evaluation eval = new Evaluation();

    private int iterationNumber = 0;

    MultiLayerNetwork network;

    private int outputLabelCount = 2;

    Logger log = LoggerFactory.getLogger(this.getClass());

    public RichNeuralNet(int outputLabelCount) {
        this.outputLabelCount = outputLabelCount;
    }

    public MultiLayerNetwork getNet() {
        return network;
    }

    public void init(MultiLayerNetwork net) {
        this.network = net;
    }

    public void initCnn(int height) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .iterations(ITERATIONS)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(1)
                        .nOut(96)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        .nIn(20)
                        .nOut(1)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build())
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputLabelCount)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, height, 1))
                .build();
        network = new MultiLayerNetwork(conf);
    }

    public void initTextDetector() {
        int outputNumber = 2;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .iterations(ITERATIONS)
                .regularization(REGULARIZATION).l2(0.0005)
                .learningRate(.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)

                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(1)
                        .stride(2, 2)
                        .nOut(1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())


                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(100).build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNumber)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(32, 32, 1))
                .backprop(true).pretrain(true).build();


        network = new MultiLayerNetwork(conf);
    }

    public void initLenetMnist() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .iterations(ITERATIONS)
                .regularization(true).l2(0.0005)
                .learningRate(LEARNING_RATE).biasLearningRate(0.02)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS) //To configure: .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(9, 9)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputLabelCount)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(32, 32, 1))
                .build();

        network = new MultiLayerNetwork(conf);
    }

    public void train(DataSetIterator data) {
        long before = System.nanoTime();
        iterationNumber++;
        log.debug("working on iteration #" + iterationNumber);
        network.fit(data);
        long after = System.nanoTime();
        log.debug("done. took " + ((after - before) / 100_000_000) + "seconds.");
    }

    public void train(INDArray data) {
        network.fit(data);
    }

    public void eval(INDArray input, INDArray actual) {
        INDArray output = network.output(input);
        eval.eval(actual, output);
    }

    public String getStats() {
        return eval.stats();
    }

    public void saveNetwork(String filename) {
        try {
            ModelSerializer.writeModel(network, filename, true);
        } catch (IOException e) {
            log.error(e.getMessage(), e);
        }
    }

    public MultiLayerNetwork loadNetwork(String filename) throws IOException {
        return this.network = ModelSerializer.restoreMultiLayerNetwork(filename);
    }
}
