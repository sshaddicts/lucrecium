package com.github.sshaddicts.lucrecium.neuralNetwork;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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


public class RichNeuralNet{
    private final int ITERATIONS = 1;
    private final double LEARNING_RATE = 0.006;

    private final boolean USE_DROP_CONNECT = false;
    private final boolean MINI_BATCH = false;
    private final boolean PRETRAIN = false;
    private final boolean BACKPROP = true;
    private final Activation ACTIVATION_FUNC = Activation.RELU;
    private final boolean REGULARIZATION = true;

    private Evaluation eval = new Evaluation();

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

    public void init(int numRows, int numCols) {
        Log.debug(String.format("using %d as input size", numRows * numCols));
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(LEARNING_RATE)
                .updater(new Nesterovs(0.98))
                .regularization(true).l2(LEARNING_RATE * 0.005)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numRows * numCols)
                        .nOut(500)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(100)
                        .nOut(outputLabelCount)
                        .build())
                .setInputType(InputType.convolutionalFlat(numRows, numCols, 1))
                .pretrain(false).backprop(true)
                .build();

        network = new MultiLayerNetwork(conf);
        network.init();
    }

    public void train(DataSetIterator data) {
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
            e.printStackTrace();
        }
    }

    public MultiLayerNetwork loadNetwork(String filename) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(filename);
    }
}
