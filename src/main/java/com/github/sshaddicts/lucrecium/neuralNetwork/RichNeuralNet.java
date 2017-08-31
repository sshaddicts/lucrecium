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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.IOException;


public class RichNeuralNet {
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

    private int classNumber = 2;

    public RichNeuralNet(int classNumber) {
        this.classNumber = classNumber;
    }

    public MultiLayerNetwork getNet() {
        return network;
    }

    public void init(MultiLayerNetwork net) {
        this.network = net;
    }

    public void init(int numRows, int numCols) {
        System.out.println(String.format("using %d as input size", numRows * numCols));
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123) //include a random seed for reproducibility
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
                .iterations(1)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(LEARNING_RATE) //specify the learning rate
                .updater(new Nesterovs(0.98))
                .regularization(true).l2(LEARNING_RATE * 0.005) // regularize learning model
                .list()
                .layer(0, new DenseLayer.Builder() //create the first input layer.
                        .nIn(numRows * numCols)
                        .nOut(500)
                        .build())
                .layer(1, new DenseLayer.Builder() //create the second input layer
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(Activation.SOFTMAX)
                        .nIn(100)
                        .nOut(classNumber)
                        .build())
                .setInputType(InputType.convolutionalFlat(numRows, numCols, 1))
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();

        network = new MultiLayerNetwork(conf);
        network.init();
    }

    public void train(DataSetIterator data) {
        network.fit(data);
    }

    public void eval(INDArray input, INDArray actual) {
        INDArray output = network.output(input);
        eval = new Evaluation(classNumber);
        eval.eval(actual, output);
    }

    public void printStats() {
        System.out.println(eval.stats());
    }

    public static void saveNetwork(MultiLayerNetwork net, String filename) {
        try {
            ModelSerializer.writeModel(net, filename, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MultiLayerNetwork loadNetwork(String filename) throws IOException {

        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(filename);

        return multiLayerNetwork;

    }
}
