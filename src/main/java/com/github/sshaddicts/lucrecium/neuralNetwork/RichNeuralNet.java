package com.github.sshaddicts.lucrecium.neuralNetwork;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RichNeuralNet {
    private int ITERATIONS = 10000;
    private double LEARNING_RATE = 0.1;

    private boolean USE_DROP_CONNECT = false;
    private boolean MINI_BATCH = false;
    private boolean PRETRAIN = false;
    private boolean BACKPROP = true;
    private Activation ACTIVATION_FUNC = Activation.SIGMOID;

    private DataSet data;

    public void setData(DataSet data) {
        this.data = data;
    }

    NeuralNetConfiguration conf;
    MultiLayerConfiguration multilayerConf;
    MultiLayerNetwork network;

    private int layers = 0;

    public MultiLayerNetwork getNet() {
        return network;
    }

    public void addLayer(Layer layer){
    }

    public void init(int inputSize, int hiddenNumber, int hiddenSize, int classNumber) {
        layers = hiddenNumber + 1;
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();

        builder.iterations(ITERATIONS);
        builder.learningRate(LEARNING_RATE);
        builder.useDropConnect(USE_DROP_CONNECT);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

        builder.biasInit(0);
        builder.miniBatch(MINI_BATCH);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

        listBuilder.layer(0, configureLayer(inputSize, hiddenSize));

        for (int i = 1; i < layers - 1; i++) {
            listBuilder.layer(i, configureLayer(hiddenSize, hiddenSize));
        }

        OutputLayer.Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);

        outputLayerBuilder.nIn(hiddenSize);
        outputLayerBuilder.nOut(classNumber);

        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        outputLayerBuilder.dist(new UniformDistribution(0, 1));
        listBuilder.layer(layers - 1, outputLayerBuilder.build());

        listBuilder.pretrain(PRETRAIN);
        listBuilder.backprop(BACKPROP);

        multilayerConf = listBuilder.build();
        network = new MultiLayerNetwork(multilayerConf);
    }

    public void train() {
        network.fit(data);
    }

    public DenseLayer configureLayer(int nIn, int nOut) {
        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        hiddenLayerBuilder.nIn(nIn);
        hiddenLayerBuilder.nOut(nOut);
        hiddenLayerBuilder.activation(ACTIVATION_FUNC);
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

        return hiddenLayerBuilder.build();
    }
}
