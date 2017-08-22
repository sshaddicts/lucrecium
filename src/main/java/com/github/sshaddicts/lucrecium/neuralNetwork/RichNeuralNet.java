package com.github.sshaddicts.lucrecium.neuralNetwork;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RichNeuralNet {
    private int ITERATIONS = 10000;
    private double LEARNING_RATE = 0.001;

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
    private int classNumber;

    public MultiLayerNetwork getNet() {
        return network;
    }

    public void addLayer(Layer layer){
    }

    public void init(int inputSize, int hiddenNumber, int hiddenSize, int classNumber) {
        layers = hiddenNumber + 1;
        this.classNumber = classNumber;
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

        listBuilder.setInputType(InputType.convolutional(18,9,1));

        multilayerConf = listBuilder.build();
        network = new MultiLayerNetwork(multilayerConf);
    }

    public void init(int numRows, int numColumns){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(ITERATIONS)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(1, new RBM.Builder().nIn(100).nOut(50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(2, new RBM.Builder().nIn(50).nOut(25).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(3, new RBM.Builder().nIn(25).nOut(10).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(4, new RBM.Builder().nIn(10).nOut(3).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //encoding stops
                .layer(5, new RBM.Builder().nIn(3).nOut(10).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //decoding starts
                .layer(6, new RBM.Builder().nIn(10).nOut(25).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(7, new RBM.Builder().nIn(25).nOut(50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(8, new RBM.Builder().nIn(50).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(100).nOut(10).build())
                .pretrain(true).backprop(true).setInputType(InputType.convolutional(18,9,1))
                .build();

        network = new MultiLayerNetwork(conf);
    }

    public void init(MultiLayerNetwork net){
        this.network = net;
    }

    public void train() {
        network.fit(data);
        INDArray output = network.output(data.getFeatureMatrix());
        Evaluation eval = new Evaluation(classNumber);
        eval.eval(data.getLabels(), output);

        System.out.println(eval.stats());
    }

    private DenseLayer configureLayer(int nIn, int nOut) {
        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        hiddenLayerBuilder.nIn(nIn);
        hiddenLayerBuilder.nOut(nOut);
        hiddenLayerBuilder.activation(ACTIVATION_FUNC);
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

        return hiddenLayerBuilder.build();
    }

    public void eval(){
        INDArray output = network.output(data.getFeatureMatrix());
        Evaluation eval = new Evaluation(classNumber);
        eval.eval(data.getLabels(), output);

        System.out.println(eval.stats());
    }
}
