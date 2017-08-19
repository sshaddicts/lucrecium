package com.github.sshaddicts.lucrecium.neuralNetwork;


import com.github.sshaddicts.lucrecium.util.MatrixOperations;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Created by Alex on 02.08.2017.
 */
//TODO add setClasses method;
//TODO add train method
//TODO add dataset integration

public class NeuralClassifier implements Classifier {

    private double LEARNING_RATE = 0.02;
    private double TRAINING_EXAMPLES = 4;
    private double REGULARIZATION_TERM = 0;

    private SimpleMatrix[] a;
    private SimpleMatrix[] z;
    private SimpleMatrix[] theta;

    private SimpleMatrix[] error;

    private SimpleMatrix[] big_delta;
    private SimpleMatrix[] D;

    private int inputSize;
    private int hiddenNumber;
    private int hiddenSize;
    private int classNumber;

    private int layers;

    public void setLearningRate(double LEARNING_RATE) {
        this.LEARNING_RATE = LEARNING_RATE;
    }

    public void setTrainingExamples(double TRAINING_EXAMPLES) {
        this.TRAINING_EXAMPLES = TRAINING_EXAMPLES;
    }

    public void setRegularizationValue(double REGULARIZATION_TERM) {
        this.REGULARIZATION_TERM = REGULARIZATION_TERM;
    }

    public NeuralClassifier(int inputSize, int hiddenNumber, int hiddenSize, int classNumber) {
        //a's size is +2 because we treat all layers as hidden
        this.layers = hiddenNumber + 2;
        this.a = new SimpleMatrix[layers];
        this.z = new SimpleMatrix[layers];
        this.inputSize = inputSize;
        this.hiddenNumber = hiddenNumber;
        this.hiddenSize = hiddenSize;
        this.classNumber = classNumber;

        init();

        big_delta = theta.clone();
        D = new SimpleMatrix[big_delta.length];
    }

    private void init() {
        //setting up layers
        for (int i = 1; i < hiddenNumber + 1; i++) {
            a[i] = MatrixOperations.createVector(hiddenSize);
        }

        //setting up input/output layers
        this.a[0] = MatrixOperations.createVector(inputSize + 1);
        this.a[hiddenNumber + 1] = MatrixOperations.createVector(classNumber);

        //for hidden size = 3
        //input -> hidden1 -> hidden2 -> hidden3 -> output
        this.theta = new SimpleMatrix[layers - 1];
        this.error = new SimpleMatrix[layers];

        Random random = new Random();
        random.setSeed(System.currentTimeMillis());

        //randomizing weight values
        theta[0] = SimpleMatrix.random(a[1].numRows(), a[0].numRows(), 2, 2, random);
        for (int i = 1; i < theta.length; i++) {
            theta[i] = SimpleMatrix.random(a[i + 1].numRows(), a[i].numRows() + 1,
                    -0.5, 0.5, random);
        }
    }

    private void setInput(SimpleMatrix input) {
        if (input.numCols() != a[a.length - 1].numCols() && input.numRows() != a[a.length - 1].numRows()) {
            throw new IllegalArgumentException("Input has inacceptible dimensions.");
        }

        a[0] = MatrixOperations.prependBias(input);
    }

    public void train(SimpleMatrix input, SimpleMatrix actual) {
        setInput(input);
        computeLayerError(input, actual);
        computeAndUpdateWeights();
        init();
    }

    public SimpleMatrix hypothesis(SimpleMatrix input) {
        setInput(input);

        for (int i = 1; i < a.length; i++) {
            z[i] = theta[i - 1].mult(a[i - 1]);

            //add bias unit to z before passing it to the activation function:
            SimpleMatrix tmp = MatrixOperations.prependToVector(MatrixOperations.vectorOfOnes(1), z[i]);

            a[i] = g(tmp);
        }

        a[a.length - 1] = theta[a.length - 2].mult(a[a.length - 2]);

        return a[a.length - 1];
    }

    public void computeLayerError(SimpleMatrix input, SimpleMatrix actual) {
        init();

        if (input.numCols() != 1 && input.numRows() != a[0].numRows()) {
            throw new IllegalArgumentException("Input has inacceptable dimensions.");
        }

        if (actual.numRows() != classNumber) {
            throw new IllegalArgumentException("Actual data has inacceptable dimensions.");
        }

        //forward propagation
        SimpleMatrix predicted = hypothesis(input);

        //layer error determination
        error[error.length - 1] = actual.minus(predicted);
        for (int i = error.length - 2; i > 0; i--) {
            error[i] = (theta[i].transpose().mult(error[i + 1])).
                    elementMult(a[i].elementMult(MatrixOperations.vectorOfOnes(a[i].numRows()).minus(a[i]))).extractMatrix(1, SimpleMatrix.END, 0, SimpleMatrix.END);
        }
    }

    public SimpleMatrix[] computeAndUpdateWeights() {
        SimpleMatrix[] newTheta = theta.clone();

        for (int i = 0; i < layers - 2; i++) {
            big_delta[i] = big_delta[i].plus(error[i + 1].mult(a[i].transpose()));
        }

        for (int l = 0; l < D.length; l++) {
            D[l] = new SimpleMatrix(big_delta[l].numRows(), big_delta[l].numCols());
            for (int i = 0; i < big_delta[l].numRows(); i++) {
                for (int j = 0; j < big_delta[l].numCols(); j++) {
                    double value = (1 / TRAINING_EXAMPLES) * big_delta[l].get(i, j);
                    if (j != 0)
                        value += REGULARIZATION_TERM * theta[l].get(i, j);
                    D[l].set(i, j, value);
                }
            }
        }

//        for(int l = 0; l < D.length; l++){
//            for(int i = 0; i<theta[l].numRows(); i++){
//                for(int j = 0; j<theta[l].numCols(); j++){
//                    //old value + learning rate * error * derivative * output of previos neuron
//                    double value = theta[l].get(i, j)  + LEARNING_RATE * error[l+1].get(i,0) * D[l].get(i,j) * a[l].get(i,0);
//                    newTheta[l].set(i, j, value);
//                }
//            }
//        }

        for (int i = layers - 2; i > 0; i--) {
            newTheta[i] = theta[i].plus((D[i].scale(LEARNING_RATE)));
        }

        return big_delta;
    }

    public double costFunction(SimpleMatrix input, SimpleMatrix actual) {
        double returnValue = 0;

        SimpleMatrix predicted = hypothesis(input);

        for (int i = 0; i < classNumber; i++) {
            returnValue += (actual.get(i, 0) * Math.log(predicted.get(i, 0))) +
                    ((1 - actual.get(i, 0)) * Math.log(1 - predicted.get(i, 0)));
        }
        return returnValue / TRAINING_EXAMPLES;
    }

    private SimpleMatrix g(SimpleMatrix z) {
        for (int x = 0; x < z.numRows(); x++) {
            for (int y = 0; y < z.numCols(); y++) {
                z.set(x, y, 1.0 / (1 + Math.exp(-z.get(x, y))));
            }
        }
        return z;
    }
}
