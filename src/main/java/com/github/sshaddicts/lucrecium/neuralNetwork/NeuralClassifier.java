package com.github.sshaddicts.lucrecium.neuralNetwork;


import com.github.sshaddicts.lucrecium.util.MatrixOperations;
import org.ejml.alg.dense.mult.MatrixDimensionException;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Alex on 02.08.2017.
 */
//TODO add setClasses method
public class NeuralClassifier implements Classifier {

    private SimpleMatrix[] a;
    private SimpleMatrix[] z;
    private SimpleMatrix[] theta;

    private SimpleMatrix[] error;

    private int classNumber;

    public NeuralClassifier(int inputSize, int hiddenNumber, int hiddenSize, int classNumber){
        //a's size is +2 because we treat all layers as hidden
        this.a = new SimpleMatrix[hiddenNumber + 2];
        this.z = new SimpleMatrix[hiddenNumber + 2];
        this.classNumber = classNumber;

        //setting up layers
        for(int i = 1; i < hiddenNumber + 1; i++){
            a[i] = MatrixOperations.createVector(hiddenSize + 1);
        }

        //setting up input/output layers
        this.a[0] = MatrixOperations.createVector(inputSize + 1);
        this.a[hiddenNumber+1] = MatrixOperations.createVector(classNumber);

        //for hidden size = 3
        //input -> hidden1 -> hidden2 -> hidden3 -> output
        this.theta = new SimpleMatrix[hiddenNumber + 1];
        this.error = theta.clone();

        Random random = new Random();

        for(int i = 0; i< hiddenNumber + 1; i++){
            theta[i] = SimpleMatrix.random(a[i+1].numRows(), a[i].numRows(),
                                          -1, 1, random);
        }
    }


    private SimpleMatrix hypothesis(SimpleMatrix input){

        if(input.numCols() != a[a.length-1].numCols() && input.numRows() != a[a.length-1].numRows()){
            throw new IllegalArgumentException("Input has inacceptible dimensions.");
        }

        CommonOps.insert(vectorOfOnes(1).combine(SimpleMatrix.END, 0, input).getMatrix(), a[0].getMatrix(), 0,0);

        for(int i = 1; i<a.length; i++){

            z[i] = theta[i-1].mult(a[i-1]);
            a[i] = g(z[i]);

        }

        return a[a.length-1];
    }

    public SimpleMatrix[] computeGradient(SimpleMatrix input, SimpleMatrix actual){
        if(input.numCols() != a[0].numCols() && input.numRows() != a[0].numRows()){
            throw new IllegalArgumentException("Input has inacceptible dimensions.");
        }

        if(actual.numRows() != classNumber){
            throw new IllegalArgumentException("Actual data has inacceptible dimensions.");
        }

        //forward propagation
        SimpleMatrix predicted = hypothesis(input);

        //cost determination
        error[error.length - 1] = actual.minus(predicted);


        for(int i = error.length - 2; i>0; i--){
            error[i] = (theta[i+1].transpose().mult(error[i+1])).
                    elementMult(a[i].elementMult(vectorOfOnes(a[i].numRows()).minus(a[i])));
        }

        return error;
    }

    public void backPropagate(){
        for(int i = 1; i<theta.length; i++){
            theta[i].elementMult(error[i]);
        }
    }

    public SimpleMatrix g(SimpleMatrix z){
        for(int x = 0; x < z.numRows(); x++){
            for(int y = 0; y < z.numCols(); y++){
                z.set(x,y,1.0/(1+Math.exp(-z.get(x,y))));
            }
        }
        return z;
    }

    private SimpleMatrix vectorOfOnes(int rows){
        double[][] data = new double[rows][1];
        for(int i = 0; i<rows; i++)
            data[i][0] = 1;
        return new SimpleMatrix(data);
    }
}
