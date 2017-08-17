package com.github.sshaddicts.lucrecium.neuralNetwork;


import com.github.sshaddicts.lucrecium.util.MatrixOperations;
import org.ejml.alg.dense.mult.MatrixDimensionException;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Alex on 02.08.2017.
 */
//TODO rewrite with arrays and loops
//TODO add setClasses method
public class TextRecognizer {

    private SimpleMatrix[] a;
    private SimpleMatrix[] z;
    private SimpleMatrix[] theta;

    private SimpleMatrix[] error;

    public TextRecognizer(int inputSize, int hiddenNumber, int hiddenSize, int classNumber){
        //a's size is +2 because we treat all layers as hidden
        this.a = new SimpleMatrix[hiddenNumber + 2];
        this.z = new SimpleMatrix[hiddenNumber];

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
            theta[i] = SimpleMatrix.random(a[i+1].numRows(), a[i].numRows() + 1,
                                          0.001, 2, random);
        }
    }


    private SimpleMatrix hypothesis(SimpleMatrix input){

        for(int i = 1; i<a.length; i++){
            theta[i-1].printDimensions();
            a[i-1].printDimensions();


            z[i] = theta[i].mult(a[i-1]);
            a[i] = g(z[i]);
        }

        System.out.println("Predicted dimensions are:");
        a[a.length-1].printDimensions();

        return a[a.length-1];
    }

    public SimpleMatrix[] computeGradient(SimpleMatrix input, SimpleMatrix actual){

        //forward propagation
        SimpleMatrix predicted = hypothesis(input);

        //cost determination
        error[error.length - 1] = actual.minus(predicted);

        for(int i = error.length - 1; i>1; i--){
            error[i] = a[i].transpose().mult(error[i+1]).elementMult(
                    a[i].elementMult(vectorOfOnes(a[i].numRows()).minus(a[i])));
        }

        return error;
    }

    public SimpleMatrix g(SimpleMatrix z){
        for(int x = 0; x < z.numRows(); x++){
            for(int y = 0; y < z.numCols(); y++){
                z.set(x,y,1/(1-Math.exp(-z.get(x,y))));
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
