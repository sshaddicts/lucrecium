package com.topin212.neuralNetwork;


import com.topin212.util.MatrixOperations;
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

    private SimpleMatrix input;

    private SimpleMatrix weights1;
    private SimpleMatrix weights2;
    private SimpleMatrix weights3;

    private SimpleMatrix z1;
    private SimpleMatrix z2;
    private SimpleMatrix z3;

    private SimpleMatrix a1;
    private SimpleMatrix a2;
    private SimpleMatrix a3;

    private SimpleMatrix[] a;
    private SimpleMatrix[] z;
    private SimpleMatrix[] theta;

    private SimpleMatrix[] error;

    public TextRecognizer(int imageHeight, int imageWidth, int hidden1Size, int hidden2Size, int classNumber){
        this.input = new SimpleMatrix(imageHeight * imageWidth + 1, 1);
        this.a1 = new SimpleMatrix(hidden1Size,1);
        this.a2 = new SimpleMatrix(hidden2Size,1);
        this.a3 = new SimpleMatrix(classNumber,1);

        Random r = new Random();

        this.weights1 = SimpleMatrix.random(a1.numRows(),input.numRows(), 0.1,2, r);
        this.weights2 = SimpleMatrix.random(a2.numRows(),a1.numRows() + 1, 0.1,2, r);
        this.weights3 = SimpleMatrix.random(a3.numRows(),a2.numRows() + 1, 0.1,2, r);
    }

    public TextRecognizer(int inputSize, int hiddenNumber, int hiddenSize, int classNumber){
        //a's size is +2 because we treat all layers as hidden =)
        this.a = new SimpleMatrix[hiddenNumber + 2];
        this.z = new SimpleMatrix[hiddenNumber];

        //setting up layers
        for(int i = 1; i < hiddenNumber - 1; i++){
            a[i] = MatrixOperations.createVector(hiddenSize);
        }

        //setting up input/output layers
        this.a[0] = MatrixOperations.createVector(inputSize);
        this.a[hiddenNumber] = MatrixOperations.createVector(classNumber);

        //for hidden size = 3
        //input -> hidden1 -> hidden2 -> hidden3 -> output
        this.theta = new SimpleMatrix[hiddenNumber + 1];
        this.error = theta.clone();

        for(int i = 0; i< hiddenNumber + 1; i++){
            theta[i] = SimpleMatrix.random(a[i+1].numRows(), a[i].numRows() + 1,
                                          0.001, 2, new Random());
        }
    }


    private SimpleMatrix hypothesis(SimpleMatrix input){

        for(int i = 1; i<a.length; i++){
            z[i] = theta[i].mult(a[i-1]);
            a[i] = g(z[i]);
        }

        System.out.println("Predicted dimensions are:");
        a[a.length-1].printDimensions();

        return a[a.length-1];
    }

    public List<SimpleMatrix> computeGradient(SimpleMatrix input, SimpleMatrix actual){

        //forward propagation
        SimpleMatrix predicted = hypothesis(input);

        //cost determination
        error[error.length - 1] = actual.minus(predicted);

        for(int i = error.length - 1; i>1; i--){
            error[i] = a[i].transpose().mult(error[i+1]).elementMult(
                    a[i].elementMult(vectorOfOnes(a[i].numRows()).minus(a[i])));
        }

        //backpropagation

        DenseMatrix64F tmp = new DenseMatrix64F();

        SimpleMatrix hidden1Error;
        SimpleMatrix hidden2Error;

        //TODO check delta dimensions
        //I guess it's similar to theta
        //and it actually is. dimensions are: error(1+l) x (a(l)T)
        //5,5!
        List<SimpleMatrix> Delta = new ArrayList<>();

        DenseMatrix64F h1res = new DenseMatrix64F();

        //basically error is predicted - real

        //forward propagate
        SimpleMatrix propagated = hypothesis(input);

        hidden2Error = predicted.minus(propagated);

        Delta.add(hidden2Error.mult(a3.transpose()));

        try{
            org.ejml.ops.CommonOps.elementMult(
                    //hidden1 should be transposed
                    //this one should be 2 x 6
                    //when transposed it becomes 6x2, and we can multiply it by 2 x 1 hidden2error
                    ((a1.transpose()).mult(hidden2Error)).getMatrix(),
                    vectorOfOnes(a3.numRows()).minus(a3).getMatrix(), h1res);

        }catch(MatrixDimensionException | IllegalArgumentException e){
            System.out.println("Error during calculating error for layer 1");
            System.out.println(e.getMessage());
            a1.transpose().printDimensions();
            hidden2Error.printDimensions();
            System.out.println("------------------------------------------");
            vectorOfOnes(7).printDimensions();
            a3.printDimensions();
        }

        hidden1Error = new SimpleMatrix(SimpleMatrix.wrap(h1res));

        Delta.add(hidden1Error.mult(a2.transpose()));

        return Delta;
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
