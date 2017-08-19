package com.github.sshaddicts.lucrecium.util;

import org.ejml.simple.SimpleMatrix;
import org.opencv.core.Mat;

/**
 * Created by Alex on 06.08.2017.
 */
public class MatrixOperations {

    public static SimpleMatrix createVector(int size) {
        return new SimpleMatrix(size, 1);
    }

    public static SimpleMatrix createVectorFrom(double[] data) {
        SimpleMatrix result = new SimpleMatrix(data.length, 1);

        for (int i = 0; i < data.length; i++) {
            result.set(i, 0, data[i]);
        }

        return result;
    }

    public static SimpleMatrix createVectorFrom(byte[] data) {
        SimpleMatrix result = new SimpleMatrix(data.length, 1);

        for (int i = 0; i < data.length; i++) {
            result.set(i, 0, data[i]);
        }

        return result;
    }

    public static SimpleMatrix createSimpleMatrixFromMat(Mat mat) {
        SimpleMatrix result;
        int x = mat.rows();
        int y = mat.cols();

        byte[] data = new byte[x * y];

        mat.get(0, 0, data);

        result = createVectorFrom(data);
        result.reshape(x, y);

        return result;
    }

    public static SimpleMatrix prependToVector(SimpleMatrix target, SimpleMatrix destination) {
        SimpleMatrix result = new SimpleMatrix(target.numRows() + destination.numRows(), 1);
        result.insertIntoThis(0, 0, target);
        result.insertIntoThis(target.numRows(), 0, destination);
        return result;
    }

    public static SimpleMatrix prependBias(SimpleMatrix target) {
        SimpleMatrix result = new SimpleMatrix(target.numRows() + 1, 1);
        result.insertIntoThis(0, 0, vectorOfOnes(1));
        result.insertIntoThis(1, 0, target);
        return result;
    }

    public static SimpleMatrix vectorOfOnes(int rows) {
        double[][] data = new double[rows][1];
        for (int i = 0; i < rows; i++)
            data[i][0] = 1;
        return new SimpleMatrix(data);
    }
}
