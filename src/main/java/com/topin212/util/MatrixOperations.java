package com.topin212.util;

import org.ejml.simple.SimpleMatrix;
import org.opencv.core.Mat;

/**
 * Created by Alex on 06.08.2017.
 */
public class MatrixOperations {

    public static SimpleMatrix createVector(int size){
        return new SimpleMatrix(size, 1);
    }
    public static SimpleMatrix createVectorFrom(double[] data){
        SimpleMatrix result = new SimpleMatrix(data.length, 1);

        for(int i = 0; i<data.length; i++){
            result.set(i,0,data[i]);
        }

        return result;
    }
    public static SimpleMatrix createVectorFrom(byte[] data){
        SimpleMatrix result = new SimpleMatrix(data.length, 1);

        for(int i = 0; i<data.length; i++){
            result.set(i,0,data[i]);
        }

        return result;
    }

    public static SimpleMatrix createSimpleMatrixFromMat(Mat mat){
        SimpleMatrix result;
        int x = mat.rows();
        int y = mat.cols();

        byte[] data = new byte[x * y];

        mat.get(0,0,data);

        result = createVectorFrom(data);
        result.reshape(x,y);

        return result;
    }
}
