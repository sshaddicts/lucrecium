package com.topin212;

import com.topin212.imageProcessing.ImageProcessor;
import com.topin212.neuralNetwork.TextRecognizer;
import com.topin212.util.MatrixOperations;
import org.ejml.simple.SimpleMatrix;
import org.opencv.core.Core;

import java.util.List;

/**
 * Created by Alex on 02.08.2017.
 */
public class NeuralNetworkRunner {

    public static String[] files = new String[]{"test_data/kek.png",
            "test_data/HelloWorld.jpg",
            "test_data/Checke.jpg",
            "test_data/Checke2.jpg",
            "test_data/TheOnlyPhotoIFound.jpg",
            "test_data/test.png",
            "test_data/bakal.jpg"};


    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        ImageProcessor testImage = new ImageProcessor(files[6]);

        testImage.resize(20);

        byte[] imageData = new byte[testImage.getImage().width() * testImage.getImage().height()];
        testImage.getImage().get(0,0,imageData);
        SimpleMatrix matrixFromMat = MatrixOperations.createVectorFrom(imageData);

        double[] outputdata = new double[]{1, 0};

        TextRecognizer recognizer = new TextRecognizer(testImage.getImage().height(),testImage.getImage().width(),5,5, 2);


        List<SimpleMatrix> gradient = recognizer.computeGradient(matrixFromMat, MatrixOperations.createVectorFrom(outputdata));

        for (SimpleMatrix matrix :
                gradient) {
            System.out.println(matrix);
        }


    }
}
