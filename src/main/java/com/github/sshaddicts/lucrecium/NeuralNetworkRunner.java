package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.neuralNetwork.NeuralClassifier;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.util.MatrixOperations;
import org.ejml.simple.SimpleMatrix;
import org.opencv.core.Core;

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
            "test_data/bakal.jpg",
            "test_data/Test.bmp"
    };


    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        testOnImages();

    }
    public static void testOnXOR(){
        double[][] inputSet = new double[][]{
                {0,0},
                {1,0},
                {0,1},
                {1,1}
        };

        double[] outputSet = new double[]{
                0,
                1,
                1,
                0
        };

        NeuralClassifier classifier = new NeuralClassifier(2,1,1,1);

        classifier.computeGradient(MatrixOperations.createVectorFrom(new double[]{0,1}), MatrixOperations.createVectorFrom(new double[]{1}));


    }


    public static void testOnImages(){

        ImageProcessor testImage = new ImageProcessor(files[7]);

        testImage.resize(20);

        byte[] imageData = new byte[testImage.getImage().width() * testImage.getImage().height()];
        testImage.getImage().get(0,0,imageData);
        SimpleMatrix matrixFromMat = MatrixOperations.createVectorFrom(imageData);

        double[] outputdata = new double[]{1, 0, 0};

        NeuralClassifier recognizer = new NeuralClassifier(
                testImage.getImage().height() * testImage.getImage().width(),
                3,
                5,
                3);

        SimpleMatrix[] gradient = recognizer.computeGradient(matrixFromMat, MatrixOperations.createVectorFrom(outputdata));

        for (int i = 0; i< gradient.length; i++) {
            if(gradient[i] != null )gradient[i].print();
        }

        recognizer.backPropagate();
    }
}
