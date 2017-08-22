package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.neuralNetwork.NeuralClassifier;
import com.github.sshaddicts.lucrecium.util.MatrixOperations;
import org.ejml.simple.SimpleMatrix;
import org.nd4j.linalg.jcublas.compression.CudaThreshold;
import org.opencv.core.Core;

import java.util.Random;

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

        demoTest();
    }

    public static void demoTest() {
        NeuralClassifier classifier = new NeuralClassifier(2, 1, 2, 1);
        SimpleMatrix[] inputData = new SimpleMatrix[]{
                MatrixOperations.createVectorFrom(new double[]{0, 0}),
                MatrixOperations.createVectorFrom(new double[]{0, 1}),
                MatrixOperations.createVectorFrom(new double[]{1, 0}),
                MatrixOperations.createVectorFrom(new double[]{1, 1})
        };

        SimpleMatrix[] outputData = new SimpleMatrix[]{
                MatrixOperations.createVectorFrom(new double[]{0}),
                MatrixOperations.createVectorFrom(new double[]{1}),
                MatrixOperations.createVectorFrom(new double[]{1}),
                MatrixOperations.createVectorFrom(new double[]{0})
        };

        Random random = new Random();

        for (int i = 0; i < 3000; i++) {
            int randomNumber = random.nextInt(4);
            classifier.train(inputData[randomNumber], outputData[randomNumber]);
        }

        xorErrorForClassifier(classifier);
        xorHypothesisForClassifier(classifier);
    }

    public static void xorErrorForClassifier(NeuralClassifier classifier) {
        double a, b, c, d;

        a = classifier.costFunction(MatrixOperations.createVectorFrom(new double[]{0, 0}), MatrixOperations.createVectorFrom(new double[]{0}));
        b = classifier.costFunction(MatrixOperations.createVectorFrom(new double[]{1, 0}), MatrixOperations.createVectorFrom(new double[]{1}));
        c = classifier.costFunction(MatrixOperations.createVectorFrom(new double[]{0, 1}), MatrixOperations.createVectorFrom(new double[]{1}));
        d = classifier.costFunction(MatrixOperations.createVectorFrom(new double[]{1, 1}), MatrixOperations.createVectorFrom(new double[]{0}));

        System.out.println("0, 0 -> " + a);
        System.out.println("1, 0 -> " + b);
        System.out.println("0, 1 -> " + c);
        System.out.println("1, 1 -> " + d);
    }

    public static void xorHypothesisForClassifier(NeuralClassifier classifier) {
        classifier.hypothesis(MatrixOperations.createVectorFrom(new double[]{0, 0})).print();
        classifier.hypothesis(MatrixOperations.createVectorFrom(new double[]{1, 0})).print();
        classifier.hypothesis(MatrixOperations.createVectorFrom(new double[]{0, 1})).print();
        classifier.hypothesis(MatrixOperations.createVectorFrom(new double[]{1, 1})).print();
    }
}
