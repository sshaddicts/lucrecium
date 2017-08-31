package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.datasets.DummyDataSet;
import com.github.sshaddicts.lucrecium.datasets.ImageDataSet;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.Validator;
import com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.opencv.core.Core;

import java.awt.*;
import java.io.IOException;

import static com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet.loadNetwork;
import static com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet.saveNetwork;


public class lucreciumRunner {

    private final static int PICTURE_NUMBER = 2;
    private final static int CLASS_NUMBER = 10;
    private final static int BATCH_SIZE = 30;
    private final static int NUMBER_OF_EPOCHS = 1000;

    private static String NETWORK_DATA_DIR = "test_kek";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Validator.MIN_AREA_THRESHOLD = 5;
        Validator.MAX_AREA_THRESHOLD = 1000;
        Validator.ASPECT_RATIO = 2 / 1;
        Validator.MIN_ASPECT_RATIO = 1 / 2;


        testNetwork();
    }

    public static void testImageProcessing() {
        Imshow newShow = new Imshow("kinda new processing");
        Imshow shower = new Imshow("experimental processing");
        ImageProcessor processor = new ImageProcessor(DummyDataSet.oldData[PICTURE_NUMBER]);
        ImageProcessor expProc = new ImageProcessor(DummyDataSet.oldData[PICTURE_NUMBER]);

        processor.computeSkewAndProcess();
        processor.detectText(ImageProcessor.MERGE_CHARS);
        newShow.showImage(processor.getImage());

        shower.showImage(expProc.experimentalProcessing());

    }

    public static void testSavedNetwork() {
        RichNeuralNet network = new RichNeuralNet(CLASS_NUMBER);

        try {
            network.init(loadNetwork("netFile"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        ImageDataSet dataset = new ImageDataSet(NETWORK_DATA_DIR, CLASS_NUMBER, BATCH_SIZE);

        while (dataset.hasNext()) {
            DataSet next = dataset.next();

            network.eval(next.getFeatureMatrix(), next.getLabels());
        }
        network.printStats();
    }


    public static void testNetwork() {
        ImageDataSet dataSet = new ImageDataSet(NETWORK_DATA_DIR, CLASS_NUMBER, BATCH_SIZE);

        DataSetIterator iter = null;
        try {
            iter = new MnistDataSetIterator(128, 10);
        } catch (IOException e) {
            e.printStackTrace();
        }

        RichNeuralNet net = new RichNeuralNet(CLASS_NUMBER);
        net.init(28, 28);


        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.getNet().setListeners(new StatsListener(statsStorage));

        for (int i = 0; i < NUMBER_OF_EPOCHS; i++) {
            net.train(iter);
        }

        Toolkit.getDefaultToolkit().beep();
        try {
            Thread.sleep(200);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Toolkit.getDefaultToolkit().beep();

        saveNetwork(net.getNet(), "netFile");
    }
}