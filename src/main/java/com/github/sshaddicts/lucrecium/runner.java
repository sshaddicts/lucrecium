package com.github.sshaddicts.lucrecium;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.sshaddicts.lucrecium.datasets.ImageDataSet;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;
import com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet;
import com.github.sshaddicts.lucrecium.neuralNetwork.TextRecognizer;
import com.github.sshaddicts.lucrecium.util.FileInteractions;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class runner {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final List<String> filenames = FileInteractions.getFileNamesFromDir("dataSources/greatData");
    private static final Logger log = LoggerFactory.getLogger(runner.class);
    private static List<String> labels;

    public static void main(String[] args) throws IOException {
        ObjectMapper mapper = new ObjectMapper();


        recognizeDigits();

    }

    public static void recognizeDigits() throws IOException {
        ImageProcessor processor = new ImageProcessor("testCase/numbers31.jpg");


        TextRecognizer recognizer = new TextRecognizer("netFile");

        List<CharContainer> textRegions = processor.findTextRegions(ImageProcessor.NO_MERGE);

        String text = recognizer.getText(textRegions);

        System.out.println(text);
        Imshow.show(processor.getImage(), "processed");

        Imgcodecs.imwrite("overlay.png", processor.getOverlay());
    }

    public static void testNetwork() throws IOException {
        RichNeuralNet net = new RichNeuralNet(10);
        net.loadNetwork("netFile");

        ImageDataSet dataSet = new ImageDataSet(3, 1);
        dataSet.initFromDirectory("testData/TrainF");

        DataSetIterator iterator = dataSet.getIterator();

        while (iterator.hasNext()) {
            int[] output = net.getNet().predict(iterator.next().getFeatureMatrix());
            System.out.println(Arrays.toString(output));
        }
        System.out.println(2 + 2);
    }

    public static void trainNetwork(int epochs) throws IOException {
        ImageDataSet dataSet = new ImageDataSet(10, 28);
        dataSet.initFromDirectory("testData/testF");

        RichNeuralNet net = new RichNeuralNet(10);

        net.initLenetMnist();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.getNet().setListeners(new StatsListener(statsStorage));

        for (int i = 0; i < epochs; i++) {
            net.train(dataSet.getIterator());
        }

        net.saveNetwork("netFile");

        labels = dataSet.getRecordReader().getLabels();
    }

}
