package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.datasets.DummyDataSet;
import com.github.sshaddicts.lucrecium.datasets.ImageDataSet;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.Validator;
import com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet;
import com.github.sshaddicts.lucrecium.util.FileInteractions;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.awt.*;
import java.io.IOException;


public class lucreciumRunner {

    public static int PICTURE_NUMBER = 5;
    public static int CLASS_NUMBER = 10;
    public static int BATCH_SIZE = 30;
    public static int NUMBER_OF_EPOCHS = 1000;

    public static String NETWORK_DATA_DIR = "test_kek";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Validator.MIN_AREA_THRESHOLD = 5;
        Validator.MAX_AREA_THRESHOLD = 1000;
        Validator.ASPECT_RATIO = 2 / 1;
        Validator.MIN_ASPECT_RATIO = 1 / 2;


        Imshow newShow = new Imshow("character detecting");
        ImageProcessor processor = new ImageProcessor(DummyDataSet.oldData[PICTURE_NUMBER]);

        processor.computeSkewAndProcess();
        processor.detectText(ImageProcessor.MERGE_WORDS);
        newShow.showImage(processor.getImage());

        FileInteractions.saveMats(processor.getMats());
    }

    public static void testSavedNetwork(){
        RichNeuralNet network = new RichNeuralNet();

        try {
            network.init(loadNetwork("netFile"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        ImageDataSet dataset = new ImageDataSet(NETWORK_DATA_DIR, CLASS_NUMBER, BATCH_SIZE);

        while(dataset.hasNext()){
            DataSet next = dataset.next();

            network.eval(next.getFeatureMatrix(), next.getLabels());
        }
        network.printStats();
    }


    public static void testNetwork() {
        ImageDataSet dataSet = new ImageDataSet(NETWORK_DATA_DIR, CLASS_NUMBER, BATCH_SIZE);

        RichNeuralNet net = new RichNeuralNet();
        net.init(18 * 9, 10, 10, CLASS_NUMBER);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.getNet().setListeners(new StatsListener(statsStorage));

        //net.init(18,9);

        //net.init();

        for (int i = 0; i < NUMBER_OF_EPOCHS; i++) {
            net.train(dataSet.getIterator());
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

    public static void saveNetwork(MultiLayerNetwork net, String filename) {
        try {
            ModelSerializer.writeModel(net, filename, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MultiLayerNetwork loadNetwork(String filename) throws IOException {

        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(filename);

        return multiLayerNetwork;

    }

    public static void createCharacterDataSet() {
        int length = DummyDataSet.realData.length;

        for (int i = 0; i < length; i++) {

            ImageProcessor processor = new ImageProcessor(DummyDataSet.realData[i]);

            processor.delta = 0;
            processor.minArea = 25;
            processor.maxArea = 21 * 852;
            processor.maxVariation = Integer.MAX_VALUE;
            processor.minDiversity = 5;
            processor.maxEvolution = 5;
            processor.areaThreshold = 5000;
            processor.minMargin = 5;
            processor.edgeBlurSize = 0;


            FileInteractions.saveMats(processor.getText());
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            System.out.println(i);
        }
    }

    public static void saveCroppedImage() {
        ImageProcessor processor = new ImageProcessor(DummyDataSet.realData[PICTURE_NUMBER]);
        processor.preProcess();
        processor.save("file_" + System.currentTimeMillis() + ".png");
    }

    public static void saveProcessorInfo(ImageProcessor processor) {
        FileInteractions.saveMatTo(processor.getImage(), String.format(
                "mser_data/picture = %d; delta = %d;" +
                        "minArea = %d;" +
                        "maxArea = %d;" +
                        "maxVariation = %d;" +
                        "minDiversity = %d;" +
                        "maxEvolution = %d;" +
                        "areaThreshold = %d;" +
                        "minMargin = %d;" +
                        "edgeBlurSize = %d;.jpg",
                PICTURE_NUMBER, processor.delta, processor.minArea, processor.maxArea,
                processor.maxVariation, processor.minDiversity,
                processor.maxEvolution, processor.areaThreshold,
                processor.minMargin, processor.edgeBlurSize
        ));

    }
}