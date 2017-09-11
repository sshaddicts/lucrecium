package com.github.sshaddicts.lucrecium.neuralNetwork;

import com.github.sshaddicts.lucrecium.datasets.ImageDataSet;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;
import com.github.sshaddicts.lucrecium.imageProcessing.containers.WordContainer;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public class TextRecognizer {
    private Mat image;

    private RichNeuralNet net;

    private Logger log = LoggerFactory.getLogger(this.getClass());

    public TextRecognizer(int outputLabels) {
        net = new RichNeuralNet(outputLabels);
        initNet();
    }

    public TextRecognizer(String filename) throws IOException {
        initNet(filename);
    }


    //TODO set offset correctly, for now outputs a one-line string
    public String getText(List<CharContainer> containers) throws IOException {
        NativeImageLoader loader = new Java2DNativeImageLoader(32, 32, 1);

        int containerSize = containers.size();

        log.debug("container size " + containerSize);

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < containerSize; i++) {
            Integer in = net.getNet().predict(loader.asMatrix(ImageProcessor.toBufferedImage(containers.get(i).getMat())))[0];
            sb.append(in).append(" ");
        }

        return sb.toString();
    }

    private void initNet() {
        net.initLenetMnist();
    }

    private void initNet(String filename) throws IOException {
        net = new RichNeuralNet(RichNeuralNet.loadNetwork(filename));
    }
}
