package com.github.sshaddicts.lucrecium.neuralNetwork;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;
import com.github.sshaddicts.lucrecium.util.RectManipulator;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

public class TextRecognizer {

    private RichNeuralNet net;

    private Logger log = LoggerFactory.getLogger(this.getClass());

    public TextRecognizer(String filename) throws IOException {
        net = new RichNeuralNet(RichNeuralNet.loadNetwork(filename));
    }

    //TODO set offset correctly, for now outputs a one-line string
    public String getText(List<CharContainer> containers) throws IOException {
        NativeImageLoader loader = new Java2DNativeImageLoader(32, 32, 1);

        int containerSize = containers.size();

        log.debug("container size " + containerSize);

        StringBuilder sb = new StringBuilder();

        int prevX = containers.get(0).getRect().x;
        int prevY = containers.get(0).getRect().y;


        for (int i = 0; i < containerSize; i++) {
            int currentY = containers.get(i).getRect().y;

            if(Math.abs(currentY - prevY) > 10){
                sb.append("\n");
            }

            prevY = currentY;


            Mat slice = containers.get(i).getMat();
            BufferedImage bufferedSlice = ImageProcessor.toBufferedImage(slice);

            INDArray array = loader.asMatrix(bufferedSlice);

            Integer in = net.predict(array)[0];
            sb.append(in);
        }

        return sb.toString();
    }
}
