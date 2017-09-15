package com.github.sshaddicts.lucrecium.neuralNetwork;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TextRecognizer {

    private RichNeuralNet net;

    private Logger log = LoggerFactory.getLogger(this.getClass());

    public TextRecognizer(String filename) throws IOException {
        net = new RichNeuralNet(RichNeuralNet.loadNetwork(filename));
    }


    //TODO refactor
    public List<ObjectNode> getText(List<CharContainer> containers) throws IOException {
        List<ObjectNode> entries = new ArrayList<>();

        NativeImageLoader loader = new Java2DNativeImageLoader(32, 32, 1);

        int containerSize = containers.size();

        log.debug("container size " + containerSize);

        StringBuilder sb = new StringBuilder();

        int prevY = containers.get(0).getRect().y;

        final JsonNodeFactory factory = JsonNodeFactory.instance;

        for (int i = 0; i < containerSize; i++) {
            Rect rect = containers.get(i).getRect();

            int currentY = rect.y;

            //the line ends at this condition
            if (Math.abs(currentY - prevY) > 10) {
                ObjectNode entry = factory.objectNode();

                double v = Double.parseDouble(sb.toString());
                entry.put("entry_" + i, v);

                entries.add(entry);
                sb.delete(0, sb.length());
            }

            prevY = currentY;

            Mat slice = containers.get(i).getMat();
            BufferedImage bufferedSlice = ImageProcessor.toBufferedImage(slice);

            INDArray array = loader.asMatrix(bufferedSlice);

            Integer in = net.predict(array)[0];
            sb.append(in);
        }

        ObjectNode entry = factory.objectNode();

        double v = Double.parseDouble(sb.toString());
        entry.put("last_entry", v);
        entries.add(entry);

        System.out.println(sb.toString());
        return entries;
    }
}
