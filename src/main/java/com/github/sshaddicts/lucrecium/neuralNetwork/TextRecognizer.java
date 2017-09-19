package com.github.sshaddicts.lucrecium.neuralNetwork;

import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class TextRecognizer {

    private RichNeuralNet net;

    private Logger log = LoggerFactory.getLogger(this.getClass());

    private ObjectMapper mapper = new ObjectMapper();

    public TextRecognizer(String filename) throws IOException {
        net = new RichNeuralNet(RichNeuralNet.loadNetwork(filename));
    }

    public TextRecognizer(InputStream is) throws IOException {
        net = new RichNeuralNet(RichNeuralNet.loadnetwork(is));
    }


    //TODO refactor
    public List<ObjectNode> recognize(List<CharContainer> containers) throws IOException {
        List<ObjectNode> entries = new ArrayList<>(containers.size());

        NativeImageLoader loader = new Java2DNativeImageLoader(32, 32, 1);

        log.debug("container size " + containers.size());

        StringBuilder sb = new StringBuilder();

        int prevY = containers.get(0).getRect().y;

        for (int i = 0; i < containers.size(); i++) {
            Rect rect = containers.get(i).getRect();

            int currentY = rect.y;

            //the line ends at this condition
            if (Math.abs(currentY - prevY) > 10) {
                entries.add((ObjectNode) mapper.valueToTree(new Occurrence(
                        "entry_" + i,
                        Double.parseDouble(sb.toString())
                )));
                sb.setLength(0);
            }

            prevY = currentY;

            Mat slice = containers.get(i).getMat();
            BufferedImage bufferedSlice = ImageProcessor.toBufferedImage(slice);

            INDArray array = loader.asMatrix(bufferedSlice);

            Integer in = net.predict(array)[0];
            sb.append(in);
        }

        entries.add((ObjectNode) mapper.valueToTree(new Occurrence(
                "last_entry",
                Double.parseDouble(sb.toString())
        )));

        return entries;
    }
}
