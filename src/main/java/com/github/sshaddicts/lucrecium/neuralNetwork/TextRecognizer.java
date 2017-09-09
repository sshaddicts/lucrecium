package com.github.sshaddicts.lucrecium.neuralNetwork;

import com.github.sshaddicts.lucrecium.datasets.ImageDataSet;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.WordContainer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class TextRecognizer {
    private Mat image;
    private WordContainer charContainer;

    private RichNeuralNet net = new RichNeuralNet(2);
    private ImageDataSet dataSet;

    private Logger log = LoggerFactory.getLogger(this.getClass());

    public TextRecognizer() throws IOException {
        initNet();

        dataSet = new ImageDataSet(2, 20);
        dataSet.initFromDirectory("daTestf");
    }

    public String getText(int i) {

        ImageProcessor proc = new ImageProcessor("testCase/good.jpg");

        int height = proc.getImage().height();
        int width = proc.getImage().width();


        for (int x = 0; x < width-32; x++) {
            for (int y = 0; y < height-32; y++) {
                INDArray array = ImageProcessor.toNdarray(proc.submat(x,y,32,32)).reshape(32,32,1);

                log.debug(array.shapeInfoToString());
                net.train(array);
            }
        }
        return "wip";
    }

    public RichNeuralNet getNet() {
        return net;
    }

    private void initNet() {
        net.initLenetMnist();
    }
}
