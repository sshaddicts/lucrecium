package neuralnet;

import com.github.sshaddicts.lucrecium.datasets.ImageDataSet;
import com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.IOException;

public class NeuralNetTests {

    private final String neuralNetDataDir = "testData";
    private final int outputLabelCount = 10;
    private final int batchSize = 128;
    private final int numberOfEpochs = 100;

    private final String netFileName = "netFile";

    private final boolean toggleUI = false;
    private final boolean verbose = false;

    Logger logger = LoggerFactory.getLogger(this.getClass());


    @Test
    public void testNetwork() {
        ImageDataSet dataSet = new ImageDataSet(neuralNetDataDir, outputLabelCount, batchSize);

        RichNeuralNet net = new RichNeuralNet(outputLabelCount);
        net.init(18, 9);

        if (toggleUI) {
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            net.getNet().setListeners(new StatsListener(statsStorage));
        }

        for (int i = 0; i < numberOfEpochs; i++) {
            net.train(dataSet.getIterator());
        }

        if (verbose) {
            Toolkit.getDefaultToolkit().beep();
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            Toolkit.getDefaultToolkit().beep();
        }

        net.saveNetwork(netFileName);
    }

    @Test
    public void testNetworkOnMnist() {
        MnistDataSetIterator iter = null;
        try {
            iter = new MnistDataSetIterator(batchSize, outputLabelCount);
        } catch (IOException e) {
            logger.trace(e.getMessage(), e);
        }

        RichNeuralNet net = new RichNeuralNet(outputLabelCount);
        net.init(28, 28);

        if (toggleUI) {
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            net.getNet().setListeners(new StatsListener(statsStorage));
        }

        for (int i = 0; i < numberOfEpochs; i++) {
            net.train(iter);
        }

        if (verbose) {
            Toolkit.getDefaultToolkit().beep();
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            Toolkit.getDefaultToolkit().beep();
        }

        net.saveNetwork("mnist" + netFileName);
    }
}
