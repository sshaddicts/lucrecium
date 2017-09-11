package neuralnet;

import com.github.sshaddicts.lucrecium.datasets.ImageDataSet;
import com.github.sshaddicts.lucrecium.neuralNetwork.RichNeuralNet;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertTrue;


//TODO transfer labels here somehow
public class NeuralNetTests {
    private List<String> labels;
    private ImageDataSet dataSet;
    private RichNeuralNet net;


    @Before
    public void init() throws IOException {
        dataSet = new ImageDataSet(3,1);
        dataSet.initFromDirectory("testCase/nnTest");
        labels = dataSet.getIterator().getLabels();
        net = new RichNeuralNet(10);
        net.loadNetwork("netFile");
    }

    @Test
    public void testCharDetection() throws IOException {
        DataSetIterator iterator = dataSet.getIterator();

        List<Integer> finalOutput = new ArrayList<>();

        for(int i = 0; iterator.hasNext(); i++){
             finalOutput.add(net.getNet().predict(iterator.next().getFeatureMatrix())[0]);
        }

        boolean containsAll = finalOutput.contains(1);
        containsAll &= finalOutput.contains(3);
        containsAll &= finalOutput.contains(6);

        assertTrue(containsAll);
    }
}
