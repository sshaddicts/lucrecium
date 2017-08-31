package imageprocessing;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.List;

import static org.junit.Assert.assertTrue;

public class ImageProcessorTests {

    private final String filename = "testCase/good.jpg";


    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }


    @Test
    public void imageContainsLetters() {
        ImageProcessor processor = new ImageProcessor(filename);
        List<Mat> mats = processor.getText(ImageProcessor.MERGE_CHARS);
        assertTrue("mat.size() is " + mats.size() + ", but should never be", mats.size() > 0);
    }

    @Test
    public void testImageProcessing() {
        ImageProcessor processor = new ImageProcessor(filename);

        processor.computeSkewAndProcess();
        processor.detectText(ImageProcessor.MERGE_CHARS);
    }
}
