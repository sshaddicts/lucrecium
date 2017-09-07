package imageprocessing;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static org.junit.Assert.assertTrue;

public class ImageProcessorTests {

    private final String filename = "testCase/good.jpg";

    Logger log = LoggerFactory.getLogger(this.getClass());

    static {

        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @Test
    public void imageContainsLetters() {
        ImageProcessor processor = new ImageProcessor(filename);
        List<Mat> mats = processor.getTextRegions();
        assertTrue("mat.size() is " + mats.size() + ", but should never be", mats.size() > 0);
    }

    @Test
    public void testImageProcessing() {
        ImageProcessor processor = new ImageProcessor(filename);
        processor.detectText(ImageProcessor.MERGE_CHARS);
    }
}
