package imageprocessing;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
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
    public void imageContainsLetters() throws IOException {
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
