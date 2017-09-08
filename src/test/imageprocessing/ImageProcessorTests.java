package imageprocessing;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ImageProcessorTests {

    private final String testCaseDir = "testCase/";

    private final String example = testCaseDir + "good.jpg";
    private final String skewed = testCaseDir + "skew.jpg";
    private final String lineSegm = testCaseDir + "lineNumber.png";

    Logger log = LoggerFactory.getLogger(this.getClass());

    static {
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @Test
    public void imageContainsLetters() {
        ImageProcessor processor = new ImageProcessor(example);
        List<Mat> mats = processor.getTextRegions(ImageProcessor.MERGE_LINES);
        assertTrue("mat.size() is " + mats.size() + ", but should never be", mats.size() > 0);
    }

    @Test
    public void testDeskewing() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        ImageProcessor processor = new ImageProcessor(skewed);

        Method deskew = processor.getClass().getDeclaredMethod("deskew", Mat.class);
        deskew.setAccessible(true);

        Double result = (Double)deskew.invoke(processor, processor.getImage());
        assertFalse(result == 0);
    }

    @Test
    public void testLineNumber() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        ImageProcessor processor = new ImageProcessor(lineSegm);

        Method lineNumberFor = processor.getClass().getDeclaredMethod("approximateLineNumberFor", Mat.class);
        lineNumberFor.setAccessible(true);

        Integer invoke = (Integer) lineNumberFor.invoke(processor, processor.getImage());
        assertTrue(invoke == 3);
    }
}
