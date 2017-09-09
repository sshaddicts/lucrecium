package imageprocessing;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.WordContainer;
import com.github.sshaddicts.lucrecium.util.RectManipulator;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;

import static org.junit.Assert.assertEquals;
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
        List<WordContainer> words = processor.getTextRegions(ImageProcessor.NO_MERGE);
        assertTrue("mat.size() is " + words.size() + ", but should never be", words.size() > 0);
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

    @Test
    public void testSplit() throws InterruptedException {
        ImageProcessor processor = new ImageProcessor(lineSegm);
        Rect rect = new Rect(0,0,processor.getImage().width(), processor.getImage().height());

        int mean = 22;
        List<Rect> split = RectManipulator.split(rect, rect.height / mean, false);

        assertTrue(split.size() == 3);
    }

    @Test
    public void testMerge(){
        Rect rect1 = new Rect(0,0,25,25);
        Rect rect2 = new Rect(25,25,25,25);

        Rect merged = RectManipulator.merge(rect1, rect2, 0);

        assertEquals(50, merged.width);
    }
}
