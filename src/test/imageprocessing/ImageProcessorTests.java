package imageprocessing;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.SearchResult;
import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;
import com.github.sshaddicts.lucrecium.util.RectManipulator;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;

import static org.junit.Assert.*;

public class ImageProcessorTests {

    static {
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private final String testCaseDir = "testCase/";
    private final String example = testCaseDir + "good.jpg";
    private final String skewed = testCaseDir + "skew.jpg";
    private final String lineSegm = testCaseDir + "lineNumber.png";

    @Test
    public void imageContainsLetters() {
        ImageProcessor processor = new ImageProcessor();
        SearchResult result = processor.findTextRegions(example);
        List<CharContainer> words = result.getChars();

        assertTrue("mat.size() is " + words.size() + ", but should never be", words.size() > 0);
    }

    @Test
    public void testDeskewing() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        ImageProcessor processor = new ImageProcessor();

        Method deskew = processor.getClass().getDeclaredMethod("deskew", Mat.class);
        deskew.setAccessible(true);

        Double result = (Double) deskew.invoke(processor, ImageProcessor.loadImage(skewed));
        assertFalse(result == 0);
    }

    @Test
    public void testLineNumber() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        ImageProcessor processor = new ImageProcessor();

        Method lineNumberFor = processor.getClass().getDeclaredMethod("approximateLineNumberFor", Mat.class);
        lineNumberFor.setAccessible(true);

        Integer invoke = (Integer) lineNumberFor.invoke(processor, ImageProcessor.loadImage(lineSegm));
        assertTrue(invoke == 3);
    }

    @Test
    public void testSplit() throws InterruptedException {

        Mat image = ImageProcessor.loadImage(lineSegm);
        Rect rect = new Rect(0, 0, image.width(), image.height());

        int mean = 22;
        List<Rect> split = RectManipulator.split(rect, rect.height / mean, false);

        assertTrue(split.size() == 3);
    }

    @Test
    public void testMerge() {
        Rect rect1 = new Rect(0, 0, 25, 25);
        Rect rect2 = new Rect(25, 25, 25, 25);

        Rect merged = RectManipulator.merge(rect1, rect2, 0);

        assertEquals(50, merged.width);
    }
}
