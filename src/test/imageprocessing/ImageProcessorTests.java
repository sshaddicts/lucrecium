package imageprocessing;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.Validator;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.List;

import static org.junit.Assert.assertTrue;

public class ImageProcessorTests {

    ImageProcessor processor = new ImageProcessor("real_data/good.jpg");

    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    }

    @Test
    public void imageContainsLetters(){
        List<Mat> mats = processor.getText();

        assertTrue("mat.size() is " + mats.size() + ", but should never be", mats.size() > 0);
    }
}
