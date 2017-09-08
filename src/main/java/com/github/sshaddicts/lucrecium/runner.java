package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.util.FileInteractions;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.lang.reflect.InvocationTargetException;
import java.util.List;

public class runner {

    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static final List<String> filenames = FileInteractions.getFileNamesFromDir("dataSources/greatData");

    public static void main(String[] args) throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
//        int number = ThreadLocalRandom.current().nextInt(0,filenames.size());
        int number = 0;
        ImageProcessor processor = new ImageProcessor(filenames.get(number));
        processor.needsRotation(false);

        ImageProcessor processor1 = new ImageProcessor(filenames.get(number));
        processor.needsRotation(true);


        List<Mat> textRegions = processor.getTextRegions(ImageProcessor.MERGE_CHARS);
        processor1.getTextRegions(ImageProcessor.MERGE_CHARS);

        FileInteractions.saveMats(textRegions, "outputData/charImages");
    }
}
