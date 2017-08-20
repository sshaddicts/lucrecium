package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.datasets.DummyDataSet;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessingException;
import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.Validator;
import com.github.sshaddicts.lucrecium.util.FileInteractions;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


public class ImageProcessingRunner {

    public static int PICTURE_NUMBER = 0;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Validator.MIN_AREA_THRESHOLD = 5;
        Validator.MAX_AREA_THRESHOLD = 1000;
        Validator.ASPECT_RATIO = 2 / 1;

        ImageProcessor processor = new ImageProcessor(DummyDataSet.realData[PICTURE_NUMBER]);

        processor.delta = 0;
        processor.minArea = 25;
        processor.maxArea = Integer.MAX_VALUE;
        processor.maxVariation = Integer.MAX_VALUE;
        processor.minDiversity = 5;
        processor.maxEvolution = 5;
        processor.areaThreshold = 5000;
        processor.minMargin = 5;
        processor.edgeBlurSize = 0;

        try {
            processor.preProcess();
        } catch (ImageProcessingException e) {
            System.out.println(e.getInfo());
        }
        processor.drawROI();

        Imshow shower = new Imshow("test");
        shower.showImage(processor.getImage());

        FileInteractions.saveMats(processor.getMats());
    }

    public static void saveCroppedImage() throws ImageProcessingException {
        ImageProcessor processor = new ImageProcessor(DummyDataSet.realData[PICTURE_NUMBER]);
        processor.preProcess();
        processor.save("file_" + System.currentTimeMillis() + ".png");
    }


    public static void saveProcessorInfo(ImageProcessor processor){
        FileInteractions.saveMatTo(processor.getImage(), String.format(
                "mser_data/picture = %d; delta = %d;" +
                        "minArea = %d;" +
                        "maxArea = %d;" +
                        "maxVariation = %d;" +
                        "minDiversity = %d;" +
                        "maxEvolution = %d;" +
                        "areaThreshold = %d;" +
                        "minMargin = %d;" +
                        "edgeBlurSize = %d;.jpg",
                PICTURE_NUMBER, processor.delta, processor.minArea, processor.maxArea,
                processor.maxVariation, processor.minDiversity,
                processor.maxEvolution, processor.areaThreshold,
                processor.minMargin, processor.edgeBlurSize
        ));

    }
}