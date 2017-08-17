package com.topin212;

import com.topin212.imageProcessing.ImageProcessor;
import com.topin212.imageProcessing.Imshow;
import com.topin212.imageProcessing.Validator;
import org.opencv.core.*;

public class Runner {
    public static String[] files = new String[]{"test_data/kek.png",
                                                "test_data/HelloWorld.jpg",
                                                "test_data/Checke.jpg",
                                                "test_data/Checke2.jpg",
                                                "test_data/TheOnlyPhotoIFound.jpg",
                                                "test_data/test.png",
                                                "test_data/bakal.jpg"};

    public static int RESIZE_PERCENTAGE = 25;
    public static int PICTURE_NUMBER = 2;



    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Validator.MAX_AREA_THRESHOLD = 1000;
        Validator.ASPECT_RATIO = 2/1;


        ImageProcessor processor = new ImageProcessor(files[PICTURE_NUMBER]);
        ImageProcessor testProcessor = new ImageProcessor(files[PICTURE_NUMBER]);


        Imshow shower = new Imshow("kek");
        Imshow test = new Imshow("test");
        shower.setResizable(true);
        test.setResizable(true);


        processor.resize(RESIZE_PERCENTAGE);
        processor.cropImage();

        processor.drawROI();

        testProcessor.resize(RESIZE_PERCENTAGE);

        shower.showImage(processor.getImage());
        test.showImage(testProcessor.cropImage());

        FileInteractions.saveMats(processor.getMats());
    }
}