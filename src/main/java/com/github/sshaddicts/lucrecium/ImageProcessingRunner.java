package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.Validator;
import com.github.sshaddicts.lucrecium.util.FileInteractions;
import org.opencv.core.Core;


public class ImageProcessingRunner {
    public static String[] files = new String[]{"test_data/kek.png",
            "test_data/HelloWorld.jpg",
            "test_data/Checke.jpg",
            "test_data/Checke2.jpg",
            "test_data/TheOnlyPhotoIFound.jpg",
            "test_data/test.png",
            "test_data/bakal.jpg"};

    public static String[] realData = new String[]{
            "real_data/IMG_20170617_124412.jpg",
            "real_data/Domins.jpg"
    };

    public static int RESIZE_PERCENTAGE = 25;
    public static int PICTURE_NUMBER = 0;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Validator.MAX_AREA_THRESHOLD = 1000;
        Validator.ASPECT_RATIO = 2 / 1;

        ImageProcessor processor = new ImageProcessor(realData[PICTURE_NUMBER]);

        processor.resize(50);

        processor.cropImage();

        Imshow shower = new Imshow("kek");

        processor.drawROI();
        shower.showImage(processor.getImage());
    }


    public static void saveCroppedImage() {

        ImageProcessor processor = new ImageProcessor(realData[PICTURE_NUMBER]);

        processor.resize(50);

        FileInteractions.saveMat(processor.cropImage());
    }

    public void saveLetters() {

        ImageProcessor processor = new ImageProcessor(realData[PICTURE_NUMBER]);
        ImageProcessor testProcessor = new ImageProcessor(realData[PICTURE_NUMBER]);


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