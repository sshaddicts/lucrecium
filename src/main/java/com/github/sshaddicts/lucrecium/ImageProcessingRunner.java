package com.github.sshaddicts.lucrecium;

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor;
import com.github.sshaddicts.lucrecium.imageProcessing.Imshow;
import com.github.sshaddicts.lucrecium.imageProcessing.Validator;
import com.github.sshaddicts.lucrecium.util.FileInteractions;
import org.opencv.core.Core;

import java.util.Objects;
import java.util.Scanner;


public class ImageProcessingRunner {
    public static String[] files = new String[]{"test_data/kek.png",
            "test_data/HelloWorld.jpg",
            "test_data/Checke.jpg",
            "test_data/Checke2.jpg",
            "test_data/TheOnlyPhotoIFound.jpg",
            "test_data/test.png",
            "test_data/bakal.jpg"};

    public static String[] realData = new String[]{
            "real_data/Domins.jpg",
            "real_data/IMG_20170617_124412.jpg",
            "real_data/IMG_20170617_124431.jpg",
            "real_data/IMG_20170617_124454.jpg",
            "real_data/IMG_20170617_124523.jpg",
            "real_data/IMG_20170617_124605.jpg",
            "real_data/IMG_20170617_124640.jpg",
            "real_data/IMG_20170617_124656.jpg",
            "real_data/IMG_20170617_124709.jpg",
            "real_data/IMG_20170617_124752.jpg",
            "real_data/IMG_20170617_124813.jpg",
            "real_data/IMG_20170617_124826.jpg",
            "real_data/IMG_20170617_124839.jpg",
            "real_data/IMG_20170617_124904.jpg",
            "real_data/IMG_20170617_124918.jpg",
            "real_data/IMG_20170617_125012.jpg",
            "real_data/IMG_20170617_125030.jpg",
            "real_data/IMG_20170617_125042.jpg",
            "real_data/IMG_20170617_125053.jpg",
            "real_data/IMG_20170617_125108.jpg",
            "real_data/IMG_20170617_125130.jpg",
            "real_data/IMG_20170617_125142.jpg",
            "real_data/IMG_20170617_125157.jpg",
            "real_data/IMG_20170617_125209.jpg",
            "real_data/IMG_20170617_125220.jpg",
            "real_data/IMG_20170617_125232.jpg",
            "real_data/IMG_20170617_125250.jpg",
            "real_data/IMG_20170617_125304.jpg",
            "real_data/IMG_20170617_125333.jpg",
            "real_data/IMG_20170617_125346.jpg",
            "real_data/IMG_20170617_125359.jpg",
            "real_data/IMG_20170617_125411.jpg"

    };

    public static int RESIZE_PERCENTAGE = 25;
    public static int PICTURE_NUMBER = 4;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Validator.MAX_AREA_THRESHOLD = 1000;
        Validator.ASPECT_RATIO = 2 / 1;

        Scanner scanner = new Scanner(System.in);

        boolean letsGo = true;
        //while (letsGo) {

            ImageProcessor processor = new ImageProcessor(realData[PICTURE_NUMBER]);

            processor.delta = 0;
            processor.minArea = 25;
            processor.maxArea = Integer.MAX_VALUE;
            processor.maxVariation = 5;
            processor.minDiversity = 5;
            processor.maxEvolution = 500;
            processor.areaThreshold = 5;
            processor.minMargin = 5;
            processor.edgeBlurSize = 0;

            processor.preProcess();

            //Imshow test = new Imshow("test");

            //test.showImage(processor.detectAndRemoveBoundary());


            processor.detectMSER();
            processor.drawROI();

            Imshow shower = new Imshow("mser test");
            shower.showImage(processor.getImage());
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

//            String answer = scanner.next();
//
//            if (Objects.equals(answer, "x")) {
//                PICTURE_NUMBER++;
//            } else if (Objects.equals(answer, "z")) {
//                PICTURE_NUMBER--;
//            }
//            else{
//                letsGo = false;
//            }
//        }

    }


    public static void testOverlapping() {
        ImageProcessor processor = new ImageProcessor(realData[PICTURE_NUMBER]);

        processor.resize();

        processor.cropImage();

        Imshow shower = new Imshow("overlapTest");

        processor.drawROI();
        shower.showImage(processor.getImage());

    }

    public static void saveCroppedImage() {

        ImageProcessor processor = new ImageProcessor(realData[PICTURE_NUMBER]);

        processor.resize();

        FileInteractions.saveMat(processor.cropImage());
    }

    public void saveLetters() {

        ImageProcessor processor = new ImageProcessor(realData[PICTURE_NUMBER]);
        ImageProcessor testProcessor = new ImageProcessor(realData[PICTURE_NUMBER]);


        Imshow shower = new Imshow("kek");
        Imshow test = new Imshow("test");
        shower.setResizable(true);
        test.setResizable(true);


        processor.resize();
        processor.cropImage();

        processor.drawROI();

        testProcessor.resize();

        shower.showImage(processor.getImage());
        test.showImage(testProcessor.cropImage());

        FileInteractions.saveMats(processor.getMats());
    }
}