package com.github.sshaddicts.lucrecium.util;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;

/**
 * Created by Alex on 01.08.2017.
 */
public class FileInteractions {

    public static void saveMats(List<Mat> list) {
        for (Mat mat :
                list) {
            Imgcodecs.imwrite("wordImages/test_" + System.nanoTime() + ".png", mat);
        }
    }

    public static void saveMat(Mat mat) {
        Imgcodecs.imwrite("cropped_Images/image_" + System.currentTimeMillis() + ".png", mat);
    }

    public static void saveMatTo(Mat mat, String filename){
        Imgcodecs.imwrite(filename, mat);
    }
}
