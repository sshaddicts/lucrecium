package com.github.sshaddicts.lucrecium;

import org.apache.commons.io.FileUtils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;

/**
 * Created by Alex on 01.08.2017.
 */
public class FileInteractions {

    public static void saveMats(List<Mat> list){
        for (Mat mat :
                list) {
            Imgcodecs.imwrite("testImages/test_" + System.nanoTime() + ".png", mat);
        }
    }
}
