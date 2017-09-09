package com.github.sshaddicts.lucrecium.util;

import org.apache.commons.io.FileUtils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class FileInteractions {

    private static Logger log = LoggerFactory.getLogger(FileInteractions.class);

    public static void saveMats(List<Mat> list, String directory) {
        for (Mat mat : list) {
            saveMat(mat, directory);
        }
    }

    public static void saveMat(Mat mat, String directory) {
        File f = new File(directory);

        if(!f.exists()){
            if(!f.mkdir()){
                throw new IllegalArgumentException("directory is hacked: " + directory);
            }
        }
        Imgcodecs.imwrite(directory + "/image_" + System.currentTimeMillis() + ".png", mat);
    }

    public static List<String> getFileNamesFromDir(String parentDir) {
        List<String> filenames = new ArrayList<>();

        File parentDirectory = FileUtils.getFile(parentDir);
        File[] files = parentDirectory.listFiles();

        assert files != null;
        for (File file : files) {
            if (file.isFile()) {
                filenames.add(parentDir + "/" + file.getName());
            }
        }
        return filenames;
    }

    public static void saveMatWithName(Mat mat, String directory, String filename) {
        Imgcodecs.imwrite(directory + "/" + directory + "_" + filename + ".png", mat);
    }
}
