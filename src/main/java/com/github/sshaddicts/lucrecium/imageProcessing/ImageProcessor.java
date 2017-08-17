package com.github.sshaddicts.lucrecium.imageProcessing;

import org.opencv.core.*;
import org.opencv.features2d.MSER;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Created by Alex on 29.07.2017.
 */
//TODO extend rect class with validation methods
public class ImageProcessor {

    private Mat image;
    private List<MatOfPoint> regions;
    private MatOfRect rect;

    private List<Rect> rois;
    private static int DEFAULT_DISTANCE_THRESHOLD = 5;

    public ImageProcessor(String filename){
        if(filename == null || Objects.equals(filename, "")){
            throw new IllegalArgumentException("Filename cannot be null or empty.");
        }

        this.image = Imgcodecs.imread(filename, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        this.regions = new ArrayList<>();
        this.rect = new MatOfRect();
        this.rois = new ArrayList<>();
    }

    public Mat testDrawRoi(){
        detectMSER();
        Mat imagecopy = image.clone();
        Imgproc.drawContours(imagecopy, regions, 0, new Scalar(10));
        return imagecopy;
    }

    public void drawROI(){
        detectMSER();
        //Imgproc.drawContours(image, regions, 1, new Scalar(0));
        List<Rect> rects = rect.toList();

        for(Rect rect : rects){
            if(Validator.isValidTextArea(rect)){
                //Imgproc.rectangle(image, rect.tl(), rect.br(), new Scalar(92));
                rois.add(rect);
            }
        }
    }

    public void resize(double percentage){
        double height, width;
        Mat tempMat = new Mat();
        height = image.height() * (percentage/100);
        width = image.width() * (percentage/100);
        Imgproc.resize(image, tempMat, new Size(width, height));
        image = tempMat;
    }

    public Mat cropImage(){
        Mat tempMat = new Mat();
        List<MatOfPoint> countours = new ArrayList<>();
        Mat hierarchy = new Mat();
        int mode = Imgproc.RETR_LIST;
        int method = Imgproc.CHAIN_APPROX_SIMPLE;
        Imgproc.threshold(image, tempMat, 75, 255, CvType.CV_8UC1);
        Imgproc.findContours(tempMat, countours, hierarchy, mode, method);

        Mat result = new Mat();

        for (Mat mat :
                countours) {
            if (Imgproc.contourArea(mat)>4000 ){
                Rect rect = Imgproc.boundingRect((MatOfPoint) mat);
                if(rect.height > 50)
                    result = image.submat(rect);
            }
        }

        return image = result;
    }

    //TODO rewrite using FeatureDetector
    public void detectMSER(){
        Mat tempMat = image.clone();
        Imgproc.blur(image, tempMat, new Size(image.width(), image.height()));
        MSER mser = MSER.create(5,
                20,
                Integer.MAX_VALUE,
                5,
                5,
                5,
                5,
                0,
                0);
        mser.detectRegions(image, regions, rect);
    }

    public List<MatOfPoint> getRegions(){
        return regions;
    }

    public Mat getImage() {
        return image;
    }

    public List<Mat> getMats(){
        List<Mat> returnList = new ArrayList<>();

        for (Rect rectang:
             rois) {
            returnList.add(image.submat(rectang));
        }

        return returnList;
    }

}
