package com.github.sshaddicts.lucrecium.imageProcessing;

import org.opencv.core.*;
import org.opencv.features2d.MSER;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Iterator;
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

    public int delta = 5;
    public int minArea = 20;
    public int maxArea = Integer.MAX_VALUE;
    public int maxVariation = 5;
    public int minDiversity = 5;
    public int maxEvolution = 5;
    public int areaThreshold = 5;
    public int minMargin = 0;
    public int edgeBlurSize = 0;

    public double resizeRate = 0.3;

    public ImageProcessor(String filename) {
        if (filename == null || Objects.equals(filename, "")) {
            throw new IllegalArgumentException("Filename cannot be null or empty.");
        }

        this.image = Imgcodecs.imread(filename, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        this.regions = new ArrayList<>();
        this.rect = new MatOfRect();
        this.rois = new ArrayList<>();
    }

    public void drawROI() {
        detectMSER();
        List<Rect> rects = rect.toList();


        for (Iterator<Rect> iterator = rects.listIterator(); iterator.hasNext(); ) {
            Rect current = iterator.next();

            current = fixRect(current, 5);
            if (Validator.isValidTextArea(current)) {

                Imgproc.rectangle(image, current.tl(), current.br(), new Scalar(92));

                rois.add(current);

            }
        }
    }

    public Mat threshold(){
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.adaptiveThreshold(image, tempMat, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 7,5);
        return tempMat;
    }

    public Mat detectAndRemoveBoundary(){

        List<MatOfPoint> countour = new ArrayList<>();
        Mat hierarchy = new Mat();

        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());

        Imgproc.findContours(image, countour, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE, new Point(5,5));

        Imgproc.drawContours(tempMat, countour, 2,new Scalar(255),2, Imgproc.LINE_8, hierarchy, 255, new Point(5,5));

        return tempMat;

    }

    public Mat blur(){
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.blur(image, tempMat, new Size(8,8));
        return tempMat;
    }

    public void resize() {
        Mat tempMat = new Mat();

        double height, width;
        height = image.height() * resizeRate;
        width = image.width() * resizeRate;
        Imgproc.resize(image, tempMat, new Size(width, height));
        image = tempMat;
    }

    public Mat cropImage() {
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
            if (Imgproc.contourArea(mat) > image.rows() * image.cols() / 4) {
                Rect rect = Imgproc.boundingRect((MatOfPoint) mat);
                if (rect.height > 50)
                    result = image.submat(rect);
            }
        }

        return image = result;
    }

    public void preProcess(){
        image = blur();
        cropImage();
        resize();
        image = threshold();

    }

    public void detectMSER() {
        MSER mser = MSER.create(delta,
                minArea,
                maxArea,
                maxVariation,
                minDiversity,
                maxEvolution,
                areaThreshold,
                minMargin,
                edgeBlurSize);
        mser.detectRegions(image, regions, rect);
    }


    public List<MatOfPoint> getRegions() {
        return regions;
    }

    public Mat getImage() {
        return image;
    }

    public List<Mat> getMats() {
        List<Mat> returnList = new ArrayList<>();

        for (Rect rectang :
                rois) {
            returnList.add(image.submat(rectang));
        }

        return returnList;
    }

    private Rect fixRect(Rect rect, int padding) {
        padding=padding == 0?10:padding;
        Rect tmpRect = new Rect(rect.x-padding, rect.y-padding, rect.width+padding*2, rect.height+padding*2);

        return tmpRect;
    }

}
