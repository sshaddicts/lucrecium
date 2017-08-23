package com.github.sshaddicts.lucrecium.imageProcessing;

import org.bytedeco.javacpp.FlyCapture2;
import org.opencv.core.*;
import org.opencv.features2d.MSER;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/**
 * Created by Alex on 29.07.2017.
 */
//TODO extend rect class with validation methods
//TODO add slant correction
public class ImageProcessor {

    private Mat image;
    private List<MatOfPoint> regions;
    private MatOfRect rect;

    private List<Rect> rois;
    private static int DEFAULT_REGION_PADDING = 1;

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

    int roisize = 0;

    public ImageProcessor(String filename) {
        if (filename == null || Objects.equals(filename, "")) {
            throw new IllegalArgumentException("Filename cannot be null or empty.");
        }

        this.image = Imgcodecs.imread(filename, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        this.regions = new ArrayList<>();
        this.rect = new MatOfRect();
        this.rois = new ArrayList<>();
    }

    private void threshold() {
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.adaptiveThreshold(image, tempMat, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 7, 5);
        image = tempMat;
    }

    private void blur() {
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.blur(image, tempMat, new Size(8, 8));
        image = tempMat;
    }

    private void resize() {
        Mat tempMat = new Mat();

        double height, width;
        height = image.height() * resizeRate;
        width = image.width() * resizeRate;
        Imgproc.resize(image, tempMat, new Size(width, height));
        image = tempMat;
    }

    private void cropImage() {
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

        image = result;
    }

    public void preProcess(){
        blur();
        cropImage();
        resize();
        threshold();
    }

    private void detectMSER() {
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

    private void drawROI() {
        detectMSER();
        List<Rect> rects = rect.toList();

        for (int i = 0; i < rects.size(); i++) {

            Rect rect = rects.get(i);

            if (Validator.isValidTextArea(rect)) {
                rect = fixRect(rect, DEFAULT_REGION_PADDING);
                //Imgproc.rectangle(image, rect.tl(), rect.br(), new Scalar(92));

                rois.add(rect);
                roisize++;
            }
        }
    }

    public List<MatOfPoint> getRegions() {
        return regions;
    }

    public Mat getImage() {
        return image;
    }

    public List<Mat> getText(){
        preProcess();
        drawROI();
        return getMats();
    }

    //TODO add built-in resizing
    public List<Mat> getMats() {
        List<Mat> returnList = new ArrayList<>();

        Mat source;

        Size outputSize = new Size(9, 18);

        for (int i = 0; i < rois.size(); i++) {
            Rect rect = rois.get(i);

            Mat temp = new Mat();
            source = image.submat(rect);

            Imgproc.resize(source, temp, outputSize);

            returnList.add(temp);
        }

        return returnList;
    }

    private Rect fixRect(Rect rect, int padding) {
        padding = padding == 0 ? 10 : padding;

        int lowX = rect.x - padding;
        int lowY = rect.y - padding;

        int highX = rect.width + (padding * 2);
        int highY = rect.height + (padding * 2);

        return new Rect(lowX <= 0 ? 0 : lowX, lowY <= 0 ? 0 : lowY,
                rect.x + highX > image.cols() -2 ? rect.width: highX,
                rect.y + highY > image.rows() -2 ? rect.height: highY);

    }

    public boolean save(String filename) {
        return Imgcodecs.imwrite(filename, image);
    }
}