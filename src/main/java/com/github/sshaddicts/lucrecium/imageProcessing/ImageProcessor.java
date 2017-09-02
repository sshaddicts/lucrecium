package com.github.sshaddicts.lucrecium.imageProcessing;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;


public class ImageProcessor {

    private Mat image;

    private List<Rect> chars;
    private List<Rect> lines;
    public static final int DEFAULT_REGION_PADDING = 1;

    public static final int MERGE_WORDS = -3;
    public static final int MERGE_LINES = -20;
    public static final int MERGE_CHARS = 1;

    private double resizeRate = 0.3;

    Logger log = LoggerFactory.getLogger(this.getClass());

    public ImageProcessor(String filename) {
        if (filename == null || Objects.equals(filename, "")) {
            throw new IllegalArgumentException("Filename cannot be null or empty. Requested filepath: " + filename);
        }

        this.image = Imgcodecs.imread(filename, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        if (image.height() == 0 || image.width() == 0)
            throw new IllegalArgumentException("Image has to be at least 1x1. current dimensions: height = "
                    + image.height() + ", width = " + image.width() + ".\n" +
                    "Requested filepath: " + filename + ", check it once again.");
        this.chars = new ArrayList<>();
    }

    private void threshold() {
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.adaptiveThreshold(image, tempMat, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY, 7, 5);
        image = tempMat;
    }

    private void blur() {
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.GaussianBlur(image, tempMat, new Size(1, 1), 0);
        image = tempMat;
    }

    private void resize() {
        if (image.height() == 0 || image.width() == 0) {
            if (image.height() < 500) {
                resizeRate *= 2.5;
            }
        }

        Mat tempMat = new Mat(image.height(), image.width(), image.type());

        double height = image.height() * resizeRate;
        double width = image.width() * resizeRate;

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

        for (Mat mat : countours) {
            if (Imgproc.contourArea(mat) > image.rows() * image.cols() / 4) {
                Rect rect = Imgproc.boundingRect((MatOfPoint) mat);
                if (rect.height > 50)
                    result = image.submat(rect);
            }
        }

        assert result.height() != 0;

        image = result;
    }

    public Mat getImage() {
        return image;
    }

    public List<Mat> getText(int mergeType) {
        process();
        detectText(mergeType);
        return getMats();
    }

    private List<Mat> getMats() {
        List<Mat> returnList = new ArrayList<>();

        Mat source;

        for (Rect rect : chars) {
            source = image.submat(rect);
            returnList.add(source);
        }

        return returnList;
    }

    private Mat deskew(Mat src, double angle) {
        Point center = new Point(src.width() / 2, src.height() / 2);
        Mat rotatedImage = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        Size size = new Size(src.width(), src.height());
        Imgproc.warpAffine(src, src, rotatedImage, size,
                Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);
        return src;
    }

    //TODO refactor
    public void process() {
        resize();

        Mat tmpMat = new Mat(image.height(), image.width(), image.type());

        Mat imageClone = image.clone();

        Mat mask = new Mat(image.height(), image.width(), image.type());
        Imgproc.threshold(imageClone, mask, 150, 255,
                Imgproc.THRESH_BINARY);

        Core.bitwise_and(image, mask, tmpMat);

        Imgproc.adaptiveThreshold(tmpMat, image, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 11, 25);

        Mat pointMat = Mat.zeros(tmpMat.size(), tmpMat.channels());
        Core.findNonZero(tmpMat, pointMat);

        MatOfPoint2f mat2f = new MatOfPoint2f();
        pointMat.convertTo(mat2f, CvType.CV_32FC2);

        RotatedRect rotated = Imgproc.minAreaRect(mat2f);

        if (rotated.size.width > rotated.size.height)
            rotated.angle += 90.f;

        image = deskew(image, rotated.angle);

        cropImage();
    }

    public void detectText(int mergeType) {
        Mat newImage = image.clone();

        Imgproc.cvtColor(image, newImage, Imgproc.COLOR_GRAY2RGB);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hiech = new Mat();

        Imgproc.findContours(image, contours, hiech,
                Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_NONE);

        contours.removeIf((mat) -> image.height() - mat.height() < 200);
        contours.removeIf((mat) -> mat.size().area() < 7);

        chars = mergeInnerRects(contours, mergeType);
        chars = mergeCloseRects(chars, mergeType);

        drawRects(image, chars);

        Mat temp = new Mat(image.size(), image.type());

        drawRects(temp, chars);
    }

    private List<Rect> mergeInnerRects(List<MatOfPoint> points, int mergeType) {
        List<Rect> result = new ArrayList<>();
        Mat mask = Mat.zeros(image.size(), image.type());

        for (MatOfPoint point : points) {
            Rect rect = Imgproc.boundingRect(point);
            rect.width -= mergeType;
            Imgproc.rectangle(mask, rect.tl(), rect.br(), new Scalar(255), -1);
        }

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(),
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            result.add(Imgproc.boundingRect(contour));
        }

        return result;
    }

    private List<Rect> mergeCloseRects(List<Rect> rects, int mergeType) {
        List<Rect> mergedRects = new ArrayList<>();

        for (int i = 0; i < rects.size(); i++) {
            Rect rect = rects.get(i);
            for (int j = i; j < rects.size(); j++) {
                Rect otherRect = rects.get(j);
                if (Math.abs((rect.y + rect.height / 2) - (otherRect.y + otherRect.height / 2)) < 1){
                    mergedRects.add(Validator.merge(rect, otherRect, mergeType));
                }
            }
        }

        return mergedRects;
    }

    private void drawRects(Mat image, List<Rect> rects) {
        for (Rect rect : rects) {
            Imgproc.rectangle(image, rect.tl(), rect.br(), new Scalar(255), 1);
        }
    }
}