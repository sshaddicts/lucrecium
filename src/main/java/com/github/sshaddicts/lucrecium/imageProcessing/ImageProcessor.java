package com.github.sshaddicts.lucrecium.imageProcessing;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;


public class ImageProcessor {

    private Mat image;
    private MatOfRect regions;

    private List<Rect> chars;
    private List<Rect> lines;
    public static final int DEFAULT_REGION_PADDING = 1;

    public static final int MERGE_WORDS = -3;
    public static final int MERGE_LINES = -20;
    public static final int MERGE_CHARS = 1;

    public double resizeRate = 0.3;


    public ImageProcessor(String filename) {
        if (filename == null || Objects.equals(filename, "")) {
            throw new IllegalArgumentException("Filename cannot be null or empty. Requested filepath: " + filename);
        }

        this.image = Imgcodecs.imread(filename, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        if (image.height() == 0 || image.width() == 0)
            throw new IllegalArgumentException("Image has to be at least 1x1. current dimensions: height = " + image.height() + ", width = " + image.width() + ".\n" +
                    "Requested filepath: " + filename + ", check it once again.");
        this.regions = new MatOfRect();
        this.chars = new ArrayList<>();
    }

    private void threshold() {
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.adaptiveThreshold(image, tempMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 7, 5);
        image = tempMat;
    }

    private void blur() {
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.GaussianBlur(image, tempMat, new Size(3, 3), 0);
        image = tempMat;
    }

    private void resize() {
        if (image.height() == 0 || image.width() == 0)

            if (image.height() < 500) {
                resizeRate *= 2.5;
            }
        Mat tempMat = new Mat(image.height(), image.width(), image.type());

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

        if (result.height() == 0) {
            return;
        }

        image = result;
    }

    private void preProcess() {
        blur();
        cropImage();
        resize();
        threshold();
    }

    private void drawRoi() {
        List<Rect> rects = regions.toList();

        for (Rect rect : rects) {
            if (Validator.isValidCharArea(rect)) {
                rect = fixRect(rect, DEFAULT_REGION_PADDING);
                chars.add(rect);
            }
        }
    }

    public Mat getImage() {
        return image;
    }

    public List<Mat> getText() {
        preProcess();
        drawRoi();
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

    private Rect fixRect(Rect rect, int padding) {
        padding = padding == 0 ? 10 : padding;

        int lowX = rect.x - padding;
        int lowY = rect.y - padding;

        int highX = rect.width + (padding * 2);
        int highY = rect.height + (padding * 2);

        return new Rect(lowX <= 0 ? 0 : lowX, lowY <= 0 ? 0 : lowY,
                rect.x + highX > image.cols() - 2 ? rect.width : highX,
                rect.y + highY > image.rows() - 2 ? rect.height : highY);

    }

    public boolean save(String filename) {
        return Imgcodecs.imwrite(filename, image);
    }

    private Mat deskew(Mat src, double angle) {
        Point center = new Point(src.width() / 2, src.height() / 2);
        Mat rotatedImage = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        Size size = new Size(src.width(), src.height());
        Imgproc.warpAffine(src, src, rotatedImage, size, Imgproc.INTER_LINEAR
                + Imgproc.CV_WARP_FILL_OUTLIERS);
        return src;
    }

    //TODO refactor
    public void computeSkewAndProcess() {
        resize();

        Mat tmpMat = new Mat(image.height(), image.width(), image.type());

        //create a mask for future use:
        Mat imageClone = image.clone();

        Mat mask = new Mat(image.height(), image.width(), image.type());
        Imgproc.threshold(imageClone, mask, 100, 255, Imgproc.THRESH_BINARY);

        Core.bitwise_and(image, mask, tmpMat);

        //Imgproc.GaussianBlur(image, tmpMat, new Size(1,1), 0);
        Imgproc.adaptiveThreshold(tmpMat, image, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 11, 12);

        //get minimal rotated rect
        Mat pointMat = Mat.zeros(tmpMat.size(), tmpMat.channels());
        Core.findNonZero(tmpMat, pointMat);

        MatOfPoint2f mat2f = new MatOfPoint2f();
        pointMat.convertTo(mat2f, CvType.CV_32FC2);

        //get angle
        RotatedRect rotated = Imgproc.minAreaRect(mat2f);

        if (rotated.size.width > rotated.size.height)
            rotated.angle += 90.f;

        System.out.println(rotated.angle);

        image = deskew(image, rotated.angle);

        cropImage();
    }

    public void detectText(int mergeType) {
        Mat newImage = image.clone();

        Imgproc.cvtColor(image, newImage, Imgproc.COLOR_GRAY2RGB);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hiech = new Mat();

        Imgproc.findContours(image, contours, hiech, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        contours.removeIf((mat) -> image.height() - mat.height() < 200);
        contours.removeIf((mat) -> mat.size().area() < 7);

        chars = mergeInnerRects(contours, mergeType);

        chars = mergeCloseRects(chars);

        Mat temp = new Mat(image.size(), image.type());

        System.out.println(image.type());
        Core.invert(image, temp);

        drawRects(temp, chars);
        Imshow.show(temp);
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
        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour :
                contours) {
            result.add(Imgproc.boundingRect(contour));
        }

        return result;
    }

    private List<Rect> mergeCloseRects(List<Rect> rects) {
        List<Rect> mergedRects = new ArrayList<>();

        for (int i = 0; i < rects.size(); i++) {
            Rect rect = rects.get(i);
            for (int j = i; j < rects.size(); j++) {
                Rect otherRect = rects.get(j);
                if (Math.abs((rect.y + rect.height / 2) - (otherRect.y + otherRect.height / 2)) < 1)
                    mergedRects.add(Validator.merge(rect, otherRect));
            }
        }

        return mergedRects;
    }


    private void drawRects(Mat image, List<Rect> rects) {
        for (Rect rect :
                rects) {
            Imgproc.rectangle(image, rect.tl(), rect.br(), new Scalar(255), 1);
        }
    }

    public Mat experimentalProcessing() {

        resize();

        Mat tmp = new Mat();
        blur();

        Imgproc.adaptiveThreshold(image, tmp, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 111, 3);

        Mat kernel = Mat.ones(new Size(5, 5), CvType.CV_8UC1);

        Mat opening = new Mat();
        Mat closing = new Mat();

        Imgproc.morphologyEx(tmp, opening, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(opening, closing, Imgproc.MORPH_CLOSE, kernel);

        return closing;
    }
}