package com.github.sshaddicts.lucrecium.imageProcessing;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class ImageProcessor {

    private Mat image;

    private List<Rect> chars;

    public static final int MERGE_WORDS = -5;
    public static final int MERGE_LINES = -20;
    public static final int MERGE_CHARS = 2;

    private double resizeRate = 0.3;

    private Logger log = LoggerFactory.getLogger(this.getClass());

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

    private Mat thresholdImage() {
        Mat tempMat = new Mat(image.rows(), image.cols(), image.type());

        Mat mask = new Mat(image.height(), image.width(), image.type());
        Imgproc.threshold(image, tempMat, 20, 255,
                Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        return mask;
    }

    private void resize() {
        if (image.height() == 0 || image.width() == 0) {
            throw new IllegalArgumentException("Image size is illegal" + image.size().toString());
        }

        if (image.height() < 500) {
            resizeRate *= 2.5;
        }

        Mat tempMat = new Mat(image.height(), image.width(), image.type());

        double height = image.height() * resizeRate;
        double width = image.width() * resizeRate;

        Imgproc.resize(image, tempMat, new Size(width, height));
        image = tempMat;
    }

    private Mat cropImage() {
        Mat tmpImage = thresholdImage();

        List<MatOfPoint> countours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(tmpImage, countours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat result = new Mat();

        for (Mat mat : countours) {
            if (Imgproc.contourArea(mat) > tmpImage.rows() * tmpImage.cols() / 4) {
                Rect rect = Imgproc.boundingRect((MatOfPoint) mat);
                if (rect.height > 50)
                    result = image.submat(rect);
            }
        }

        if (result.height() != 0 && result.width() != 0) {
            return result;
        }
        log.warn("Not found a contour suitable for cropping the image. Consider brighter light, less blurred image or higher contrast levels");
        return image;
    }

    private Mat adjustContrast(Mat image, double alpha, double beta) {
        Mat result = Mat.zeros(image.size(), image.type());
        image.convertTo(result, -1, alpha, beta);

        return result;
    }

    public Mat getImage() {
        return image;
    }

    public List<Mat> getTextRegions(int mergeType) {
        detectText(mergeType);
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

    private double deskew(Mat src) {
        Mat pointMat = Mat.zeros(src.size(), src.channels());
        Core.findNonZero(src, pointMat);

        MatOfPoint2f mat2f = new MatOfPoint2f();
        pointMat.convertTo(mat2f, CvType.CV_32FC2);

        RotatedRect rotated = Imgproc.minAreaRect(mat2f);

        if (rotated.size.width > rotated.size.height)
            rotated.angle += 90.f;

        log.debug("Rotation angle: " + rotated.angle + " degrees.");

        src = deskew(src, rotated.angle);

        return rotated.angle;
    }

    private void process() {
        resize();

        image = adjustContrast(image, 5, -800);

        Imgproc.threshold(image, image, 20, 255,
                Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        deskew(image);

        image = cropImage();
    }

    private void detectText(int mergeType) {
        process();

        Mat imageClone = image.clone();
        Imgproc.cvtColor(image, imageClone, Imgproc.COLOR_GRAY2RGB);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hiech = new Mat();

        Imgproc.findContours(image, contours, hiech,
                Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_NONE);

        contours.removeIf((mat) -> image.height() - mat.height() < 200);
        contours.removeIf((mat) -> mat.size().area() < 7);

        chars = mergeInnerRects(contours, mergeType);
        chars = mergeCloseRects(chars, mergeType);

        chars.removeIf((rect) -> rect.height < 17);
    }

    private List<Rect> mergeInnerRects(List<MatOfPoint> points, int mergeType) {
        List<Rect> result = new ArrayList<>();
        Mat mask = Mat.zeros(image.size(), image.type());

        for (MatOfPoint point : points) {
            Rect rect = Imgproc.boundingRect(point);
            rect.width -= mergeType;
            Imgproc.rectangle(mask, rect.tl(), rect.br(), new Scalar(255), -10);
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
                if (Math.abs((rect.y + rect.height / 2) - (otherRect.y + otherRect.height / 2)) == 0) {
                    mergedRects.add(Validator.merge(rect, otherRect, mergeType));
                }
            }
        }

        return mergedRects;
    }

    public Mat submat(int x, int y, int width, int height) {
        Rect rect = new Rect(x, y, width, height);
        return image.submat(rect);
    }

    public int approximateLineNumberFor(Mat m) {
        int lineNumber = 0;

        Mat sums = calculateLineSums(m);

        int mean = (int) Core.mean(sums).val[0] / 3;

        boolean prev = false;
        for (int i = 0; i < sums.height(); i++) {
            int value = (int) sums.get(i, 0)[0];

            if (prev) {
                prev = value < mean;
            } else {
                prev = value < mean;
                lineNumber += prev ? 1 : 0;
            }
        }

        return lineNumber;
    }

    private Mat calculateLineSums(Mat m) {
        if (m.width() < 20) {
            throw new IllegalArgumentException("m is too small, and probably is not valid");
        }
        Mat result = new Mat(m.height(), 1, CvType.CV_32S);

        for (int i = 0; i < m.height(); i++) {
            result.put(i, 0, (int) (Core.sumElems(m.row(i)).val[0]));
        }
        return result;
    }

    private void drawRects(Mat image, List<Rect> rects, Scalar color) {
        for (Rect rect : rects) {
            Imgproc.rectangle(image, rect.tl(), rect.br(), color, 1);
        }
    }

    private void drawContours(Mat image, List<MatOfPoint> contours, Scalar color) {
        Imgproc.drawContours(image, contours, -1, color, -1);
    }

    public static BufferedImage toBufferedImage(Mat image) {
        return Imshow.toBufferedImage(image);
    }

    public static byte[] toByteArray(Mat image) {
        byte[] byteData = new byte[image.height() * image.width()];
        image.get(0, 0, byteData);
        return byteData;
    }

    public static INDArray toNdarray(Mat mat) {

        byte[] matData = new byte[mat.width() * mat.height()];
        double[] retData = new double[matData.length];

        mat.get(0, 0, matData);

        for (int i = 0; i < matData.length; i++) {
            retData[i] = (double) matData[i];
        }

        return Nd4j.create(retData, new int[]{mat.width(), mat.height()});
    }

    public static List<Rect> getBoundingBoxes(List<MatOfPoint> contours) {
        ArrayList<Rect> result = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            result.add(Imgproc.boundingRect(contour));
        }
        return result;
    }
}