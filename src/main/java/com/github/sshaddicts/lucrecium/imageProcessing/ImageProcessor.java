package com.github.sshaddicts.lucrecium.imageProcessing;

import com.github.sshaddicts.lucrecium.util.RectManipulator;
import com.google.common.collect.Lists;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

//TODO add stream constructor
public class ImageProcessor {

    private Mat image;

    private List<Rect> chars;
    private List<Rect> words;
    private List<WordContainer> wordCharAssoc;

    public static final int MERGE_WORDS = -5;
    public static final int MERGE_LINES = -20;
    public static final int NO_MERGE = 2;

    private double resizeRate = 0.3;
    private boolean isRotationNeeded = false;

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
        this.words = new ArrayList<>();
        this.wordCharAssoc = new ArrayList<>();
    }

    public void needsRotation(boolean needsRotation) {
        this.isRotationNeeded = needsRotation;
    }

    public Mat getImage() {
        return image;
    }

    private Mat thresholdImage(Mat src) {
        Mat mask = new Mat(src.height(), src.width(), src.type());
        Imgproc.threshold(src, mask, 20, 255,
                Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

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

        Imgproc.resize(image, tempMat, new Size(width, height), 2, 2, Imgproc.INTER_AREA);
        image = tempMat;
    }

    private Mat cropImage() {
        Mat tmpImage = thresholdImage(image);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(tmpImage, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat result = new Mat();

        for (Mat mat : contours) {
            if (Imgproc.contourArea(mat) > tmpImage.rows() * tmpImage.cols() / 4) {
                Rect rect = Imgproc.boundingRect((MatOfPoint) mat);
                if (rect.height > 50) {
                    rect.x -= 1;
                    rect.y -= 1;
                    rect.width += 1;
                    rect.height += 1;
                    result = image.submat(rect);
                }
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

    public List<WordContainer> getTextRegions(int mergeType) {
        detectText(mergeType);

        getWordRects();

        return wordCharAssoc;
    }

    public WordContainer getCharRegions(){
        List<Mat> result = new ArrayList<>();

        for (int i = 0; i < chars.size(); i++) {
            result.add(image.submat(chars.get(i)));
        }
        return new WordContainer(result, chars);
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

        Mat tmpMat = src.clone();
        Imgproc.adaptiveThreshold(tmpMat, tmpMat, 255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV,
                21, 22);

        Mat pointMat = Mat.zeros(tmpMat.size(), tmpMat.channels());
        Core.findNonZero(tmpMat, pointMat);

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

        if (isRotationNeeded) {
            deskew(image);
        }

        resize();

        image = adjustContrast(image, 5, -780);

        image = thresholdImage(image);

        image = cropImage();
        log.debug("Image size is " + image.size());
    }

    private void detectText(int mergeType) {
        process();

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hiech = new Mat();

        Imgproc.findContours(image, contours, hiech,
                Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_NONE);

        contours.removeIf((mat) -> image.height() - mat.height() < 200);
        contours.removeIf((mat) -> mat.size().area() < 7);

        chars = mergeInnerRects(contours, mergeType);
        chars.removeIf(rect -> rect.width < 4 || rect.height < 4);


        chars = Lists.reverse(chars);

        int meanHeight = calculateMean(chars, false);
        chars = splitForThreshold(chars, meanHeight, false);

        words = mergeCloseRects(chars);


        Mat imageClone = new Mat(image.size(), image.type());
        Imgproc.cvtColor(image, imageClone, Imgproc.COLOR_GRAY2RGB);

        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(imageClone, contours, i,
                    new Scalar(ThreadLocalRandom.current().nextInt(0,255),
                            ThreadLocalRandom.current().nextInt(0,255),
                            ThreadLocalRandom.current().nextInt(0,255))
                    ,-1);
        }

        drawRects(imageClone, chars, new Scalar(255, 128, 0));
        //drawRects(imageClone, words, new Scalar(0, 128, 255));

        Imshow.show(imageClone, "processing");
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

        log.debug("ROI number after merging inner rects: " + result.size());

        return result;
    }

    private List<Rect> mergeCloseRects(List<Rect> rects) {
        List<Rect> mergedRects = new ArrayList<>();

        int horizontalDistance, verticalDistance;
        int mergedNumber;

        for (int i = 0; i < rects.size(); i += mergedNumber) {
            //line is over when we suddenly drop on x
            mergedNumber = 1;
            Rect resultingRect = rects.get(i);
            int prevx = resultingRect.x;

            for (int j = i + 1; j < rects.size(); j++) {
                if (prevx < rects.get(j).x) {
                    break;
                } else {
                    prevx = rects.get(j).x;
                }

                Rect toMerge = rects.get(j);
                horizontalDistance = RectManipulator.getDistanceBetweenBorders(resultingRect, toMerge);

                if (horizontalDistance < 30) {
                    resultingRect = RectManipulator.merge(resultingRect, toMerge, -2);
                    mergedNumber++;
                }
            }
            mergedRects.add(resultingRect);
        }

        log.debug("Word merging done.");

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

    public List<WordContainer> getWordRects() {

        if (words.size() == 0) {
            throw new IllegalStateException("Word size is equal to 0, but should never be. " +
                    "It is probably a severe bug. Consider using https://github.com/sshaddicts/Tracker/issues " +
                    "to open an issue");
        }

        log.debug("Getting WordContainers... Expecting " + words.size() + " entries.");

        for (Rect word : words) {
            WordContainer container = new WordContainer(word);

            List<Rect> characters = new ArrayList<>();
            for (Rect character : chars) {
                if (RectManipulator.contains(word, character)) {
                    characters.add(character);
                }
            }

            characters.sort(Comparator.comparingInt(rect -> rect.x));
            container.setCharLocations(characters);

            List<Mat> mats = new ArrayList<>();
            for (Rect rect : characters) {
                mats.add(image.submat(rect));
            }

            container.setChars(mats);
            wordCharAssoc.add(container);
        }

        return wordCharAssoc;
    }

    private int calculateMean(List<Rect> rects, boolean horizontal) {
        int total = 0;

        for (Rect rect : rects) {
            total += horizontal ? rect.width : rect.height;
        }

        return total / rects.size();
    }

    private List<Rect> splitForThreshold(List<Rect> rects, int mean, boolean horizontal) {
        List<Rect> result = new ArrayList<>();

        for (int i = 0; i < rects.size(); i++) {
            Rect rect = rects.get(i);
            if (rect.height > mean * 2) {
                List<Rect> split = RectManipulator.split(rect, (int) Math.floor(rect.height / mean), horizontal);
                result.addAll(split);
            } else {
                result.add(rect);
            }
        }

        return result;
    }
}