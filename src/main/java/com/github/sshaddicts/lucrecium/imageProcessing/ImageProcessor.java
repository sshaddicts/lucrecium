package com.github.sshaddicts.lucrecium.imageProcessing;

import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;
import com.github.sshaddicts.lucrecium.util.RectManipulator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;


public class ImageProcessor {

    public static final int MERGE_WORDS = -5;
    public static final int MERGE_LINES = -20;
    public static final int NO_MERGE = 2;

    private static final int RESIZE_THRESHOLD = 1000;

    private double resizeRate = 0.3;
    private Logger log = LoggerFactory.getLogger(this.getClass());

    public static BufferedImage toBufferedImage(Mat image) {
        return Imshow.toBufferedImage(image);
    }

    public static byte[] toByteArray(Mat src) {
        byte[] byteData = new byte[src.height() * src.width() * src.channels()];
        src.get(0, 0, byteData);
        return byteData;
    }

    public static INDArray toNdarray(Mat mat) throws IOException {
        byte[] matData = new byte[mat.width() * mat.height()];
        double[] retData = new double[matData.length];

        mat.get(0, 0, matData);

        for (int i = 0; i < matData.length; i++) {
            retData[i] = (double) matData[i];
        }

        return Nd4j.create(retData, new int[]{mat.width(), mat.height()});
    }

    public static Mat loadImage(byte[] bytes) {
        return Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
    }

    public static Mat loadImage(String filename) {
        if (filename == null || Objects.equals(filename, "")) {
            throw new IllegalArgumentException("Filename cannot be null or empty. Requested filepath: " + filename);
        }

        Mat image = Imgcodecs.imread(filename, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);

        if (image.height() == 0 || image.width() == 0) {
            throw new IllegalArgumentException("Image has to be at least 1x1. current dimensions: height = "
                    + image.height() + ", width = " + image.width() + ".\n" +
                    "Requested filepath: " + filename + ", check it once again.");
        }

        return image;
    }

    private Mat thresholdImage(Mat src) {
        Mat mask = new Mat(src.height(), src.width(), src.type());
        Imgproc.threshold(src, mask, 20, 255,
                Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        return mask;
    }

    private Mat resize(Mat image) {
        if (image.height() == 0 || image.width() == 0) {
            throw new IllegalArgumentException("Image size is illegal" + image.size().toString());
        }

        if (image.height() < RESIZE_THRESHOLD || image.width() < RESIZE_THRESHOLD) {
            resizeRate = 1;
        }

        Mat tempMat = new Mat(image.height(), image.width(), image.type());

        double height = image.height() * resizeRate;
        double width = image.width() * resizeRate;

        Imgproc.resize(image, tempMat, new Size(width, height), 2, 2, Imgproc.INTER_AREA);
        return tempMat;
    }

    private Mat cropImage(Mat image) {
        Mat tmpImage = thresholdImage(image);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(tmpImage, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat result = new Mat();

        for (Mat mat : contours) {
            if (Imgproc.contourArea(mat) > tmpImage.rows() * tmpImage.cols() / 4) {
                Rect rect = Imgproc.boundingRect((MatOfPoint) mat);
                if (rect.height > 50 && (rect.height != image.height() || rect.width != image.width())) {
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

    public SearchResult findTextRegions(String filename) {
        return findTextRegions(filename, ImageProcessor.NO_MERGE, false);
    }

    public SearchResult findTextRegions(String filename, int mergeType, boolean isRotationNeeded) {
        return findTextRegions(ImageProcessor.loadImage(filename), mergeType, isRotationNeeded);
    }

    public SearchResult findTextRegions(Mat image) {
        return findTextRegions(image, ImageProcessor.NO_MERGE, false);
    }

    public SearchResult findTextRegions(Mat image, int mergeType, boolean isRotationNeeded) {
        List<Rect> chars = detectText(image, mergeType, isRotationNeeded);

        return constructCharRegions(image, chars);
    }

    public SearchResult findTextRegions(byte[] bytes) {
        return findTextRegions(bytes, ImageProcessor.NO_MERGE, false);
    }

    public SearchResult findTextRegions(byte[] bytes, int mergeType, boolean isRotationNeeded) {
        return findTextRegions(ImageProcessor.loadImage(bytes), mergeType, isRotationNeeded);
    }

    public SearchResult constructCharRegions(Mat image, List<Rect> chars) {
        ArrayList<CharContainer> charsList = new ArrayList<>(chars.size());

        for (Rect rect : chars) {
            charsList.add(new CharContainer(image.submat(rect), rect));
        }

        Collections.sort(charsList);

        return new SearchResult(charsList, makeOverlay(image, chars));
    }

    private void deskew(Mat src, double angle) {
        Point center = new Point(src.width() / 2, src.height() / 2);
        Mat rotatedImage = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        Size size = new Size(src.width(), src.height());
        Imgproc.warpAffine(src, src, rotatedImage, size,
                Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);
    }

    private double deskew(Mat src) {
        Mat tmpMat = src.clone();

        Imgproc.adaptiveThreshold(
                tmpMat,
                tmpMat,
                255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV,
                21,
                22
        );

        Mat pointMat = Mat.zeros(tmpMat.size(), tmpMat.channels());
        Core.findNonZero(tmpMat, pointMat);

        MatOfPoint2f mat2f = new MatOfPoint2f();
        pointMat.convertTo(mat2f, CvType.CV_32FC2);

        RotatedRect rotated = Imgproc.minAreaRect(mat2f);

        if (rotated.size.width > rotated.size.height)
            rotated.angle += 90.f;

        log.debug("Rotation angle: " + rotated.angle + " degrees.");

        deskew(src, rotated.angle);

        return rotated.angle;
    }

    private Mat process(Mat image, boolean isRotationNeeded) {
        if (isRotationNeeded) {
            deskew(image);
        }

        image = resize(image);
        image = adjustContrast(image, 5, -780);
        image = thresholdImage(image);

        image = cropImage(image);
        log.debug("Image size is " + image.size());

        return image;
    }

    private List<Rect> detectText(Mat image, int mergeType, boolean isRotationNeeded) {

        Mat processed = process(image, isRotationNeeded);

        List<MatOfPoint> contours = new LinkedList<>();
        Mat hiech = new Mat();

        Imgproc.findContours(
                processed,
                contours,
                hiech,
                Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_NONE
        );

        if (processed.height() > 500) {
            for (Iterator<MatOfPoint> it = contours.iterator(); it.hasNext(); ) {
                MatOfPoint cont = it.next();

                if (processed.height() - cont.height() < 200) {
                    it.remove();
                }
            }
        }

        for (Iterator<MatOfPoint> it = contours.iterator(); it.hasNext(); ) {
            MatOfPoint cont = it.next();

            if (cont.size().area() < 7) {
                it.remove();
            }
        }

        List<Rect> chars = mergeInnerRects(image, contours, mergeType);

        for (Iterator<Rect> rectIterator = chars.iterator(); rectIterator.hasNext(); ) {
            Rect current = rectIterator.next();
            if (current.width < 5 || current.height < 5) {
                rectIterator.remove();
            }
        }

        Collections.reverse(chars);

        log.debug("Contours size: " + contours.size());

        int meanHeight = calculateMean(chars, false);

        return splitForThreshold(chars, meanHeight, false);
    }

    private List<Rect> mergeInnerRects(Mat image, List<MatOfPoint> points, int mergeType) {
        List<Rect> result = new ArrayList<>();
        Mat mask = Mat.zeros(image.size(), image.type());

        for (MatOfPoint point : points) {
            Rect rect = Imgproc.boundingRect(point);
            rect.width -= mergeType;
            Imgproc.rectangle(mask, rect.tl(), rect.br(), new Scalar(255), -10);
        }

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(
                mask, contours,
                new Mat(),
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        for (MatOfPoint contour : contours) {
            result.add(Imgproc.boundingRect(contour));
        }

        log.debug("ROI number after merging inner rects: " + result.size());

        return result;
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

    private int calculateMean(List<Rect> rects, boolean horizontal) {
        int total = 0;

        for (Rect rect : rects) {
            total += horizontal ? rect.width : rect.height;
        }

        return total / rects.size();
    }

    private List<Rect> splitForThreshold(List<Rect> rects, int mean, boolean horizontal) {
        List<Rect> result = new ArrayList<>(rects.size());

        for (Rect rect : rects) {
            if (rect.height > mean * 2) {
                List<Rect> split = RectManipulator.split(rect, (int) Math.floor(rect.height / mean), horizontal);
                result.addAll(split);
            } else {
                result.add(rect);
            }
        }

        return result;
    }

    private Mat makeOverlay(Mat image, List<Rect> chars) {
        Mat imageClone = Mat.zeros(image.size(), 16);
        Imgproc.cvtColor(image, imageClone, Imgproc.COLOR_GRAY2RGB);
        log.debug("debug image type is " + imageClone.type());

        drawRects(imageClone, chars, new Scalar(0, 128, 255));

        return imageClone;
    }

    public void drawRects(Mat image, List<Rect> rects, Scalar color) {
        for (Rect rect : rects) {
            Imgproc.rectangle(image, rect.tl(), rect.br(), color, 1);
        }
    }

    public void drawContours(Mat image, List<MatOfPoint> contours, Scalar color) {
        Imgproc.drawContours(image, contours, -1, color, -1);
    }
}