package com.github.sshaddicts.lucrecium.util;

import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;

public class RectManipulator {

    public static final double MAX_AREA_THRESHOLD = 1000;
    public static final double ASPECT_RATIO = 2;


    //TODO refactor
    public static List<Rect> split(Rect rect, int slices, boolean horizontal) {
        List<Rect> result = new ArrayList<>();

        if (horizontal) {
            int singleSliceWidth = rect.width / slices;

            for (int i = 1; i < slices + 1; i++) {
                Rect slice = new Rect(new Point(rect.x + singleSliceWidth * (i - 1), rect.y), new Point(rect.x + singleSliceWidth * i, rect.y + rect.height));
                result.add(slice);
            }
        } else {
            int singleSliceHeight = rect.height / slices;

            for (int i = 1; i < slices + 1; i++) {
                Rect slice = new Rect(new Point(rect.x, rect.y + singleSliceHeight * (i - 1)), new Point(rect.x + rect.width, rect.y + singleSliceHeight * i));
                result.add(slice);
            }
        }

        return result;
    }

    public static Rect merge(Rect rect1, Rect rect2, int mergeType) {

        return new Rect(Integer.min(rect1.x, rect2.x),
                Integer.min(rect1.y, rect2.y),

                (rect1.width + rect2.width) + mergeType,
                Integer.max(rect1.height, rect2.height));
    }

    public static int[] getDistanceBetweenRectCenters(Rect rect1, Rect rect2) {

        int[] distance = new int[2];

        //vertical distance
        distance[0] = Math.abs((rect1.y + rect1.height / 2) - (rect2.y + rect2.height / 2));
        //horizontal distance
        distance[1] = Math.abs((rect1.x + rect1.width) / 2) - ((rect2.y + rect2.height) / 2);

        return distance;
    }

    public static int getDistanceBetweenBorders(Rect rect1, Rect rect2) {
        return Math.abs((rect1.x + rect1.width) - rect2.x);
    }

    public static int getVerticalDistance(Rect rect1, Rect rect2) {
        return Math.abs((rect1.y + rect1.height) - (rect2.y + rect2.height));
    }

    public static boolean contains(Rect rect1, Rect rect2) {
        if (rect1 == rect2)
            return true;

        if (rect1.contains(getCenterForRect(rect2))) {
            return true;
        }

        return false;
    }

    public static Point getCenterForRect(Rect rect) {
        return new Point((rect.tl().x + rect.br().x) / 2, (rect.tl().y + rect.br().y) / 2);
    }

    static boolean isValidCharArea(Rect rect) {
        if (rect.height < 15 || rect.width == 0) {
            return false;
        }

        double area = rect.size().area();
        boolean isValidArea = area < MAX_AREA_THRESHOLD;

        double realRatio = rect.height / rect.width;
        boolean isValidAspectRatio = realRatio < ASPECT_RATIO;

        return isValidArea && isValidAspectRatio;
    }
}