package com.github.sshaddicts.lucrecium.imageProcessing;

import org.opencv.core.Rect;


/**
 * Created by Alex on 01.08.2017.
 */
//TODO: remove this one
public class Validator {

    public static final double MAX_AREA_THRESHOLD = 1000;
    public static final double ASPECT_RATIO = 2;

    static boolean isValidCharArea(Rect rect) {
        if (rect.height < 15 || rect.width == 0)
            return false;

        //check size
        double area = rect.size().area();
        boolean isValidArea = area < MAX_AREA_THRESHOLD;

        //check aspect ratio
        double realRatio = rect.height / rect.width;
        boolean isValidAspectRatio = realRatio < ASPECT_RATIO;

        return isValidArea && isValidAspectRatio;
    }

    static Rect merge(Rect rect1, Rect rect2, int mergeType) {

        return new Rect(Integer.min(rect1.x, rect2.x),
                Integer.max(rect1.y, rect2.y),

                Integer.max(rect1.width, rect2.width) + mergeType,
                Integer.max(rect1.height, rect2.height));
    }
}
