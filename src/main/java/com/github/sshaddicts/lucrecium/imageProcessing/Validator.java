package com.github.sshaddicts.lucrecium.imageProcessing;

import org.opencv.core.Rect;


/**
 * Created by Alex on 01.08.2017.
 */
public class Validator {

    public static double MAX_AREA_THRESHOLD;
    public static double MIN_AREA_THRESHOLD;
    public static double ASPECT_RATIO;

    public static boolean isValidTextArea(Rect rect) {

        //check size
        boolean lessThanMaximum = rect.size().area() < MAX_AREA_THRESHOLD;
        boolean biggerThanMinimum = rect.size().area() > MIN_AREA_THRESHOLD;

        //check aspect ratio
        boolean aspectRatio = (rect.height / rect.width) < ASPECT_RATIO;

        return lessThanMaximum && biggerThanMinimum && aspectRatio;
    }

    public static boolean isOverlapping(Rect rect1, Rect rect2) {
        if(rect1 == rect2)
            return true;
        if (rect1.contains(rect2.tl()) || rect1.contains(rect2.br())) {
            return true;
        }
        return false;
    }
}
