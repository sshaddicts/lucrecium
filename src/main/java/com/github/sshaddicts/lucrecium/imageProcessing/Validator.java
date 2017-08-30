package com.github.sshaddicts.lucrecium.imageProcessing;

import org.opencv.core.Rect;


/**
 * Created by Alex on 01.08.2017.
 */
public class Validator {

    public static double MAX_AREA_THRESHOLD = 1000;
    public static double MIN_AREA_THRESHOLD = 25;
    public static double ASPECT_RATIO = 2/1;
    public static double MIN_ASPECT_RATIO = 1;

    private static Rect RECT;
    private static int height;


    public static boolean isValidTextArea(Rect rect) {


        RECT = rect;
        height = rect.height;

        if(rect.height == 0 || rect.width == 0 )
            return false;


        //check if height is no bigger than the area

        //check size
        double area = rect.size().area();
        boolean isValidArea = area < MAX_AREA_THRESHOLD;

        //check aspect ratio
        double realRatio = rect.height / rect.width;
        boolean isValidAspectRatio = realRatio < ASPECT_RATIO;

        return isValidArea && isValidAspectRatio;
    }

    public static boolean isOverlapping(Rect rect1, Rect rect2) {
        if(rect1 == rect2)
            return true;
        if (rect1.contains(rect2.tl()) || rect1.contains(rect2.br())) {
            return true;
        }
        return false;
    }

    public static Rect merge(Rect rect1, Rect rect2){

        return new Rect(Integer.min(rect1.x, rect2.x),
                        Integer.max(rect1.y, rect2.y),

                        Integer.max(rect1.x + rect1.width, rect2.x + rect2.width),
                        //Integer.max(rect1.y, rect2.y ) + Integer.max(rect1.height,rect2.height));
                        rect1.height);
    }
}
