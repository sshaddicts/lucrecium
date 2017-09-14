package com.github.sshaddicts.lucrecium.imageProcessing.containers;

import org.jetbrains.annotations.NotNull;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

public class CharContainer implements Comparable{
    private final Mat mat;
    private final Rect rect;

    public Mat getMat() {
        return mat;
    }

    public Rect getRect() {
        return rect;
    }

    public CharContainer(Mat mat, Rect rect) {
        this.mat = mat;
        this.rect = rect;
    }

    @Override
    public int compareTo(@NotNull Object o) {
        Rect comparable = ((CharContainer) o).getRect();

        if(Math.abs(comparable.y - rect.y) > 5){
            return rect.y - comparable.y;
        }else{
            return rect.x - comparable.x;
        }


    }
}
