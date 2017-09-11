package com.github.sshaddicts.lucrecium.imageProcessing.containers;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

public class CharContainer {
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
}
