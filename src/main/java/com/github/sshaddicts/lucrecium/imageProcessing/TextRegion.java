package com.github.sshaddicts.lucrecium.imageProcessing;

/**
 * Created by Alex on 31.07.2017.
 */
public class TextRegion {

    public double minX;
    public double minY;
    public double maxX;
    public double maxY;

    public int threshold;
    public static int REGION_OFFSET = 2;

    public TextRegion(double x, double y, int threshold){
        this.minX = this.maxX = x;
        this.minY = this.maxY = y;

        this.threshold = threshold;
    }

    public void add(double x, double y){
        minX = Math.min(minX, x) - TextRegion.REGION_OFFSET;
        minY = Math.min(minY, y) - TextRegion.REGION_OFFSET;

        maxX = Math.max(maxX, x) + TextRegion.REGION_OFFSET;
        maxY = Math.max(maxY, y) + TextRegion.REGION_OFFSET;
    }

    public boolean isNear(double x, double y){
        double centerX = (minX + maxX) /2;
        double centerY = (minY + maxY) /2;

        double distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));

        return distance < threshold;
    }

}
