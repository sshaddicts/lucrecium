package com.github.sshaddicts.lucrecium.imageProcessing;

import com.github.sshaddicts.lucrecium.util.RectManipulator;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.List;

public class WordContainer {

    private Rect word;
    private List<Mat> chars;

    private List<Rect> charLocations;

    public WordContainer(Rect word){
        this.word = word;
    }
    public WordContainer(List<Mat> chars, List<Rect> charLocations){
        this.chars = chars;
        this.charLocations = charLocations;
    }

    public int getCharLength(){
        return chars.size();
    }

    public Rect getWordRect() {
        return word;
    }

    public List<Mat> getChars() {
        return chars;
    }

    public void setChars(List<Mat> chars) {
        this.chars = chars;
    }

    public List<Rect> getCharLocations() {
        return charLocations;
    }

    public void setCharLocations(List<Rect> charLocations) {
        this.charLocations = charLocations;
    }

}
