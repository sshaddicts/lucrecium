package com.github.sshaddicts.lucrecium.imageProcessing;

import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;
import org.opencv.core.Mat;

import java.util.List;

public class SearchResult {
    private List<CharContainer> chars;
    private Mat overlay;

    public SearchResult(List<CharContainer> chars, Mat overlay) {
        this.chars = chars;
        this.overlay = overlay;
    }

    public Mat getOverlay() {
        return overlay;
    }

    public List<CharContainer> getChars() {
        return chars;
    }
}
