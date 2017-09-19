package com.github.sshaddicts.lucrecium.imageProcessing;

import com.github.sshaddicts.lucrecium.imageProcessing.containers.CharContainer;

import java.util.List;

public class SearchResult {
    private List<CharContainer> chars;
    private byte[] overlay;

    public SearchResult(List<CharContainer> chars, byte[] overlay) {
        this.chars = chars;
        this.overlay = overlay;
    }

    public byte[] getOverlay() {
        return overlay;
    }

    public List<CharContainer> getChars() {
        return chars;
    }
}
