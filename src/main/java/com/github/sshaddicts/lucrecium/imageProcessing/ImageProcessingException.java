package com.github.sshaddicts.lucrecium.imageProcessing;

public class ImageProcessingException extends Exception {

    private String message;

    public ImageProcessingException(String message){
        this.message = message;
    }

    public String getInfo(){
        return message;
    }
}
