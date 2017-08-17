package com.github.sshaddicts.lucrecium.imageProcessing;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 29.07.2017.
 */
public class ImageDrawer{

    List<BufferedImage> images;

    public ImageDrawer(){
        this.images = new ArrayList<>();
    }

    public ImageDrawer(String filename){
        try {
            this.images = new ArrayList<>();
            images.add(ImageIO.read(new File(filename)));
        } catch (IOException e) {
            System.out.println(filename + " does not exist.");
            e.printStackTrace();
        }
    }

    public static void drawImage(Image img){
        ImageIcon icon = new ImageIcon(img);
        JLabel label = new JLabel(icon);
        JOptionPane.showMessageDialog(null, label);
    }

    public void addImage(BufferedImage image){
        images.add(image);
    }

    public void drawImages(){
        for (BufferedImage image :
                images) {
            drawImage(image);
        }
    }
}
