package com.github.sshaddicts.lucrecium.datasets;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

//TODO refactor
public class ImageDataSet {

    private DataSetIterator iterator;

    private final int outputLabelCount;
    private final int batchSize;

    private ImageRecordReader recordReader;

    public ImageRecordReader getRecordReader() {
        return recordReader;
    }

    public ImageDataSet(int outputLabelCount, int batchSize) {
        this.outputLabelCount = outputLabelCount;
        this.batchSize = batchSize;
    }

    public void initFromDirectory(String parentDir) throws IOException {
        File parentDirectory = new File(parentDir);

        String[] allowedExtensions = new String[]{".png", ".jpg"};
        FileSplit fileSplit = new FileSplit(parentDirectory, allowedExtensions, new Random());

        ParentPathLabelGenerator labelMarker = new ParentPathLabelGenerator();

        this.recordReader = new ImageRecordReader(32, 32, 1, labelMarker);

        this.iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputLabelCount);
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(this.iterator);
        this.iterator.setPreProcessor(scaler);
        recordReader.initialize(fileSplit);
    }

    public DataSetIterator getIterator() {
        return iterator;
    }
}
