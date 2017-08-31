package com.github.sshaddicts.lucrecium.datasets;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImageFlatteningDataSetPreProcessor;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class ImageDataSet {

    ImageRecordReader recordReader;
    FileSplit fileSplit;

    InputSplit training;
    InputSplit testing;

    BalancedPathFilter pathFilter;

    DataSetIterator iterator;

    private int batchSize = 20;
    private int numberOfClasses = 76;

    public ImageDataSet(String parentDir, int classNumber, int batchSize) {
        File parentDirectory = new File(parentDir);

        String[] allowedExtensions = new String[]{".png", ".jpg"};
        this.fileSplit = new FileSplit(parentDirectory, allowedExtensions, new Random());

        ParentPathLabelGenerator labelMarker = new ParentPathLabelGenerator();

        this.recordReader = new ImageRecordReader(18, 9, 1, labelMarker);
        this.pathFilter = new BalancedPathFilter(new Random(), allowedExtensions, labelMarker);

        this.numberOfClasses = classNumber;
        this.batchSize = batchSize;

        this.iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numberOfClasses);

        iterator.setPreProcessor(new ImageFlatteningDataSetPreProcessor());

        try {
            recordReader.initialize(fileSplit);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void splitData(int train, int test) {
        InputSplit[] data = fileSplit.sample(pathFilter, train, test);
        training = data[0];
        testing = data[1];
    }

    public DataSet next() {
        return iterator.next();
    }

    public boolean hasNext() {
        return iterator.hasNext();
    }

    public void resetIterator() {
        iterator.reset();
    }

    public DataSetIterator getIterator() {
        return iterator;
    }

}
