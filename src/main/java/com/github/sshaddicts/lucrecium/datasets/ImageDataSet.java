package com.github.sshaddicts.lucrecium.datasets;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class ImageDataSet {

    private FileSplit fileSplit;
    private BalancedPathFilter pathFilter;
    private DataSetIterator iterator;

    private InputSplit training;
    private InputSplit testing;

    private final int outputLabelCount;
    private final int batchSize;

    Logger log = LoggerFactory.getLogger(this.getClass());

    public ImageDataSet(int outputLabelCount, int batchSize) {
        this.outputLabelCount = outputLabelCount;
        this.batchSize = batchSize;
    }

    public void initFromDirectory(String parentDir) throws IOException {
        File parentDirectory = new File(parentDir);

        String[] allowedExtensions = new String[]{".png", ".jpg"};
        this.fileSplit = new FileSplit(parentDirectory, allowedExtensions, new Random());

        ParentPathLabelGenerator labelMarker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(18, 9, 1, labelMarker);
        this.pathFilter = new BalancedPathFilter(new Random(), allowedExtensions, labelMarker);

        this.iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputLabelCount);

        recordReader.initialize(fileSplit);
    }

    public void addPreProcessor(DataSetPreProcessor preProcessor) {
        this.iterator.setPreProcessor(preProcessor);
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
