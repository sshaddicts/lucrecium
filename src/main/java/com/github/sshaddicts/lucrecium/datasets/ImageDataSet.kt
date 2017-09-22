package com.github.sshaddicts.lucrecium.datasets

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import java.io.File
import java.io.IOException
import java.util.*

class ImageDataSet(private val outputLabelCount: Int, private val batchSize: Int) {

    var iterator: DataSetIterator? = null
        private set

    var recordReader: ImageRecordReader? = null
        private set

    @Throws(IOException::class)
    fun initFromDirectory(parentDir: String) {
        val parentDirectory = File(parentDir)

        val allowedExtensions = arrayOf(".png", ".jpg")
        val fileSplit = FileSplit(parentDirectory, allowedExtensions, Random())

        val labelMarker = ParentPathLabelGenerator()

        this.recordReader = ImageRecordReader(32, 32, 1, labelMarker)

        this.iterator = RecordReaderDataSetIterator(recordReader, batchSize, 1, outputLabelCount)
        val scaler = ImagePreProcessingScaler(0.0, 1.0)
        scaler.fit(this.iterator)
        this.iterator!!.preProcessor = scaler
        recordReader!!.initialize(fileSplit)
    }
}
