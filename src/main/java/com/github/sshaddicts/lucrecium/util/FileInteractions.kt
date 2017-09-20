package com.github.sshaddicts.lucrecium.util

import org.apache.commons.io.FileUtils
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.slf4j.LoggerFactory
import java.io.File
import java.util.*


object FileInteractions {

    private val log = LoggerFactory.getLogger(FileInteractions::class.java)

    fun saveMats(list: List<Mat>, directory: String) {
        for (mat in list) {
            saveMat(mat, directory)
        }
    }

    fun saveMat(mat: Mat, directory: String) {
        val f = File(directory)

        if (!f.exists() && !f.mkdir()) {
            throw IllegalArgumentException("directory is hacked: " + directory)
        }

        Imgcodecs.imwrite(directory + "/image_" + System.nanoTime() + ".png", mat)
    }

    fun getFileNamesFromDir(parentDir: String): List<String> {
        val filenames = ArrayList<String>()

        val parentDirectory = FileUtils.getFile(parentDir)
        val files = parentDirectory.listFiles()!!

        for (file in files) {
            if (file.isFile) {
                filenames.add(parentDir + "/" + file.name)
            }
        }
        return filenames
    }

    fun saveMatWithName(mat: Mat, directory: String, filename: String) {
        Imgcodecs.imwrite(directory + "/" + directory + "_" + filename + ".png", mat)
    }
}
