package com.github.sshaddicts.lucrecium.imageProcessing.containers

import org.opencv.core.Mat
import org.opencv.core.Rect

class CharContainer(val mat: Mat, val rect: Rect) : Comparable<CharContainer> {

    override operator fun compareTo(other: CharContainer): Int {
        val comparable = other.rect

        return if (Math.abs(comparable.y - rect.y) > 5) {
            rect.y - comparable.y
        } else {
            rect.x - comparable.x
        }
    }
}
