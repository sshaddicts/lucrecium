package com.github.sshaddicts.lucrecium.util

import org.opencv.core.Point
import org.opencv.core.Rect

/**
 * Splits rect horizontally or vertically into number of [slices]
 */
fun Rect.split(slices: Int, horizontal: Boolean) = if (horizontal) {
    val singleSliceWidth = this.width / slices

    (1 until slices + 1).map { Rect(Point((this.x + singleSliceWidth * (it - 1)).toDouble(), this.y.toDouble()), Point((this.x + singleSliceWidth * it).toDouble(), (this.y + this.height).toDouble())) }
} else {
    val singleSliceHeight = this.height / slices

    (1 until slices + 1).map { Rect(Point(this.x.toDouble(), (this.y + singleSliceHeight * (it - 1)).toDouble()), Point((this.x + this.width).toDouble(), (this.y + singleSliceHeight * it).toDouble())) }
}

/**
 * Center [Point] of the [Rect]
 */
val Rect.center: Point
    get() = Point((this.tl().x + this.br().x) / 2, (this.tl().y + this.br().y) / 2)

operator fun Rect.plus(rect2: Rect) = this.merge(rect2, 0)

fun Rect.merge(rect2: Rect, mergeType: Int) = Rect(Integer.min(this.x, rect2.x),
        Integer.min(this.y, rect2.y),

        this.width + rect2.width + mergeType,
        Integer.max(this.height, rect2.height))

object RectManipulator {

    val MAX_AREA_THRESHOLD = 1000.0
    val ASPECT_RATIO = 2.0


    //TODO refactor

    fun getDistanceBetweenRectCenters(rect1: Rect, rect2: Rect): IntArray {

        val distance = IntArray(2)

        //vertical distance
        distance[0] = Math.abs(rect1.y + rect1.height / 2 - (rect2.y + rect2.height / 2))
        //horizontal distance
        distance[1] = Math.abs((rect1.x + rect1.width) / 2) - (rect2.y + rect2.height) / 2

        return distance
    }

    fun getDistanceBetweenBorders(rect1: Rect, rect2: Rect): Int {
        return Math.abs(rect1.x + rect1.width - rect2.x)
    }

    fun getVerticalDistance(rect1: Rect, rect2: Rect): Int {
        return Math.abs(rect1.y + rect1.height - (rect2.y + rect2.height))
    }

    fun contains(rect1: Rect, rect2: Rect): Boolean {
        if (rect1 === rect2)
            return true

        return rect1.contains(rect2.center)
    }


    internal fun isValidCharArea(rect: Rect): Boolean {
        if (rect.height < 15 || rect.width == 0) {
            return false
        }

        val area = rect.size().area()
        val isValidArea = area < MAX_AREA_THRESHOLD

        val realRatio = (rect.height / rect.width).toDouble()
        val isValidAspectRatio = realRatio < ASPECT_RATIO

        return isValidArea && isValidAspectRatio
    }
}