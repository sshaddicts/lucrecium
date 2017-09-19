package imageprocessing

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor
import com.github.sshaddicts.lucrecium.util.RectManipulator
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue


object ImageProcessorSpec : Spek({

    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

    val testCaseDir = "testCase/";
    val example = testCaseDir + "good.jpg";
    val skewed = testCaseDir + "skew.jpg";
    val lineSegm = testCaseDir + "lineNumber.png";

    describe("Rect") {
        it("should be possible to split rect") {
            val image = ImageProcessor.loadImage(lineSegm)
            val rect = Rect(0, 0, image.width(), image.height())

            val mean = 22
            val split = RectManipulator.split(rect, rect.height / mean, false)

            assertTrue(split.size == 3)
        }

        it("should be possible to merge rects") {
            val rect1 = Rect(0, 0, 25, 25)
            val rect2 = Rect(25, 25, 25, 25)

            val merged = RectManipulator.merge(rect1, rect2, 0)

            assertEquals(50, merged.width)
        }
    }

    describe("ImageProcessor") {
        on("descewing") {
            val processor = ImageProcessor()

            val deskew = processor::class.java.getDeclaredMethod("deskew", Mat::class.java)
            deskew.isAccessible = true

            val result = deskew.invoke(processor, ImageProcessor.loadImage(skewed)) as Double

            it("should work properly") {
                assertFalse(result == 0.toDouble())
            }
        }

        on("finding text regions") {
            val processor = ImageProcessor()
            val result = processor.findTextRegions(example)

            it("should return the result that contains letters") {
                assertTrue(
                        result.chars.size > 0,
                        "mat.size() is ${result.chars.size}, but should never be"
                )
            }
        }

        on("approximating line numbers for given image") {
            val processor = ImageProcessor()

            val lineNumberFor = processor::class.java.getDeclaredMethod(
                    "approximateLineNumberFor", Mat::class.java
            )

            lineNumberFor.isAccessible = true
            val result = lineNumberFor.invoke(processor, ImageProcessor.loadImage(lineSegm)) as Int

            it("should work properly") {
                assertEquals(3, result)
            }
        }
    }
})