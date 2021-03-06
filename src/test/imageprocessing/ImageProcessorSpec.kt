package imageprocessing

import com.github.sshaddicts.lucrecium.imageProcessing.ImageProcessor
import com.github.sshaddicts.lucrecium.util.plus
import com.github.sshaddicts.lucrecium.util.split
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import kotlin.test.assertEquals
import kotlin.test.assertTrue


object ImageProcessorSpec : Spek({

    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

    val testCaseDir = "testCase/"
    val example = testCaseDir + "good.jpg"
    val skewed = testCaseDir + "skew.jpg"
    val lineSegm = testCaseDir + "lineNumber.png"

    describe("Rect") {
        it("should be possible to split rect") {
            val image = ImageProcessor.loadImage(lineSegm)
            val rect = Rect(0, 0, image.width(), image.height())

            val mean = 22
            val split = rect.split(rect.height / mean, false)

            assertTrue(split.size == 3)
        }

        it("should be possible to merge rects") {
            val rect1 = Rect(0, 0, 25, 25)
            val rect2 = Rect(25, 25, 25, 25)

            val merged = rect1 + rect2

            assertEquals(50, merged.width)
        }
    }

    describe("ImageProcessor") {
        on("descewing") {
            val processor = ImageProcessor()

            val deskew = processor::class.java.getDeclaredMethod("deskew", Mat::class.java)
            deskew.isAccessible = true

            val result = deskew.invoke(processor, ImageProcessor.loadImage(skewed)) as Double

            it("should not be equal to zero") {
                assertTrue(result != .0)
            }
        }

        on("finding text regions") {
            val processor = ImageProcessor()
            val result = processor.findTextRegions(example)

            it("should return the result that contains letters") {
                assertTrue(
                        result.chars.isNotEmpty(),
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

            it("should have approximately 3 lines") {
                assertEquals(3, result)
            }
        }
    }
})