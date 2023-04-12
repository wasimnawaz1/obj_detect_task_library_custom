package com.example.horse_human_classify

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

class ImageClassifier(private val context: Context) {

    companion object {
        private const val BATCH_SIZE = 1
        private const val INPUT_SIZE = 224
        private const val NUM_CLASSES = 2
        private const val PIXEL_SIZE = 3
        private const val IMAGE_MEAN = 0
        private const val IMAGE_STD = 255.0f
        private const val MODEL_FILE = "model.tflite"
        private const val LABEL_FILE = "labels.txt"
    }

    private val interpreter: Interpreter by lazy {
        Interpreter(loadModelFile(), Interpreter.Options())
    }
    private val labelList: List<String> by lazy {
        loadLabels()
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(): List<String> {
        context.assets.open(LABEL_FILE).use { inputStream ->
            InputStreamReader(inputStream).use { inputStreamReader ->
                BufferedReader(inputStreamReader).useLines { lines ->
                    return lines.toList()
                }
            }
        }
    }

    fun classifyImage(bitmap: Bitmap): String {
        // Resize the bitmap to the input size required by the model
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE,true)

        // Convert the resized bitmap to a ByteBuffer
        val byteBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Run inference using the model on the input ByteBuffer
        val output = Array(BATCH_SIZE) { FloatArray(NUM_CLASSES) }
        interpreter.run(byteBuffer, output)

        // Get the index of the class with the highest probability
        val index = output[0].indices.maxByOrNull { output[0][it] } ?: -1

        // Return the label corresponding to the index of the highest probability class
        return if (index >= 0 && index < labelList.size) labelList[index] else "Unknown"
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4* INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val pixelValue = pixels[i * INPUT_SIZE + j]
                byteBuffer.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        return byteBuffer
    }
}
