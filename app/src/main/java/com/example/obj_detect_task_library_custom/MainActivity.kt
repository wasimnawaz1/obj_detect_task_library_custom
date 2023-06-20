package com.example.obj_detect_task_library_custom

import android.content.ContentValues.TAG
import android.graphics.*
import android.graphics.ImageDecoder.ImageInfo
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class MainActivity : AppCompatActivity() {

    private lateinit var loadImageButton: Button
    private lateinit var imageView: ImageView
    private lateinit var selectedImage: Bitmap
    private lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.image_view)
        resultTextView = findViewById(R.id.result_textview)
        loadImageButton = findViewById(R.id.load_image_button)

        loadImageButton.setOnClickListener {
            pickImage.launch("image/*")
        }

        imageView.setOnClickListener {
            runTensorFlowLiteObjectDetection(selectedImage)
        }

    }
    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                "To detect objects, click the image".also { resultTextView.text = it } // Clear text
                val source = ImageDecoder.createSource(contentResolver, uri)
                selectedImage = ImageDecoder.decodeBitmap(source)
                //ImageDecoder.decodeBitmap(source)
                { imageDecoder: ImageDecoder, imageInfo: ImageInfo?, source1: ImageDecoder.Source? ->
                    imageDecoder.isMutableRequired = true
                }
                imageView.setImageBitmap(selectedImage)
            }
        }
    // Task Library Object Detection function
    private fun runTensorFlowLiteObjectDetection(bitmap: Bitmap) {
        // Initialization
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(BaseOptions.builder().build())   //useGpu()    before .build()
            .setMaxResults(3)
            .build()
        val objectDetector = ObjectDetector.createFromFileAndOptions(
            //this, "lite-model_efficientdet_lite0_detection_metadata_1.tflite", options
            this, "lite-model_efficientdet_lite1_detection_default_1_with_labels.tflite", options
            //ssd_mobilenet_v1_1_metadata_2
            // ssd_mobilenet_v1_1_default_1_with_labels

        )
        //ssd_mobilenet_v1_1_metadata_1.tflite   // this model gives inaccurate results
        //mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
        //"ssd_mobilenet_v1_1_default_1.tflite"        // gives error because metadata (label file) is not there
        //"ssd_mobilenet_v1_1_metadata_2.tflite"
        // Run inference
        val inputImage = TensorImage.fromBitmap(bitmap)
        val outputs: List<Detection> = objectDetector.detect(inputImage)

        val detectedObjects = outputs.map {
            var text = "Unknown"

            if (it.categories.isNotEmpty()) {
                val category = it.categories.first()
                text = "${category.label}, ${category.score.times(100).toInt()}%"
            }

            dataClassBoxText(it.boundingBox, text)
        }

        val visualizedResult = drawBoxTextDetections(bitmap, detectedObjects)
        imageView.setImageBitmap(visualizedResult)
    }

    // Draw bounding boxes with object names around detected objects

    private fun drawBoxTextDetections(
        bitmap: Bitmap,
        detectionResults: List<dataClassBoxText>
    ): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        val pen = Paint()
        pen.textAlign = Paint.Align.LEFT

        detectionResults.forEach {
            // draw bounding box
            pen.color = Color.GREEN
            pen.strokeWidth = 1.5F
            pen.style = Paint.Style.STROKE
            val box = it.box
            canvas.drawRect(box, pen)

            val tagSize = Rect(0, 0, 0, 0)

            // calculate the right font size
            pen.style = Paint.Style.FILL_AND_STROKE
            pen.color = Color.RED
            pen.strokeWidth = 1.5F
            pen.textSize = 80F
            pen.getTextBounds(it.text, 0, it.text.length, tagSize)
            val fontSize: Float = pen.textSize * box.width() / tagSize.width()

            // adjust the font size so texts are inside the bounding box
            if (fontSize < pen.textSize) pen.textSize = fontSize

            var margin = (box.width() - tagSize.width()) / 2.0F
            if (margin < 0F) margin = 0F
            canvas.drawText(
                it.text, box.left + margin,
                box.top + tagSize.height().times(1F), pen
            )
        }
        return outputBitmap
    }

   private fun print_results(detectedObjects: List<Detection>) {
        detectedObjects.forEachIndexed { index, detectedObject ->
            val box = detectedObject.boundingBox

            Log.d(TAG, "Detected object: $index")
            //Log.d(TAG, " trackingId: ${detectedObject.trackingId}")
            Log.d(TAG, " boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")
            detectedObject.categories.forEach {
                Log.d(TAG, " categories: ${it.label}")
                Log.d(TAG, " confidence: ${it.score}")
            }
        }
   }
}

// Data class to store detection results for visualization
data class dataClassBoxText(val box: RectF, val text: String)


