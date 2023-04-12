package com.example.horse_human_classify

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.ImageDecoder.ImageInfo
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.IOException

class MainActivity : AppCompatActivity() {

    // Constants for image classifier
    companion object {
        private const val REQUEST_CODE_LOAD_IMAGE = 1
    }

    private lateinit var loadImageButton: Button
    private lateinit var imageView: ImageView
    private lateinit var selectedImage: Bitmap
    private lateinit var classifier: ImageClassifier
    private lateinit var resultTextView: TextView

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                resultTextView.text = "" // Clear classification result text
                val source = ImageDecoder.createSource(contentResolver, uri)
                selectedImage = ImageDecoder.decodeBitmap(source)
                //ImageDecoder.decodeBitmap(source)
                { imageDecoder: ImageDecoder, imageInfo: ImageInfo?, source1: ImageDecoder.Source? ->
                    imageDecoder.isMutableRequired = true
                }
                imageView.setImageBitmap(selectedImage)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        loadImageButton = findViewById(R.id.load_image_button)
        imageView = findViewById(R.id.image_view)
        resultTextView = findViewById(R.id.classification_result_textview)

        try {
            classifier = ImageClassifier(this)
        } catch (e: IOException) {
            e.printStackTrace()
        }

        loadImageButton.setOnClickListener {
            pickImage.launch("image/*")
        }

        imageView.setOnClickListener {
            val result = classifier.classifyImage(selectedImage)
            resultTextView.text = result
        }
    }
}
