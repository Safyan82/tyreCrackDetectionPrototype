package com.example.tyre
import android.os.Debug
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.tyre.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF


class MainActivity : AppCompatActivity() {
    lateinit var selectBtn: Button
    lateinit var predBtn: Button
    lateinit var resView: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap
    lateinit var model: Model
    lateinit var imageProcessor: ImageProcessor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.selectBtn)
        predBtn = findViewById(R.id.predictBtn)
        resView = findViewById(R.id.resView)
        imageView = findViewById(R.id.imageView)

        // Load the TensorFlow Lite model
        model = Model.newInstance(this)

        // Create the image processor
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        selectBtn.setOnClickListener {
            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

//        predBtn.setOnClickListener {
//            if (::bitmap.isInitialized) {
//                val tensorImage = TensorImage(DataType.FLOAT32)
//                tensorImage.load(bitmap)
//                val processedImage = imageProcessor.process(tensorImage)
//
//                val inputShape = intArrayOf(1, 224, 224, 3)
//                val inputBuffer = processedImage.buffer
//
//                // Convert the input buffer to a ByteArray
//                val inputByteArray = ByteArray(inputBuffer.remaining())
//                inputBuffer.get(inputByteArray)
//
//                // Convert the ByteArray to a float array
//                val inputFloatArray = FloatArray(inputByteArray.size / 4)
//                ByteBuffer.wrap(inputByteArray).asFloatBuffer().get(inputFloatArray)
//
//                // Normalize pixel values to [0, 1]
//                val normalizedFloatArray = inputFloatArray.map { it / 255.0f }.toFloatArray()
//
//                // Create a new ByteBuffer from the normalized float array
//                val normalizedByteBuffer = ByteBuffer.allocate(normalizedFloatArray.size * 4).apply {
//                    asFloatBuffer().put(normalizedFloatArray)
//                }
//
//                // Create a TensorBuffer with the normalized ByteBuffer
//                val inputFeature0 = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
//                inputFeature0.loadBuffer(normalizedByteBuffer)
//
//                val outputs = model.process(inputFeature0)
//                val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray
//
//                // Assuming the model output is a probability between 0 and 1
//                val crackProbability = outputFeature0[0]
//                val threshold = 0.5 // Adjust the threshold as needed
//
//                Log.d(crackProbability.toString(), "res")
//                val resultText = if (crackProbability >= threshold) {
//                    "Crack (Probability: ${String.format("%.2f", crackProbability)})"
//                } else {
//                    "No Crack (Probability: ${String.format("%.2f", 1 - crackProbability)})"
//                }
//
//                resView.text = "Result: $resultText"
//            } else {
//                resView.text = "Please select an image first"
//            }
//        }
        predBtn.setOnClickListener {
            if (::bitmap.isInitialized) {

                // Add a marker before starting crack prediction
                Debug.startMethodTracing("CrackPrediction")

                val inputImage = preprocessImage(bitmap)

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                inputFeature0.loadArray(inputImage)

                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                val predictions = outputFeature0.floatArray

                // Print the preprocessed input data for debugging
//                Log.d("Preprocessed Input", inputImage.contentDeepToString())

                // Print the model's output for debugging
                Log.d("Model Output", predictions.contentToString())

                // Assuming the model output is a probability between 0 and 1
                val crackProbability = predictions[0]
                val threshold = 0.6f // Adjust the threshold as needed

                Log.d("Model Output prob", crackProbability.toString())
                if (crackProbability >= threshold) {
                    resView.text = "Crack (Probability: ${String.format("%.2f", crackProbability)})"
                    // Draw bounding box on the image
                    val bitmapWithBoundingBox = drawBoundingBox(bitmap, crackProbability)
                    imageView.setImageBitmap(bitmapWithBoundingBox)
                } else {
                    resView.text = "No Crack Found"
                }


                // Add a marker after crack prediction is complete
                Debug.stopMethodTracing()

            } else {
                resView.text = "Please select an image first"
            }
        }

    }

    private fun drawBoundingBox(bitmap: Bitmap, crackProbability: Float): Bitmap {
        val bitmapWithBoundingBox = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmapWithBoundingBox)
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.color = Color.RED
        paint.strokeWidth = 5f

        val boundingBoxLeft = 0f
        val boundingBoxTop = 0f
        val boundingBoxRight = bitmap.width.toFloat()
        val boundingBoxBottom = bitmap.height.toFloat()

        val boundingBoxRect = RectF(boundingBoxLeft, boundingBoxTop, boundingBoxRight, boundingBoxBottom)
        canvas.drawRect(boundingBoxRect, paint)

        return bitmapWithBoundingBox
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputArray = FloatArray(1 * 224 * 224 * 3)

        val pixels = IntArray(224 * 224)
        resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        for (i in pixels.indices) {
            val pixelValue = pixels[i]

            inputArray[i * 3] = (pixelValue shr 16 and 0xFF) / 255.0f
            inputArray[i * 3 + 1] = (pixelValue shr 8 and 0xFF) / 255.0f
            inputArray[i * 3 + 2] = (pixelValue and 0xFF) / 255.0f
        }

        return inputArray
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100 && resultCode == RESULT_OK) {
            val uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release model resources
        model.close()
    }
}