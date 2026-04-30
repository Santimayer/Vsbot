package com.example.ilumel

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.ilumel.objectdetector.ObjectDetectorHelper
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// Clase simple para representar lo que detectamos en la UI
data class DetectionUI(
    val label: String,
    val score: Float
)

class MainActivity : ComponentActivity(), ObjectDetectorHelper.DetectorListener {

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var cameraExecutor: ExecutorService

    // Este es nuestro "Estado". Compose detectará cambios aquí y refrescará la lista.
    private var detectedObjects = mutableStateListOf<DetectionUI>()

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, "Permiso de cámara denegado", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        objectDetectorHelper = ObjectDetectorHelper(
            context = this,
            objectDetectorListener = this,
            currentDelegate = ObjectDetectorHelper.DELEGATE_GPU,
            modelName = "efficientdet-lite2.tflite"
        )

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            DetectionScreen(detectedObjects)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        objectDetectorHelper.detectLivestreamFrame(imageProxy)
                    }
                }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("MainActivity", "Error al iniciar cámara", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // --- Implementación del Listener ---

    override fun onResults(resultBundle: ObjectDetectorHelper.ResultBundle) {
        val height = resultBundle.inputImageHeight
        val width = resultBundle.inputImageWidth
        val detections = resultBundle.results.firstOrNull()?.detections() ?: emptyList()

        // 1. Logcat Debug: Aquí mantenemos la data cruda (incluyendo coordenadas para tu micro)
        detections.forEach { detection ->
            val category = detection.categories().firstOrNull()
            val label = category?.categoryName() ?: "Desconocido"
            val score = category?.score() ?: 0f
            val box = detection.boundingBox()

            Log.d("DETECCION_DATA", "Resolución: $width x $height |Objeto: $label | Score: ${"%.2f".format(score)} | Box: [L:${box.left}, T:${box.top}, R:${box.right}, B:${box.bottom}]")
        }

        // 2. UI Update: Mapeamos y ordenamos para las barritas en pantalla
        val newDetections = detections.map { detection ->
            val category = detection.categories().firstOrNull()
            DetectionUI(
                label = category?.categoryName() ?: "Desconocido",
                score = category?.score() ?: 0f
            )

        }.sortedByDescending { it.score }

        // Actualizamos la lista en el hilo principal de la UI
        runOnUiThread {
            detectedObjects.clear()
            detectedObjects.addAll(newDetections)
        }
    }

    override fun onError(error: String) {
        Log.e("ObjectDetector", "Error: $error")
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        objectDetectorHelper.clearObjectDetector()
    }
}

// --- Componentes de Interfaz ---

@Composable
fun DetectionScreen(objects: List<DetectionUI>) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .statusBarsPadding()
    ) {
        Text(
            text = "Objetos Detectados",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        if (objects.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(text = "Apunte a un objeto...", color = Color.Gray)
            }
        } else {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(objects) { obj ->
                    DetectionItem(obj)
                }
            }
        }
    }
}

@Composable
fun DetectionItem(detection: DetectionUI) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = detection.label, fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
                Text(text = "${(detection.score * 100).toInt()}%", color = MaterialTheme.colorScheme.primary)
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // La barra de porcentaje que querías
            LinearProgressIndicator(
                progress = { detection.score },
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.surfaceVariant
            )
        }
    }
}
