# Complete Step-by-Step Guide: Building a Camera App from Scratch

## 📱 Part 1: Create a New Project

### Step 1: Open Android Studio
1. Launch Android Studio
2. Click **"New Project"** (or File → New → New Project)

### Step 2: Choose Project Template
1. Select **"Empty Views Activity"**
2. Click **Next**

### Step 3: Configure Your Project
1. **Name**: `CameraX2`
2. **Package name**: `com.example.camerax2`
3. **Save location**: Choose where to save
4. **Language**: `Kotlin`
5. **Minimum SDK**: `API 26: Android 8.0 (Oreo)`
6. **Build configuration language**: `Kotlin DSL (Recommended)`
7. Click **Finish**

### Step 4: Wait for Project Setup
- Android Studio will download dependencies
- Wait for "Gradle sync" to complete (see progress bar at bottom)

---

## 🔧 Part 2: Configure Project Files

### Step 5: Update app/build.gradle.kts
1. In the left panel, navigate to: `Gradle Scripts` → `build.gradle.kts (Module :app)`
2. **Replace the entire file** with:

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.camerax2"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.camerax2"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // CameraX dependencies
    val camerax_version = "1.3.0"
    implementation("androidx.camera:camera-core:${camerax_version}")
    implementation("androidx.camera:camera-camera2:${camerax_version}")
    implementation("androidx.camera:camera-lifecycle:${camerax_version}")
    implementation("androidx.camera:camera-view:${camerax_version}")
    
    // AndroidX dependencies
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    
    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
```

3. Click **"Sync Now"** in the yellow bar that appears

---

## 📄 Part 3: Create the MainActivity Code

### Step 6: Update MainActivity.kt
1. Navigate to: `app` → `java` → `com.example.camerax2` → `MainActivity`
2. **Delete everything** in the file
3. **Paste this entire code**:

```kotlin
package com.example.camerax2

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Range
import android.view.MotionEvent
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.camerax2.databinding.ActivityMainBinding
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    @SuppressLint("UnsafeOptInUsageError")
    private var camera2CameraControl: Camera2CameraControl? = null

    private var isoRange: Range<Int>? = null
    private var exposureTimeRange: Range<Long>? = null
    private var minFocusDistance: Float = 0f
    
    // Store ISO bounds separately for easier access
    private var isoLower: Int = 0
    private var isoUpper: Int = 0

    companion object {
        private const val TAG = "camerax2"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).apply {
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }.toTypedArray()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        binding.captureButton.setOnClickListener { takePhoto() }
        binding.switchCameraButton.setOnClickListener { switchCamera() }

        setupProControlsListeners()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))

        binding.previewView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                camera?.let { cam ->
                    val factory = binding.previewView.meteringPointFactory
                    val point = factory.createPoint(event.x, event.y)
                    val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF)
                        .setAutoCancelDuration(5, java.util.concurrent.TimeUnit.SECONDS)
                        .build()
                    cam.cameraControl.startFocusAndMetering(action)
                }
                return@setOnTouchListener true
            }
            false
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val currentCameraProvider = cameraProvider ?: run {
            Log.e(TAG, "Camera initialization failed.")
            Toast.makeText(this, "Camera initialization failed.", Toast.LENGTH_SHORT).show()
            return
        }

        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .build()

        try {
            currentCameraProvider.unbindAll()
            camera = currentCameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageCapture
            )

            camera?.let {
                camera2CameraControl = Camera2CameraControl.from(it.cameraControl)
            }

            setupCameraInfoAndControls()

        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
            Toast.makeText(this, "Could not start camera: ${exc.message}", Toast.LENGTH_LONG).show()
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun setupCameraInfoAndControls() {
        val camInfo = camera?.cameraInfo ?: return
        val cam2Info = Camera2CameraInfo.from(camInfo)

        // EV Compensation
        val exposureState = camInfo.exposureState
        if (exposureState.isExposureCompensationSupported) {
            val evRange = exposureState.exposureCompensationRange
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                binding.evSeekBar.min = evRange.lower
            }
            binding.evSeekBar.max = evRange.upper
            binding.evSeekBar.progress = exposureState.exposureCompensationIndex - evRange.lower
            binding.evValueText.text = "%.1f".format(exposureState.exposureCompensationIndex * exposureState.exposureCompensationStep.toFloat())
            binding.evSeekBar.visibility = View.VISIBLE
            binding.evLabel.visibility = View.VISIBLE
            binding.evValueText.visibility = View.VISIBLE
        } else {
            binding.evSeekBar.visibility = View.GONE
            binding.evLabel.visibility = View.GONE
            binding.evValueText.visibility = View.GONE
        }

        // Zoom
        val zoomState = camInfo.zoomState.value
        if (zoomState != null && (zoomState.maxZoomRatio > zoomState.minZoomRatio)) {
            val minZoom = (zoomState.minZoomRatio * 10).toInt()
            val maxZoom = (zoomState.maxZoomRatio * 10).toInt()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                binding.zoomSeekBar.min = minZoom
            }
            binding.zoomSeekBar.max = maxZoom
            binding.zoomSeekBar.progress = (zoomState.zoomRatio * 10).toInt() - minZoom
            binding.zoomValueText.text = "%.1fx".format(zoomState.zoomRatio)
            binding.zoomSeekBar.visibility = View.VISIBLE
            binding.zoomLabel.visibility = View.VISIBLE
            binding.zoomValueText.visibility = View.VISIBLE
        } else {
            binding.zoomSeekBar.visibility = View.GONE
            binding.zoomLabel.visibility = View.GONE
            binding.zoomValueText.visibility = View.GONE
        }

        // Flash Mode
        if (camInfo.hasFlashUnit()) {
            val flashModes = listOf("AUTO", "ON", "OFF")
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, flashModes)
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            binding.flashModeSpinner.adapter = adapter
            binding.flashModeSpinner.setSelection(
                when(imageCapture?.flashMode) {
                    ImageCapture.FLASH_MODE_ON -> 1
                    ImageCapture.FLASH_MODE_OFF -> 2
                    else -> 0
                }
            )
            binding.flashModeSpinner.visibility = View.VISIBLE
            binding.flashLabel.visibility = View.VISIBLE
        } else {
            binding.flashModeSpinner.visibility = View.GONE
            binding.flashLabel.visibility = View.GONE
        }

        // Manual Focus
        minFocusDistance = cam2Info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        val focusSupported = minFocusDistance > 0f ||
                (cam2Info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.isNotEmpty() == true &&
                        cam2Info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) != null)

        if (focusSupported) {
            binding.focusSeekBar.max = 100
            binding.focusSeekBar.progress = 0
            binding.focusValueText.text = "Auto"
            binding.focusSeekBar.visibility = View.VISIBLE
            binding.focusLabel.visibility = View.VISIBLE
            binding.focusValueText.visibility = View.VISIBLE
        } else {
            binding.focusSeekBar.visibility = View.GONE
            binding.focusLabel.visibility = View.GONE
            binding.focusValueText.visibility = View.GONE
        }

        // ISO
        isoRange = cam2Info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        if (isoRange != null) {
            isoLower = isoRange!!.lower
            isoUpper = isoRange!!.upper
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                binding.isoSeekBar.min = isoLower
            }
            binding.isoSeekBar.max = isoUpper
            // For Camera2 interop, we can't easily get current values, so default to auto/lower
            binding.isoSeekBar.progress = isoLower
            binding.isoValueText.text = "Auto"
            binding.isoSeekBar.visibility = View.VISIBLE
            binding.isoLabel.visibility = View.VISIBLE
            binding.isoValueText.visibility = View.VISIBLE
        } else {
            binding.isoSeekBar.visibility = View.GONE
            binding.isoLabel.visibility = View.GONE
            binding.isoValueText.visibility = View.GONE
        }

        // Shutter Speed
        exposureTimeRange = cam2Info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        if (exposureTimeRange != null) {
            val minEtMs = exposureTimeRange!!.lower / 1_000_000L
            val maxEtMs = minOf(exposureTimeRange!!.upper / 1_000_000L, 2000L)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                binding.shutterSpeedSeekBar.min = 0
            }
            binding.shutterSpeedSeekBar.max = 100

            // Default to auto since we can't easily get current exposure time
            binding.shutterSpeedSeekBar.progress = 0
            binding.shutterSpeedValueText.text = "Auto"
            
            binding.shutterSpeedSeekBar.visibility = View.VISIBLE
            binding.shutterSpeedLabel.visibility = View.VISIBLE
            binding.shutterSpeedValueText.visibility = View.VISIBLE
        } else {
            binding.shutterSpeedSeekBar.visibility = View.GONE
            binding.shutterSpeedLabel.visibility = View.GONE
            binding.shutterSpeedValueText.visibility = View.GONE
        }

        // White Balance
        val awbModes = cam2Info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES)
        if (awbModes != null && awbModes.isNotEmpty()) {
            val awbModeStrings = awbModes.map { modeToStringAWB(it) }
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, awbModeStrings)
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            binding.whiteBalanceSpinner.adapter = adapter

            // Default to AUTO
            val autoIndex = awbModes.indexOf(CameraMetadata.CONTROL_AWB_MODE_AUTO)
            binding.whiteBalanceSpinner.setSelection(if (autoIndex != -1) autoIndex else 0)
            binding.whiteBalanceSpinner.visibility = View.VISIBLE
            binding.whiteBalanceLabel.visibility = View.VISIBLE
        } else {
            binding.whiteBalanceSpinner.visibility = View.GONE
            binding.whiteBalanceLabel.visibility = View.GONE
        }
    }

    private fun modeToStringAWB(mode: Int): String {
        return when (mode) {
            CameraMetadata.CONTROL_AWB_MODE_OFF -> "OFF"
            CameraMetadata.CONTROL_AWB_MODE_AUTO -> "AUTO"
            CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT -> "INCANDESCENT"
            CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT -> "FLUORESCENT"
            CameraMetadata.CONTROL_AWB_MODE_WARM_FLUORESCENT -> "WARM_FLUORESCENT"
            CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT -> "DAYLIGHT"
            CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT -> "CLOUDY_DAYLIGHT"
            CameraMetadata.CONTROL_AWB_MODE_TWILIGHT -> "TWILIGHT"
            CameraMetadata.CONTROL_AWB_MODE_SHADE -> "SHADE"
            else -> "UNKNOWN ($mode)"
        }
    }

    private fun stringToModeAWB(modeString: String): Int {
        return when (modeString) {
            "OFF" -> CameraMetadata.CONTROL_AWB_MODE_OFF
            "AUTO" -> CameraMetadata.CONTROL_AWB_MODE_AUTO
            "INCANDESCENT" -> CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT
            "FLUORESCENT" -> CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT
            "WARM_FLUORESCENT" -> CameraMetadata.CONTROL_AWB_MODE_WARM_FLUORESCENT
            "DAYLIGHT" -> CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT
            "CLOUDY_DAYLIGHT" -> CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
            "TWILIGHT" -> CameraMetadata.CONTROL_AWB_MODE_TWILIGHT
            "SHADE" -> CameraMetadata.CONTROL_AWB_MODE_SHADE
            else -> CameraMetadata.CONTROL_AWB_MODE_AUTO
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun setupProControlsListeners() {
        // EV Compensation
        binding.evSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val evRange = camera?.cameraInfo?.exposureState?.exposureCompensationRange
                    val actualProgress = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        progress
                    } else {
                        progress + (evRange?.lower ?: 0)
                    }
                    camera?.cameraControl?.setExposureCompensationIndex(actualProgress)
                    val evValue = actualProgress * (camera?.cameraInfo?.exposureState?.exposureCompensationStep?.toFloat() ?: 0f)
                    binding.evValueText.text = "%.1f".format(evValue)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Zoom
        binding.zoomSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val zoomState = camera?.cameraInfo?.zoomState?.value
                    val minZoom = (zoomState?.minZoomRatio ?: 1f) * 10f
                    val actualProgress = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        progress / 10f
                    } else {
                        (progress + minZoom.toInt()) / 10f
                    }
                    camera?.cameraControl?.setZoomRatio(actualProgress.coerceIn(
                        zoomState?.minZoomRatio ?: 1f,
                        zoomState?.maxZoomRatio ?: 1f
                    ))
                    binding.zoomValueText.text = "%.1fx".format(actualProgress)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Flash Mode
        binding.flashModeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                imageCapture?.flashMode = when (position) {
                    0 -> ImageCapture.FLASH_MODE_AUTO
                    1 -> ImageCapture.FLASH_MODE_ON
                    2 -> ImageCapture.FLASH_MODE_OFF
                    else -> ImageCapture.FLASH_MODE_AUTO
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Manual Focus
        binding.focusSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser && (minFocusDistance > 0f || camera2CameraControl != null)) {
                    val builder = CaptureRequestOptions.Builder()
                    if (progress == 0) {
                        builder.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                        binding.focusValueText.text = "Auto"
                    } else {
                        val focusDistanceValue = (progress / 100.0f) * minFocusDistance
                        builder.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, focusDistanceValue)
                        builder.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                        binding.focusValueText.text = "%.2f D".format(focusDistanceValue)
                    }
                    camera2CameraControl?.addCaptureRequestOptions(builder.build())?.addListener({}, ContextCompat.getMainExecutor(this@MainActivity))
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // ISO
        binding.isoSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser && isoRange != null) {
                    val actualIsoValue = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        progress
                    } else {
                        progress + isoLower
                    }
                    val isoValue = actualIsoValue.coerceIn(isoLower, isoUpper)
                    val builder = CaptureRequestOptions.Builder()
                        .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, isoValue)
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                    camera2CameraControl?.addCaptureRequestOptions(builder.build())?.addListener({}, ContextCompat.getMainExecutor(this@MainActivity))
                    binding.isoValueText.text = isoValue.toString()
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Shutter Speed
        binding.shutterSpeedSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progressValue: Int, fromUser: Boolean) {
                if (fromUser && exposureTimeRange != null) {
                    val minEtNs = exposureTimeRange!!.lower
                    val maxEtNsForUi = minOf(exposureTimeRange!!.upper, 2_000_000_000L)

                    val exposureTimeNs = minEtNs + (((maxEtNsForUi - minEtNs) * progressValue) / 100L)
                    val coercedExposureTime = exposureTimeNs.coerceIn(minEtNs, exposureTimeRange!!.upper)

                    val builder = CaptureRequestOptions.Builder()
                        .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, coercedExposureTime)
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                    camera2CameraControl?.addCaptureRequestOptions(builder.build())?.addListener({}, ContextCompat.getMainExecutor(this@MainActivity))
                    binding.shutterSpeedValueText.text = "${coercedExposureTime / 1_000_000L}ms"
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // White Balance
        binding.whiteBalanceSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedModeString = parent?.getItemAtPosition(position) as? String ?: return
                val awbMode = stringToModeAWB(selectedModeString)
                val builder = CaptureRequestOptions.Builder()
                    .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, awbMode)
                camera2CameraControl?.addCaptureRequestOptions(builder.build())?.addListener({}, ContextCompat.getMainExecutor(this@MainActivity))
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ProCameraX-Image")
            }
        }

        val outputOptions = ImageCapture.OutputFileOptions.Builder(
            contentResolver,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        ).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    Toast.makeText(baseContext, "Photo capture failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            }
        )
    }

    private fun switchCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        bindCameraUseCases()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
```

---

## 🎨 Part 4: Create the Layout

### Step 7: Create activity_main.xml
1. Navigate to: `app` → `res` → `layout`
2. Right-click on `layout` folder → **New** → **Layout Resource File**
3. **File name**: `activity_main`
4. **Root element**: Keep default
5. Click **OK**
6. Switch to **Code** view (bottom tabs)
7. **Replace everything** with this:

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent