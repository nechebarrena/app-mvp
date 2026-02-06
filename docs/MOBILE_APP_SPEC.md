# Mobile App Specification

> **Version:** 1.0.0  
> **Last Updated:** January 31, 2026  
> **Status:** MVP Phase 1 - Remote Processing  
> **Target Platforms:** Android (primary), iOS (secondary)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Core Modules](#5-core-modules)
6. [API Contract](#6-api-contract)
7. [UI/UX Specification](#7-uiux-specification)
8. [Data Models](#8-data-models)
9. [Development Phases](#9-development-phases)
10. [Testing Strategy](#10-testing-strategy)
11. [Configuration & Environment](#11-configuration--environment)
12. [Deployment Guide](#12-deployment-guide)

---

## 1. Project Overview

### 1.1 Objective

Build a mobile application for **weightlifting video analysis** that:
1. Captures video clips of lifting sessions
2. Allows manual selection of the barbell disc in the first frame
3. Sends video + selection to a remote server for AI processing
4. Receives and displays analysis results (trajectories, metrics, visualizations)
5. **Future:** Runs the entire AI pipeline locally on the device

### 1.2 MVP Scope

**Phase 1 (Current):** Remote Processing
- Video capture from camera
- Interactive disc selection on first frame
- Upload to remote server via REST API
- Display results from JSON response
- Basic metric visualizations

**Phase 2 (Future):** Local Processing
- On-device AI inference (CoreML/TFLite)
- Offline capability
- No server dependency

### 1.3 Key Constraints

- **Single user focus:** MVP is for personal use/testing
- **Network dependent:** Phase 1 requires internet connection
- **Resource intensive:** Future local inference requires GPU-capable devices
- **Cross-platform design:** Code structure must support iOS later

---

## 2. Architecture

### 2.1 Phase 1: Remote Processing

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MOBILE APP                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐    │
│  │   Camera    │──▶│   Video     │──▶│    Disc Selector        │    │
│  │   Module    │   │   Storage   │   │    (Frame 0 + Touch)    │    │
│  └─────────────┘   └─────────────┘   └───────────┬─────────────┘    │
│                                                   │                  │
│                                      ┌───────────▼─────────────┐    │
│                                      │      API Client         │    │
│                                      │   - Upload video        │    │
│                                      │   - Poll status         │    │
│                                      │   - Fetch results       │    │
│                                      └───────────┬─────────────┘    │
│                                                   │                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Results Display                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │
│  │  │   Video     │  │   Metric    │  │    Trajectory       │  │    │
│  │  │   Player    │  │   Graphs    │  │    Overlay          │  │    │
│  │  │  (+ masks)  │  │             │  │                     │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTPS (Ngrok tunnel)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      REMOTE SERVER (Python)                          │
│                                                                      │
│  FastAPI ──▶ Pipeline ──▶ YOLO ──▶ Tracking ──▶ Metrics             │
│                                                                      │
│  Returns: JSON with tracks, metrics, summary (~200KB)               │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Phase 2: Local Processing (Future)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MOBILE APP                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐    │
│  │   Camera    │──▶│   Video     │──▶│    Disc Selector        │    │
│  │   Module    │   │   Storage   │   │                         │    │
│  └─────────────┘   └─────────────┘   └───────────┬─────────────┘    │
│                                                   │                  │
│                              ┌────────────────────▼────────────────┐ │
│                              │        LOCAL AI ENGINE              │ │
│                              │   ┌─────────────────────────────┐   │ │
│                              │   │  YOLO (CoreML/TFLite)       │   │ │
│                              │   │  Tracking (Native)          │   │ │
│                              │   │  Metrics (Native)           │   │ │
│                              │   └─────────────────────────────┘   │ │
│                              └────────────────────┬────────────────┘ │
│                                                   │                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Results Display (same as Phase 1)         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ Optional: Sync history to cloud
                                ▼
```

---

## 3. Technology Stack

### 3.1 Recommended: Kotlin Multiplatform (KMP)

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Framework** | Kotlin Multiplatform | Share business logic across platforms |
| **UI (Android)** | Jetpack Compose | Modern, declarative, native performance |
| **UI (iOS)** | SwiftUI (later) | Native iOS experience |
| **Networking** | Ktor | Cross-platform HTTP client |
| **Serialization** | kotlinx.serialization | Type-safe JSON |
| **Video Capture** | CameraX (Android) | Modern camera API |
| **Video Playback** | ExoPlayer (Android) | Robust media handling |
| **Charts** | Vico or MPAndroidChart | Native charting |
| **Local AI (Future)** | TensorFlow Lite | On-device inference |

### 3.2 Alternative: React Native

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Framework** | React Native | Single codebase, faster development |
| **UI** | React Native Paper / NativeBase | Material Design components |
| **Networking** | Axios / Fetch | Standard HTTP |
| **Video** | react-native-camera | Camera access |
| **Charts** | react-native-charts-wrapper | Native charts |
| **Local AI (Future)** | react-native-tensorflow-lite | On-device inference |

### 3.3 Decision: **Kotlin Multiplatform**

Reasons:
1. **Performance:** Native compilation, better for video/AI
2. **Future-proof:** Easier to integrate TFLite/CoreML
3. **Type safety:** Catch errors at compile time
4. **Growing ecosystem:** Google-backed, enterprise adoption

---

## 4. Project Structure

```
lifting-app/
├── README.md                           # Project overview
├── build.gradle.kts                    # Root build configuration
├── settings.gradle.kts                 # Module settings
├── gradle.properties                   # Gradle properties
│
├── shared/                             # Shared Kotlin code (KMP)
│   ├── build.gradle.kts
│   └── src/
│       ├── commonMain/kotlin/
│       │   ├── domain/                 # Domain models
│       │   │   ├── models/
│       │   │   │   ├── DiscSelection.kt
│       │   │   │   ├── Track.kt
│       │   │   │   ├── Metrics.kt
│       │   │   │   └── AnalysisResult.kt
│       │   │   └── repository/
│       │   │       └── AnalysisRepository.kt
│       │   │
│       │   ├── data/                   # Data layer
│       │   │   ├── api/
│       │   │   │   ├── LiftingApiClient.kt
│       │   │   │   ├── ApiModels.kt
│       │   │   │   └── ApiConfig.kt
│       │   │   └── local/
│       │   │       └── HistoryStorage.kt
│       │   │
│       │   └── util/                   # Utilities
│       │       ├── MetricsCalculator.kt
│       │       └── TrajectoryRenderer.kt
│       │
│       ├── androidMain/kotlin/         # Android-specific
│       │   └── platform/
│       │       └── AndroidPlatform.kt
│       │
│       └── iosMain/kotlin/             # iOS-specific (future)
│           └── platform/
│               └── IosPlatform.kt
│
├── androidApp/                         # Android application
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── kotlin/com/lifting/app/
│       │   ├── MainActivity.kt
│       │   ├── LiftingApp.kt           # App entry point
│       │   │
│       │   ├── ui/
│       │   │   ├── theme/
│       │   │   │   ├── Theme.kt
│       │   │   │   ├── Color.kt
│       │   │   │   └── Typography.kt
│       │   │   │
│       │   │   ├── screens/
│       │   │   │   ├── HomeScreen.kt
│       │   │   │   ├── CaptureScreen.kt
│       │   │   │   ├── SelectionScreen.kt
│       │   │   │   ├── ProcessingScreen.kt
│       │   │   │   ├── ResultsScreen.kt
│       │   │   │   └── HistoryScreen.kt
│       │   │   │
│       │   │   ├── components/
│       │   │   │   ├── VideoPlayer.kt
│       │   │   │   ├── DiscSelector.kt
│       │   │   │   ├── TrajectoryOverlay.kt
│       │   │   │   ├── MetricChart.kt
│       │   │   │   └── LoadingIndicator.kt
│       │   │   │
│       │   │   └── navigation/
│       │   │       └── NavGraph.kt
│       │   │
│       │   ├── camera/
│       │   │   ├── CameraManager.kt
│       │   │   └── VideoRecorder.kt
│       │   │
│       │   └── viewmodel/
│       │       ├── CaptureViewModel.kt
│       │       ├── SelectionViewModel.kt
│       │       ├── ProcessingViewModel.kt
│       │       └── ResultsViewModel.kt
│       │
│       └── res/
│           ├── values/
│           │   ├── strings.xml
│           │   └── colors.xml
│           └── drawable/
│
├── iosApp/                             # iOS application (future)
│   └── ...
│
└── docs/
    ├── API_CONTRACT.md
    ├── UI_MOCKUPS.md
    └── TESTING.md
```

---

## 5. Core Modules

### 5.1 Camera Module

**Responsibility:** Capture video from device camera.

```kotlin
// CameraManager.kt
interface CameraManager {
    suspend fun startPreview(surfaceProvider: Preview.SurfaceProvider)
    suspend fun startRecording(outputFile: File): Flow<RecordingStatus>
    suspend fun stopRecording(): File
    fun switchCamera()
    fun release()
}

enum class RecordingStatus {
    STARTING, RECORDING, STOPPING, COMPLETED, ERROR
}
```

**Requirements:**
- Support front and back camera
- Record at minimum 720p, 30fps
- Save as MP4 (H.264)
- Show recording duration
- Maximum recording time: 60 seconds (configurable)

### 5.2 Disc Selector Module

**Responsibility:** Allow user to select disc center and radius on first frame.

```kotlin
// DiscSelector.kt
@Composable
fun DiscSelector(
    frame: Bitmap,
    onSelectionComplete: (DiscSelection) -> Unit,
    onCancel: () -> Unit
)

data class DiscSelection(
    val centerX: Float,      // Pixels from left
    val centerY: Float,      // Pixels from top
    val radius: Float,       // Pixels
    val frameWidth: Int,     // Original frame dimensions
    val frameHeight: Int
)
```

**UI Flow:**
1. Display first frame of video (full screen)
2. User taps to set center (show crosshair)
3. User drags to set radius (show circle preview)
4. Buttons: "Confirm" / "Reset" / "Cancel"
5. Show coordinate values as user selects

### 5.3 API Client Module

**Responsibility:** Communicate with remote server.

```kotlin
// LiftingApiClient.kt
interface LiftingApiClient {
    suspend fun uploadVideo(
        videoFile: File,
        discSelection: DiscSelection?
    ): UploadResponse
    
    suspend fun getStatus(videoId: String): StatusResponse
    
    suspend fun getResults(videoId: String): AnalysisResult
    
    suspend fun deleteVideo(videoId: String): Boolean
}

// Implementation with Ktor
class KtorLiftingApiClient(
    private val baseUrl: String,
    private val httpClient: HttpClient
) : LiftingApiClient {
    
    override suspend fun uploadVideo(
        videoFile: File,
        discSelection: DiscSelection?
    ): UploadResponse {
        return httpClient.submitFormWithBinaryData(
            url = "$baseUrl/api/v1/videos/upload",
            formData = formData {
                append("file", videoFile.readBytes(), Headers.build {
                    append(HttpHeaders.ContentType, "video/mp4")
                    append(HttpHeaders.ContentDisposition, "filename=\"${videoFile.name}\"")
                })
                discSelection?.let {
                    append("disc_center_x", it.centerX.toString())
                    append("disc_center_y", it.centerY.toString())
                    append("disc_radius", it.radius.toString())
                }
            }
        ) {
            // Required header for Ngrok
            header("ngrok-skip-browser-warning", "true")
        }.body()
    }
    
    // ... other methods
}
```

### 5.4 Results Display Module

**Responsibility:** Visualize analysis results.

**Components:**

1. **Video Player with Overlay**
   - Play video at normal/slow speed
   - Draw track trajectories on video
   - Draw masks/bounding boxes (optional)
   - Sync playhead with graphs

2. **Metric Graphs**
   - Height vs Time
   - Velocity vs Time
   - Power vs Time
   - X-Y Trajectory

3. **Summary Cards**
   - Peak speed
   - Peak power
   - Max height
   - Total reps (future)

---

## 6. API Contract

### 6.1 Base URL

```
# Development (local network)
http://192.168.x.x:8000

# Remote access (Ngrok)
https://{random-id}.ngrok-free.app
```

### 6.2 Required Headers

```
ngrok-skip-browser-warning: true
Content-Type: multipart/form-data  (for upload)
Content-Type: application/json      (for other requests)
```

### 6.3 Endpoints

#### Upload Video

```http
POST /api/v1/videos/upload
Content-Type: multipart/form-data

Form Fields:
- file: Video file (required)
- disc_center_x: Float (optional, pixels)
- disc_center_y: Float (optional, pixels)  
- disc_radius: Float (optional, pixels)

Response 200:
{
  "video_id": "abc123def",
  "status": "pending",
  "message": "Video uploaded successfully..."
}
```

#### Check Status

```http
GET /api/v1/videos/{video_id}/status

Response 200:
{
  "video_id": "abc123def",
  "status": "processing",  // pending | processing | completed | failed
  "progress": 0.45,
  "current_step": "yolo_detection",
  "message": "Detecting objects...",
  "created_at": "2026-01-31T12:00:00",
  "updated_at": "2026-01-31T12:00:15"
}
```

#### Get Results

```http
GET /api/v1/videos/{video_id}/results

Response 200:
{
  "video_id": "abc123def",
  "status": "completed",
  "metadata": {
    "fps": 29.97,
    "width": 1080,
    "height": 1920,
    "duration_s": 4.5,
    "total_frames": 135
  },
  "tracks": [
    {
      "track_id": 1,
      "class_name": "frisbee",  // disc
      "frames": {
        "0": {"bbox": {"x1": 100, "y1": 200, "x2": 180, "y2": 280}, "confidence": 0.95},
        "1": {...},
        ...
      },
      "trajectory": [[140, 240], [142, 238], ...]
    },
    {
      "track_id": 2,
      "class_name": "person",
      ...
    }
  ],
  "metrics": {
    "frames": [0, 1, 2, ...],
    "time_s": [0.0, 0.033, 0.067, ...],
    "height_m": [0.5, 0.52, 0.55, ...],
    "speed_m_s": [0.0, 0.8, 1.2, ...],
    "power_w": [0, 150, 320, ...]
  },
  "summary": {
    "peak_speed_m_s": 2.45,
    "peak_power_w": 1850,
    "max_height_m": 0.82,
    "min_height_m": -0.15
  }
}
```

#### Delete Video

```http
DELETE /api/v1/videos/{video_id}

Response 200:
{
  "video_id": "abc123def",
  "deleted": true,
  "message": "Video and all associated data have been deleted."
}
```

### 6.4 Error Responses

```json
{
  "detail": "Error description here"
}
```

Status codes: 400, 404, 413, 500

---

## 7. UI/UX Specification

### 7.1 Screen Flow

```
┌──────────┐     ┌───────────┐     ┌────────────┐     ┌─────────────┐     ┌───────────┐
│  Home    │ ──▶ │  Camera   │ ──▶ │  Selection │ ──▶ │ Processing  │ ──▶ │  Results  │
│  Screen  │     │  Capture  │     │  (Disc)    │     │  (Upload)   │     │  Display  │
└──────────┘     └───────────┘     └────────────┘     └─────────────┘     └───────────┘
     │                                                                           │
     │                                                                           │
     └───────────────────────────────────────────────────────────────────────────┘
                                      History
```

### 7.2 Home Screen

**Layout:**
- App logo/title
- "New Recording" button (primary action)
- "History" button
- Settings icon

### 7.3 Camera Capture Screen

**Layout:**
- Full-screen camera preview
- Recording button (bottom center)
- Timer display (during recording)
- Front/back camera switch
- Cancel button

**Interactions:**
- Tap record to start
- Tap again to stop
- Auto-stop at 60 seconds
- Vibrate feedback on start/stop

### 7.4 Disc Selection Screen

**Layout:**
- Full-screen first frame
- Instruction text: "Tap on the disc center, then drag to set radius"
- Coordinate display (center X, Y, radius)
- "Confirm" button (enabled after valid selection)
- "Reset" button
- "Skip" button (process without selection)

**Interactions:**
- First tap: Set center (red crosshair)
- Second tap/drag: Set radius (blue circle preview)
- Pinch to zoom frame (optional)

### 7.5 Processing Screen

**Layout:**
- Progress bar with percentage
- Current step indicator
- Cancel button
- Estimated time remaining (optional)

**States:**
- Uploading video...
- Processing: Detecting objects...
- Processing: Tracking...
- Processing: Calculating metrics...
- Complete!

### 7.6 Results Screen

**Layout:**
- 3-panel structure (similar to desktop viewer)
- Top: Video player with playback controls
- Bottom left: Selectable metric graph
- Bottom right: X-Y trajectory

**Controls:**
- Play/Pause
- Slow motion (0.25x)
- Frame scrubber
- Graph selector dropdown
- Export/Share button (future)

---

## 8. Data Models

### 8.1 Shared Models (Kotlin)

```kotlin
// DiscSelection.kt
@Serializable
data class DiscSelection(
    val centerX: Float,
    val centerY: Float,
    val radius: Float,
    val frameWidth: Int,
    val frameHeight: Int
)

// Track.kt
@Serializable
data class Track(
    val trackId: Int,
    val className: String,
    val frames: Map<Int, FrameDetection>,
    val trajectory: List<List<Float>>
)

@Serializable
data class FrameDetection(
    val bbox: BoundingBox,
    val confidence: Float,
    val mask: List<List<Float>>? = null
)

@Serializable
data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

// Metrics.kt
@Serializable
data class MetricsSeries(
    val frames: List<Int>,
    @SerialName("time_s") val timeS: List<Float>,
    @SerialName("height_m") val heightM: List<Float>,
    @SerialName("speed_m_s") val speedMS: List<Float>,
    @SerialName("power_w") val powerW: List<Float>
)

@Serializable
data class MetricsSummary(
    @SerialName("peak_speed_m_s") val peakSpeedMS: Float,
    @SerialName("peak_power_w") val peakPowerW: Float,
    @SerialName("max_height_m") val maxHeightM: Float,
    @SerialName("min_height_m") val minHeightM: Float
)

// AnalysisResult.kt
@Serializable
data class AnalysisResult(
    val videoId: String,
    val status: String,
    val metadata: VideoMetadata,
    val tracks: List<Track>,
    val metrics: MetricsSeries,
    val summary: MetricsSummary
)

@Serializable
data class VideoMetadata(
    val fps: Float,
    val width: Int,
    val height: Int,
    @SerialName("duration_s") val durationS: Float,
    @SerialName("total_frames") val totalFrames: Int
)
```

---

## 9. Development Phases

### Phase 1: MVP Foundation (2-3 weeks)

- [ ] Project setup (Gradle, dependencies)
- [ ] Basic navigation structure
- [ ] Camera capture module
- [ ] Video storage management
- [ ] API client with upload
- [ ] Processing screen with status polling

### Phase 2: Disc Selection (1 week)

- [ ] First frame extraction
- [ ] Touch-based center selection
- [ ] Radius selection with preview
- [ ] Selection data passing to API

### Phase 3: Results Display (2 weeks)

- [ ] Video player with controls
- [ ] Trajectory overlay rendering
- [ ] Metric graph components
- [ ] Graph-video synchronization

### Phase 4: Polish & History (1 week)

- [ ] History screen (past analyses)
- [ ] Settings screen
- [ ] Error handling improvements
- [ ] UI polish and animations

### Phase 5: Local Inference (Future)

- [ ] TFLite model integration
- [ ] Native tracking implementation
- [ ] Metrics calculation on device
- [ ] Offline mode

---

## 10. Testing Strategy

### 10.1 Unit Tests

- ViewModel logic
- Data transformations
- API response parsing
- Metrics calculations

### 10.2 Integration Tests

- API client with mock server
- Database operations
- File system operations

### 10.3 UI Tests

- Screen navigation
- User interactions
- State transitions

### 10.4 Manual Testing Checklist

- [ ] Record video in different lighting
- [ ] Test with various video lengths
- [ ] Verify disc selection accuracy
- [ ] Check results display on different screen sizes
- [ ] Test offline behavior
- [ ] Verify memory usage during video playback

---

## 11. Configuration & Environment

### 11.1 Build Configuration

```kotlin
// androidApp/build.gradle.kts
android {
    namespace = "com.lifting.app"
    compileSdk = 34
    
    defaultConfig {
        applicationId = "com.lifting.app"
        minSdk = 26  // Android 8.0+
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
        
        // Server URL (can be changed via build variants)
        buildConfigField("String", "API_BASE_URL", "\"http://localhost:8000\"")
    }
    
    buildTypes {
        debug {
            // Use local server or Ngrok
            buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8000\"")
        }
        release {
            buildConfigField("String", "API_BASE_URL", "\"https://your-production-url.com\"")
        }
    }
}
```

### 11.2 Required Permissions

```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" 
                 android:maxSdkVersion="28"/>
<uses-feature android:name="android.hardware.camera" android:required="true" />
```

### 11.3 Environment Variables

```
# .env (for development)
API_BASE_URL=http://192.168.1.100:8000
NGROK_URL=https://xyz.ngrok-free.app
DEBUG_MODE=true
MAX_VIDEO_DURATION_SEC=60
```

---

## 12. Deployment Guide

### 12.1 Development Setup

```bash
# Clone repository
git clone git@github.com:your-org/lifting-app.git
cd lifting-app

# Open in Android Studio
# - File > Open > Select project root
# - Wait for Gradle sync

# Connect device or start emulator
# - Enable USB debugging on physical device
# - Or use Android Studio emulator

# Run app
# - Select device in toolbar
# - Click Run (green arrow)
```

### 12.2 Server Configuration

Before running the app, ensure the server is ready:

```bash
# On server machine
cd app-mvp/ai-core
PYTHONPATH=src:. uv run python control_panel.py

# In Control Panel web UI:
# 1. Start FastAPI server
# 2. Start Ngrok (if needed for remote access)
# 3. Copy the public URL to app configuration
```

### 12.3 Testing Connection

```bash
# Test from device
curl -X GET "http://{server-ip}:8000/health" \
  -H "ngrok-skip-browser-warning: true"

# Expected response
{"status": "healthy", "timestamp": "..."}
```

### 12.4 Release Build

```bash
# Generate signed APK
./gradlew assembleRelease

# Output: androidApp/build/outputs/apk/release/androidApp-release.apk
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Track** | A detected object followed across multiple frames |
| **Disc** | The weighted plate on a barbell |
| **Trajectory** | The path of an object through space over time |
| **Mask** | Pixel-level segmentation of an object |
| **BBox** | Bounding box - rectangular region around an object |
| **Metrics** | Calculated physical quantities (velocity, power, etc.) |

---

## Appendix B: Reference Links

- [Kotlin Multiplatform Documentation](https://kotlinlang.org/docs/multiplatform.html)
- [Jetpack Compose](https://developer.android.com/jetpack/compose)
- [CameraX Documentation](https://developer.android.com/training/camerax)
- [Ktor HTTP Client](https://ktor.io/docs/client.html)
- [ExoPlayer](https://exoplayer.dev/)
- [TensorFlow Lite Android](https://www.tensorflow.org/lite/android)

---

*End of Mobile App Specification*
