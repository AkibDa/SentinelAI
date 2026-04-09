# 🚀 Sentinel Visual Detector

An AI-powered deepfake detection system that analyzes images and videos to determine whether they are **Real or Fake** using a trained deep learning model.

---

## 🧠 What This Project Does

This system detects deepfakes by analyzing:

- 🖼️ Images
- 🎥 Uploaded videos
- 🔗 Videos from URLs (YouTube, etc.)

It uses a trained **EfficientNet-based Keras model** to classify media as:

- ✅ Real
- ❌ Fake

---

## ⚙️ How It Works

### 🔄 Processing Flow

1. User sends an image, video, or URL
2. API receives request via FastAPI routes
3. Media is preprocessed (resized to 224×224, normalized)
4. Frames (for video) are extracted and sampled
5. Model predicts scores for each frame/image
6. Final verdict is generated using threshold logic

---

## 🏗️ Architecture

- **API Layer** → FastAPI (`routes.py`)
- **Service Layer** → Inference & Downloader
- **Model Layer** → TensorFlow/Keras model

---

## 🤖 Model Details

- Framework: TensorFlow / Keras
- Architecture: EfficientNet (preprocessing used)
- Input Size: 224 × 224
- Output: Probability score (0–1)

### 🔍 Prediction Logic

- Score compared with threshold
- Configurable mapping:
  - Fake if score > threshold OR < threshold (based on config)
- Confidence calculated dynamically

---

## 📡 API Endpoints

### ✅ Health Check

GET /

Response:
{
  "status": "success",
  "message": "Sentinel Visual Detector API is running."
}

---

### Analyze Video from URL
POST /analyze_url

Description:
Downloads a video from a given URL and analyzes it for deepfakes.

Request:
{
  "url": "https://example.com/video"
}

Response:
{
  "verdict": "Fake",
  "confidence": 87.45,
  "raw_score": 0.91,
  "frames_analyzed": 120,
  "suspicious_frames": 85,
  "source_url": "https://example.com/video"
}

Errors:
- 400 → Failed to download video
- 500 → Processing error

---

### Upload Video
POST /analyze_video

Description:
Uploads a video file and analyzes it for deepfake detection.

Request:
- Content-Type: multipart/form-data
- Field: file (video)

Response:
{
  "verdict": "Real",
  "confidence": 78.12,
  "raw_score": 0.21,
  "frames_analyzed": 95,
  "suspicious_frames": 10
}

Errors:
- 400 → Invalid file type
- 500 → Processing error

---

### Analyze Image
POST /analyze_image

Description:
Uploads an image and detects whether it is fake or real.

Request:
- Content-Type: multipart/form-data
- Field: file (image)

Response:
{
  "verdict": "Fake",
  "confidence": 92.33,
  "raw_score": 0.95
}

Errors:
- 400 → Invalid image
- 500 → Processing error

Interactive API docs: http://localhost:8080/docs
