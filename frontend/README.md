# 🎨 Sentinel Visual Detector – Frontend

A modern and interactive frontend interface for the Sentinel Visual Detector system, allowing users to easily upload and analyze images and videos for deepfake detection.

---

## 🧠 What This Frontend Does

This UI provides a smooth and user-friendly way to interact with the deepfake detection backend.

Users can:

- 🖼️ Upload images for deepfake detection  
- 🎥 Upload videos for analysis  
- 🔗 Paste video URLs for direct processing  
- 📊 View results with verdict and confidence score  

---

## ⚙️ Tech Stack

- ⚛️ React (with Vite)
- 🎨 Tailwind CSS
- 🎞️ Framer Motion (animations)
- 🧠 Zustand (state management)
- 🌌 Three.js / React Three Fiber (3D UI elements)
- ✨ Spline (interactive 3D scenes)
- 🎇 tsparticles (background effects)

---

## 🎯 Features

- 🚀 Fast and responsive UI  
- 🎥 Multi-input support (image, video, URL)  
- 📊 Real-time result display  
- 🎨 Modern animated interface  
- 🌌 Interactive 3D visuals  
- ⚡ Smooth transitions using Framer Motion  

---

## 🔗 Backend Integration
The frontend communicates with the backend API to perform deepfake analysis on images and videos.

**Base URL**
http://localhost:8080

**Supported Endpoints**

1. Analyze Image  
   POST /analyze_image  
   - Upload an image file (multipart/form-data)  
   - Returns verdict and confidence score  

2. Analyze Video  
   POST /analyze_video  
   - Upload a video file (multipart/form-data)  
   - Returns verdict, confidence, frames analyzed, and suspicious frames  

3. Analyze Video from URL  
   POST /analyze_url  
   - Send a JSON body with video URL  
   - Example:
     {
       "url": "https://example.com/video"
     }  
   - Downloads and analyzes the video, returning detailed results  

All responses include a clear classification (Real/Fake) along with confidence metrics.

This frontend communicates with the backend API for processing.

**Expected Backend URL:**
http://localhost:8080 (for local development)

