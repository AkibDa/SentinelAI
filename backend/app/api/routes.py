from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os
import cv2
import numpy as np
import asyncio
import yt_dlp

from app.models.schemas import URLRequest
from app.services.downloader import download_video_with_ytdlp
from app.services.inference import analyze_video_for_deepfakes, analyze_image_for_deepfakes

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "success", "message": "Sentinel Visual Detector API is running."}

@router.post("/analyze_url")
async def analyze_url(request: URLRequest):
    temp_dir = tempfile.mkdtemp()
    video_path = None

    try:
        video_path = await asyncio.to_thread(download_video_with_ytdlp, request.url, temp_dir)
        result = await asyncio.to_thread(analyze_video_for_deepfakes, video_path)
        result["source_url"] = request.url
        return JSONResponse(content=result)

    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

@router.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    temp_video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_video_path = tmp.name

        result = await asyncio.to_thread(analyze_video_for_deepfakes, temp_video_path)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@router.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image data.")

        result = await asyncio.to_thread(analyze_image_for_deepfakes, img)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")