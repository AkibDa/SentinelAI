from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os
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
    # 1. Prepare the raw bytes
    await file.seek(0)
    contents = await file.read()

    if not contents:
      raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # 2. STAGE 1: Fast Local Visual & Metadata Scan (from inference.py)
    local_result = await asyncio.to_thread(analyze_image_for_deepfakes, contents)

    # 3. Determine if Stage 2 is needed
    verdict = local_result["visual_analysis"]["verdict"]
    confidence = local_result["visual_analysis"]["confidence"]
    risk = local_result["metadata_analysis"]["risk_level"]

    needs_deep_analysis = (
        verdict == "Fake" or
        risk == "High" or
        (verdict == "Real" and confidence < 85.0)
    )

    if not needs_deep_analysis:
      local_result["gemini_analysis"] = "Bypassed. Image passed local security thresholds."
      return JSONResponse(content=local_result)

    # 4. STAGE 2: Deep Contextual Analysis (from gemini_service.py)
    gemini_text = await asyncio.to_thread(analyze_with_gemini, contents, local_result)
    local_result["gemini_analysis"] = gemini_text

    return JSONResponse(content=local_result)

  except ValueError as ve:
    raise HTTPException(status_code=400, detail=str(ve))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")