# backend/app/services/inference.py

import cv2
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pymediainfo import MediaInfo
from PIL import Image, ExifTags
from ..core.config import settings

model = None


def get_model():
  """Lazy-load the model so API startup does not fail when path is invalid."""
  global model
  if model is None:
    try:
      model = load_model(settings.MODEL_PATH)
    except Exception as e:
      raise RuntimeError(
        f"Model failed to load from MODEL_PATH='{settings.MODEL_PATH}'. "
        "Set a valid model path in backend/app/core/config.py."
      ) from e
  return model

SUSPICIOUS_VIDEO_ENCODERS = [
  "lavf", "ffmpeg", "after effects", "premiere",
  "davinci", "blender", "obs", "handbrake"
]

SUSPICIOUS_IMAGE_SOFTWARE = [
  "photoshop", "lightroom", "midjourney", "stable diffusion",
  "dall-e", "gimp", "canva", "topaz"
]


def analyze_video_metadata_risk(metadata: dict) -> dict:
  """Evaluates the extracted video metadata for deepfake/editing signatures."""
  flags = []
  risk_level = "Low"

  if "error" in metadata:
    return {"risk_level": "Unknown", "flags": ["Could not parse metadata."]}

  general_info = metadata.get("general", {})
  encoded_app = str(general_info.get("encoded_application", "")).lower()

  for suspect in SUSPICIOUS_VIDEO_ENCODERS:
    if suspect in encoded_app:
      flags.append(f"Video was processed using {suspect.title()} (Found: {general_info.get('encoded_application')})")
      risk_level = "High"
      break

  if not encoded_app and not metadata.get("video", {}).get("codec"):
    flags.append("Metadata is heavily stripped or missing standard container tags.")
    if risk_level != "High":
      risk_level = "Medium"

  return {"risk_level": risk_level, "flags": flags}


def analyze_image_metadata_risk(metadata: dict) -> dict:
  """Evaluates the extracted image EXIF metadata for editing/generative signatures."""
  flags = []
  risk_level = "Low"

  if "error" in metadata:
    return {"risk_level": "Unknown", "flags": ["Could not parse metadata."]}

  software = str(metadata.get("exif", {}).get("Software", "")).lower()

  for suspect in SUSPICIOUS_IMAGE_SOFTWARE:
    if suspect in software:
      flags.append(f"Image was processed using {suspect.title()} (Found: {metadata['exif']['Software']})")
      risk_level = "High"
      break

  if not metadata.get("exif"):
    flags.append(
      "EXIF metadata is completely stripped, which can indicate intentional obfuscation or social media compression.")
    if risk_level != "High":
      risk_level = "Medium"

  return {"risk_level": risk_level, "flags": flags}


def extract_video_metadata(video_path: str) -> dict:
  """Extracts container and track metadata from a video file."""
  try:
    media_info = MediaInfo.parse(video_path)
    metadata = {"general": {}, "video": {}, "audio": {}}

    for track in media_info.tracks:
      if track.track_type == "General":
        metadata["general"]["format"] = track.format
        metadata["general"]["file_size_bytes"] = track.file_size
        metadata["general"]["encoded_application"] = track.encoded_application
        metadata["general"]["encoded_date"] = track.encoded_date
      elif track.track_type == "Video":
        metadata["video"]["codec"] = track.codec_id
        metadata["video"]["width"] = track.width
        metadata["video"]["height"] = track.height
        metadata["video"]["frame_rate"] = track.frame_rate
      elif track.track_type == "Audio":
        metadata["audio"]["codec"] = track.format
        metadata["audio"]["sampling_rate"] = track.sampling_rate

    return metadata
  except Exception as e:
    return {"error": f"Could not extract metadata: {str(e)}"}


def extract_image_metadata(image_path: str) -> dict:
  """Extracts EXIF and basic file metadata from an image."""
  try:
    img = Image.open(image_path)
    metadata = {
      "format": img.format,
      "mode": img.mode,
      "size": img.size,
      "exif": {}
    }

    exif_data = img.getexif()
    if exif_data:
      for tag_id, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        if isinstance(value, bytes):
          continue
        metadata["exif"][tag_name] = str(value)

    return metadata
  except Exception as e:
    return {"error": f"Could not extract image metadata: {str(e)}"}


def get_verdict_and_confidence(score: float, custom_threshold: float = None):
  """Calculates verdict and scales confidence relative to a strictly high threshold."""
  thresh = custom_threshold if custom_threshold is not None else settings.THRESHOLD

  if settings.FAKE_IS_LOW_SCORE:
    is_fake = score < thresh
    if is_fake:
      confidence = 1.0 - (score / thresh)
    else:
      confidence = (score - thresh) / (1.0 - thresh)
  else:
    is_fake = score > thresh
    if is_fake:
      confidence = (score - thresh) / (1.0 - thresh)
    else:
      confidence = 1.0 - (score / thresh)

  return "Fake" if is_fake else "Real", round(confidence * 100, 2)


def preprocess_to_match_training(img: np.ndarray) -> np.ndarray:
  """
  Crops to maintain aspect ratio, resizes to 224x224, and applies EfficientNet preprocessing.
  """
  h, w = img.shape[:2]
  min_dim = min(h, w)
  start_x = w // 2 - min_dim // 2
  start_y = h // 2 - min_dim // 2
  cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

  resized_img = cv2.resize(cropped_img, (224, 224))
  img_array = np.array(resized_img, dtype=np.float32)
  return tf.keras.applications.efficientnet.preprocess_input(img_array)


def analyze_video_for_deepfakes(video_path: str, target_fps: int = 2, threshold: float = 0.85):
  raw_metadata = extract_video_metadata(video_path)
  metadata_risk = analyze_video_metadata_risk(raw_metadata)

  cap = cv2.VideoCapture(video_path)
  actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30

  frame_interval = max(1, int(actual_fps / target_fps))
  batch_frames, frame_timestamps = [], []
  frame_count = 0

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    if frame_count % frame_interval == 0:
      timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      batch_frames.append(preprocess_to_match_training(rgb))
      frame_timestamps.append(timestamp_sec)

    frame_count += 1
  cap.release()

  if not batch_frames:
    return {"verdict": "Unable to analyze", "confidence": 0}

  # 3. Model Predictions
  BATCH_SIZE = 32
  predictions = []

  for i in range(0, len(batch_frames), BATCH_SIZE):
    batch_array = np.array(batch_frames[i:i + BATCH_SIZE])
    batch_preds = get_model().predict(batch_array, verbose=0).flatten()
    predictions.extend(batch_preds)

  predictions = np.array(predictions)

  if settings.FAKE_IS_LOW_SCORE:
    agg_score = float(np.percentile(predictions, 15))
  else:
    agg_score = float(np.percentile(predictions, 85))

  suspicious_frame_details = []
  for i, score in enumerate(predictions):
    is_fake = (score < threshold) if settings.FAKE_IS_LOW_SCORE else (score > threshold)
    if is_fake:
      suspicious_frame_details.append({
        "timestamp_sec": round(frame_timestamps[i], 2),
        "score": round(float(score), 4)
      })

  visual_verdict, visual_confidence = get_verdict_and_confidence(agg_score, custom_threshold=threshold)

  system_warning = None
  if visual_verdict == "Real" and visual_confidence < 75.0 and metadata_risk["risk_level"] == "High":
    system_warning = "Visuals appear real, but high-risk metadata suggests the file was heavily edited or generated."

  return {
    "visual_analysis": {
      "verdict": visual_verdict,
      "confidence": visual_confidence,
      "aggregated_score": round(agg_score, 4),
      "frames_analyzed": len(predictions),
      "suspicious_frames_count": len(suspicious_frame_details),
      "suspicious_frame_details": suspicious_frame_details
    },
    "metadata_analysis": {
      "risk_level": metadata_risk["risk_level"],
      "flags": metadata_risk["flags"],
      "raw_data": raw_metadata
    },
    "system_warning": system_warning
  }

def extract_image_metadata_from_bytes(image_bytes: bytes) -> dict:
  """Extracts EXIF and basic file metadata from raw image bytes in memory."""
  try:
    img = Image.open(io.BytesIO(image_bytes))
    metadata = {
      "format": img.format,
      "mode": img.mode,
      "size": img.size,
      "exif": {}
    }

    exif_data = img.getexif()
    if exif_data:
      for tag_id, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        if isinstance(value, bytes):
          continue
        metadata["exif"][tag_name] = str(value)

    return metadata
  except Exception as e:
    return {"error": f"Could not extract image metadata: {str(e)}"}


def analyze_image_for_deepfakes(image_bytes: bytes, threshold: float = 0.85):
  raw_metadata = extract_image_metadata_from_bytes(image_bytes)
  metadata_risk = analyze_image_metadata_risk(raw_metadata)

  try:
    pil_img = Image.open(io.BytesIO(image_bytes))

    if pil_img.mode != "RGB":
      pil_img = pil_img.convert("RGB")

    rgb_img = np.array(pil_img)

  except Exception as e:
    raise ValueError(
      f"Could not decode image. The file might be corrupted or in an unsupported format. Error: {str(e)}")

  processed = preprocess_to_match_training(rgb_img)
  processed = np.expand_dims(processed, axis=0)

  score = float(get_model().predict(processed, verbose=0)[0][0])

  visual_verdict, visual_confidence = get_verdict_and_confidence(score, custom_threshold=threshold)

  system_warning = None
  if visual_verdict == "Real" and visual_confidence < 75.0 and metadata_risk["risk_level"] == "High":
    system_warning = "Visuals appear real, but high-risk metadata suggests the image was edited or AI-generated."

  return {
    "visual_analysis": {
      "verdict": visual_verdict,
      "confidence": visual_confidence,
      "raw_score": round(score, 4)
    },
    "metadata_analysis": {
      "risk_level": metadata_risk["risk_level"],
      "flags": metadata_risk["flags"],
      "raw_data": raw_metadata
    },
    "system_warning": system_warning
  }