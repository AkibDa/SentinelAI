# backend/app/services/inference.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.core.config import settings

# Load model at startup
model = load_model(settings.MODEL_PATH)

def get_verdict_and_confidence(score: float):
    """Calculates verdict and confidence based on how the model was mapped."""
    if settings.FAKE_IS_LOW_SCORE:
        is_fake = score < settings.THRESHOLD
        confidence = (1.0 - score) if is_fake else score
    else:
        is_fake = score > settings.THRESHOLD
        confidence = score if is_fake else (1.0 - score)

    return "Fake" if is_fake else "Real", round(confidence * 100, 2)


def preprocess_to_match_training(img: np.ndarray) -> np.ndarray:
    """
    Exactly replicates Keras ImageDataGenerator preprocessing:
    1. Resize to (224, 224)
    2. Convert to float32
    3. Apply EfficientNet specific scaling
    """
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img, dtype=np.float32)
    processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return processed_img


def analyze_image_for_deepfakes(img: np.ndarray):
    if img is None:
        raise ValueError("Could not read image file")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = preprocess_to_match_training(rgb_img)
    processed = np.expand_dims(processed, axis=0)

    score = float(model.predict(processed, verbose=0)[0][0])
    verdict, confidence = get_verdict_and_confidence(score)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_score": round(score, 4)
    }


def analyze_video_for_deepfakes(video_path: str, target_fps: int = 2):
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps == 0:
        actual_fps = 30

    frame_interval = max(1, int(actual_fps / target_fps))
    batch_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = preprocess_to_match_training(rgb)
            batch_frames.append(processed)

        frame_count += 1

    cap.release()

    if not batch_frames:
        return {
            "verdict": "Unable to analyze",
            "confidence": 0,
            "frames_analyzed": 0
        }

    batch_array = np.array(batch_frames)
    predictions = model.predict(batch_array, verbose=0).flatten()
    avg_score = float(np.mean(predictions))

    if settings.FAKE_IS_LOW_SCORE:
        suspicious_frames = sum(1 for p in predictions if p < settings.THRESHOLD)
    else:
        suspicious_frames = sum(1 for p in predictions if p > settings.THRESHOLD)

    verdict, confidence = get_verdict_and_confidence(avg_score)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_score": round(avg_score, 4),
        "frames_analyzed": len(predictions),
        "suspicious_frames": suspicious_frames
    }