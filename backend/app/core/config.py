# backend/app/core/config.py

class Settings:
  PROJECT_NAME: str = "Sentinel Visual Detector API"
  DESCRIPTION: str = "An API to detect deepfakes in uploaded videos, images, and URLs."
  VERSION: str = "1.2.0"

  # Model configuration
  MODEL_PATH: str = "/Users/skakibahammed/code_playground/SentinelAI/backend/sentinel_visual_detector.keras"
  THRESHOLD: float = 0.50
  FAKE_IS_LOW_SCORE: bool = True

settings = Settings()
