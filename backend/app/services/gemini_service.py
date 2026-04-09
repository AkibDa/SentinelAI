# backend/app/services/gemini_service.py

import os
import io
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

my_api_key = os.getenv("GOOGLE_API_KEY")

if not my_api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

genai.configure(api_key=my_api_key)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')


def analyze_with_gemini(image_bytes: bytes, local_result: dict) -> str:
  """
  Sends the image and the local model's findings to Gemini for deep contextual analysis.
  """
  try:
    pil_img = Image.open(io.BytesIO(image_bytes))

    visual_verdict = local_result["visual_analysis"]["verdict"]
    visual_confidence = local_result["visual_analysis"]["confidence"]
    metadata_risk = local_result["metadata_analysis"]["risk_level"]
    flags = ", ".join(local_result["metadata_analysis"]["flags"]) if local_result["metadata_analysis"][
      "flags"] else "None"

    system_prompt = f"""
        You are an expert digital forensics AI. The user has uploaded an image for deepfake analysis.

        Our internal primary system has already flagged this file with the following technical data:
        - Local Visual Model Verdict: {visual_verdict} (Confidence: {visual_confidence}%)
        - EXIF Metadata Risk Level: {metadata_risk}
        - Detected Metadata Flags: {flags}

        Your task: Analyze the attached image visually. Look for common AI generation artifacts (e.g., inconsistent lighting, asymmetrical features, structural errors in hands/teeth, background blending issues, or hyper-smooth textures). 

        Provide a concise, 3-4 sentence forensic report explaining WHY this image looks authentic or AI-generated. Do not use markdown formatting, just return plain text.
        """

    response = gemini_model.generate_content([pil_img, system_prompt])
    return response.text.strip()

  except Exception as e:
    return f"Deep contextual analysis temporarily unavailable: {str(e)}"