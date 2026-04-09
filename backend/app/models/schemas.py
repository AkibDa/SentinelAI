# backend/models/schemas.py

from pydantic import BaseModel, Field
from typing import List

class URLRequest(BaseModel):
    url: str

class VerifyNewsRequest(BaseModel):
  headline: str = Field(..., min_length=5)

class VerifyNewsResponse(BaseModel):
  claim: str
  verdict: str
  confidence: float
  evidence: List[str]
  reasoning: str