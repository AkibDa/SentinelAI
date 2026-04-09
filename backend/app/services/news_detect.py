# backend/app/services/news_detect.py

import os
import asyncio
from transformers import pipeline
import httpx
from typing import List

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
if not GNEWS_API_KEY:
    raise ValueError("GNEWS_API_KEY is missing from environment variables")

nli_model = pipeline(
    "text-classification",
    model="MoritzLaurer/deberta-v3-base-mnli",
    top_k=None
)

async def retrieve_evidence(claim: str, top_k: int = 3) -> List[str]:
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": claim,
        "lang": "en",
        "max": top_k,
        "apikey": GNEWS_API_KEY
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()

    articles = response.json().get("articles", [])
    evidence = []

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        source = article.get("source", {}).get("name", "Unknown source")
        snippet = f"[{source}] {title}. {description}".strip()
        evidence.append(snippet)

    return evidence

def run_nli_inference(claim: str, evidence_text: str):
    """Synchronous function to wrap the CPU-bound ML model."""
    return nli_model({
        "text": evidence_text,
        "text_pair": claim
    }, truncation=True)

async def verify_claim_against_evidence(claim: str, evidence_list: List[str]):
    if not evidence_list:
        return {
            "verdict": "Insufficient Evidence",
            "confidence": 0.0,
            "reasoning": "No relevant news articles found to verify this claim."
        }

    all_scores = []

    for evidence_text in evidence_list:
      result = await asyncio.to_thread(run_nli_inference, claim, evidence_text)
      parsed = result if isinstance(result, list) else [result]

      for item in parsed:
        all_scores.append({
          "label": item["label"].lower(),
          "score": item["score"]
        })

    results = await asyncio.to_thread(run_nli_inference, claim, evidence_text)

    parsed_results = results if isinstance(results, list) else [results]

    scores = {}
    for item in parsed_results:
      if isinstance(item, dict):
        scores[item["label"].lower()] = item["score"]

    entailment = max([x["score"] for x in all_scores if x["label"] == "entailment"], default=0)
    contradiction = max([x["score"] for x in all_scores if x["label"] == "contradiction"], default=0)
    neutral = max([x["score"] for x in all_scores if x["label"] == "neutral"], default=0)

    if contradiction > max(entailment, neutral):
        return {
            "verdict": "Likely Fake",
            "confidence": round(contradiction * 100, 2),
            "reasoning": "Live trusted news evidence contradicts the submitted claim."
        }
    elif entailment > max(contradiction, neutral):
        return {
            "verdict": "Likely Real",
            "confidence": round(entailment * 100, 2),
            "reasoning": "Live trusted news evidence supports the submitted claim."
        }
    else:
        return {
            "verdict": "Inconclusive",
            "confidence": round(neutral * 100, 2),
            "reasoning": "Live evidence is inconclusive for this claim."
        }