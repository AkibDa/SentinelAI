# backend/app/services/news_detect.py

import os
import asyncio
import httpx
from typing import List
import time

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

_nli_model = None
_evidence_cache: dict[str, tuple[float, List[str]]] = {}
_verification_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_SEC = 10 * 60  # 10 minutes


def _now() -> float:
    return time.monotonic()


def get_nli_model():
    """
    Lazy-load the NLI model to avoid heavy import-time startup cost.
    This improves perceived latency (especially in dev reloads) and makes failures clearer.
    """
    global _nli_model
    if _nli_model is None:
        from transformers import pipeline  # local import to reduce import-time overhead

        _nli_model = pipeline(
            "text-classification",
            model="MoritzLaurer/deberta-v3-base-mnli",
            top_k=None,
        )
    return _nli_model

async def retrieve_evidence(claim: str, top_k: int = 3) -> List[str]:
    # If the API key isn't configured, degrade gracefully.
    if not GNEWS_API_KEY:
        return []

    cache_key = claim.strip().lower()
    cached = _evidence_cache.get(cache_key)
    if cached and (_now() - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": claim,
        "lang": "en",
        "max": top_k,
        "apikey": GNEWS_API_KEY
    }

    try:
        timeout = httpx.Timeout(connect=2.5, read=6.0, write=3.0, pool=2.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
    except Exception:
        # Network/API failures shouldn't crash the endpoint.
        return []

    articles = response.json().get("articles", [])
    evidence = []

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        source = article.get("source", {}).get("name", "Unknown source")
        snippet = f"[{source}] {title}. {description}".strip()
        evidence.append(snippet)

    _evidence_cache[cache_key] = (_now(), evidence)
    return evidence

def run_nli_inference(claim: str, evidence_text: str):
    """Synchronous function to wrap the CPU-bound ML model."""
    model = get_nli_model()
    return model({
        "text": evidence_text,
        "text_pair": claim
    }, truncation=True)

async def verify_claim_against_evidence(claim: str, evidence_list: List[str]):
    cache_key = claim.strip().lower()
    cached = _verification_cache.get(cache_key)
    if cached and (_now() - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]

    if not evidence_list:
        result = {
            "verdict": "Insufficient Evidence",
            "confidence": 0.0,
            "reasoning": "No relevant news articles found to verify this claim."
        }
        _verification_cache[cache_key] = (_now(), result)
        return result

    try:
        # Run NLI across evidence concurrently to reduce wall-clock time.
        tasks = [asyncio.to_thread(run_nli_inference, claim, ev) for ev in evidence_list]
        per_evidence_results = await asyncio.gather(*tasks, return_exceptions=False)
    except Exception:
        result = {
            "verdict": "Inconclusive",
            "confidence": 0.0,
            "reasoning": "Verification model is temporarily unavailable. Please try again shortly."
        }
        _verification_cache[cache_key] = (_now(), result)
        return result

    all_scores = []
    for res in per_evidence_results:
        parsed = res if isinstance(res, list) else [res]
        for item in parsed:
            if isinstance(item, dict) and "label" in item and "score" in item:
                all_scores.append({
                    "label": str(item["label"]).lower(),
                    "score": float(item["score"]),
                })

    entailment = max([x["score"] for x in all_scores if x["label"] == "entailment"], default=0)
    contradiction = max([x["score"] for x in all_scores if x["label"] == "contradiction"], default=0)
    neutral = max([x["score"] for x in all_scores if x["label"] == "neutral"], default=0)

    if contradiction > max(entailment, neutral):
        result = {
            "verdict": "Likely Fake",
            "confidence": round(contradiction * 100, 2),
            "reasoning": "Live trusted news evidence contradicts the submitted claim."
        }
    elif entailment > max(contradiction, neutral):
        result = {
            "verdict": "Likely Real",
            "confidence": round(entailment * 100, 2),
            "reasoning": "Live trusted news evidence supports the submitted claim."
        }
    else:
        result = {
            "verdict": "Inconclusive",
            "confidence": round(neutral * 100, 2),
            "reasoning": "Live evidence is inconclusive for this claim."
        }

    _verification_cache[cache_key] = (_now(), result)
    return result