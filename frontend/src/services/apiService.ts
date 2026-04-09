import { GoogleGenAI, Type } from '@google/genai';

/**
 * SentinelAI API Service
 * Handles communication with FastAPI, n8n, and Gemini API.
 */

const FASTAPI_URL =
  import.meta.env.VITE_FASTAPI_URL ||
  import.meta.env.VITE_API_URL ||
  '/api';
const N8N_WEBHOOK_URL = import.meta.env.VITE_N8N_WEBHOOK_URL || '';

export interface AnalysisResult {
  verdict: 'Real' | 'Fake';
  newsVerdict?: string;
  confidence: number;
  framesAnalyzed?: number;
  suspiciousFrames?: number;
  sourceUrl?: string;
  timestamp: string;
  reasoning?: string;
  claim?: string;
  evidence?: string[];
  geminiAnalysis?: string;
  systemWarning?: string | null;
  metadataRiskLevel?: string;
  metadataFlags?: string[];
  suspiciousFrameDetails?: Array<{ timestampSec: number; score: number }>;
  raw?: unknown;
}

function normalizeAnalysisResult(raw: any): AnalysisResult {
  // Backend may return either a flat legacy schema or a nested schema under `visual_analysis`.
  const visual = raw?.visual_analysis ?? raw?.visualAnalysis ?? raw;
  const metadata = raw?.metadata_analysis ?? raw?.metadataAnalysis ?? undefined;

  const verdict = (visual?.verdict ?? raw?.verdict) as AnalysisResult['verdict'] | undefined;
  const confidence = visual?.confidence ?? raw?.confidence;

  const framesAnalyzed =
    visual?.frames_analyzed ??
    visual?.framesAnalyzed ??
    raw?.frames_analyzed ??
    raw?.framesAnalyzed;

  const suspiciousFrames =
    visual?.suspicious_frames_count ??
    visual?.suspiciousFramesCount ??
    raw?.suspicious_frames ??
    raw?.suspiciousFrames;

  const suspiciousFrameDetailsRaw =
    visual?.suspicious_frame_details ??
    visual?.suspiciousFrameDetails ??
    raw?.suspicious_frame_details ??
    raw?.suspiciousFrameDetails;

  const suspiciousFrameDetails = Array.isArray(suspiciousFrameDetailsRaw)
    ? suspiciousFrameDetailsRaw
        .map((d: any) => ({
          timestampSec: Number(d?.timestamp_sec ?? d?.timestampSec),
          score: Number(d?.score),
        }))
        .filter((d: any) => Number.isFinite(d.timestampSec) && Number.isFinite(d.score))
    : undefined;

  const sourceUrl = (raw?.source_url ?? raw?.sourceUrl) as string | undefined;

  const timestamp =
    typeof raw?.timestamp === 'string' && raw.timestamp
      ? raw.timestamp
      : new Date().toISOString();

  return {
    verdict: verdict === 'Fake' ? 'Fake' : 'Real',
    confidence: typeof confidence === 'number' ? confidence : Number(confidence ?? 0),
    framesAnalyzed: typeof framesAnalyzed === 'number' ? framesAnalyzed : Number(framesAnalyzed ?? 0),
    suspiciousFrames:
      typeof suspiciousFrames === 'number' ? suspiciousFrames : Number(suspiciousFrames ?? 0),
    sourceUrl,
    timestamp,
    // If the backend includes these, surface them in UI.
    geminiAnalysis: typeof raw?.gemini_analysis === 'string' ? raw.gemini_analysis : undefined,
    systemWarning: typeof raw?.system_warning === 'string' || raw?.system_warning === null ? raw.system_warning : undefined,
    metadataRiskLevel: typeof metadata?.risk_level === 'string' ? metadata.risk_level : undefined,
    metadataFlags: Array.isArray(metadata?.flags) ? metadata.flags : undefined,
    suspiciousFrameDetails,
    raw,
  };
}

export const apiService = {
  /**
   * Pathway A: Local Upload
   */
  analyzeFile: async (file: File): Promise<AnalysisResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const endpoint = file.type.startsWith('video/') ? '/analyze_video' : '/analyze_image';

    const response = await fetch(`${FASTAPI_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    const raw = await response.json();
    return normalizeAnalysisResult(raw);
  },

  /**
   * Pathway B: URL Analysis
   */
  analyzeUrl: async (url: string): Promise<AnalysisResult> => {
    // Prefer n8n when configured, but fall back to FastAPI.
    if (N8N_WEBHOOK_URL) {
      const response = await fetch(N8N_WEBHOOK_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      if (response.ok) {
        const raw = await response.json();
        return normalizeAnalysisResult(raw);
      }
    }

    const response = await fetch(`${FASTAPI_URL}/analyze_url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      throw new Error(`URL analysis failed: ${response.statusText}${detail ? ` - ${detail}` : ''}`);
    }

    const raw = await response.json();
    return normalizeAnalysisResult(raw);
  },

  /**
   * Pathway C: Text Analysis (Fake News Detection)
   * Uses FastAPI `/verify_news` (alias: `/analyze_news`)
   */
  analyzeText: async (text: string): Promise<AnalysisResult> => {
    const response = await fetch(`${FASTAPI_URL}/verify_news`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ headline: text }),
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      throw new Error(`News analysis failed: ${response.statusText}${detail ? ` - ${detail}` : ''}`);
    }

    const data: any = await response.json();
    const newsVerdict = typeof data?.verdict === 'string' ? data.verdict : '';
    const verdictRaw = newsVerdict.toLowerCase();
    const verdict: AnalysisResult['verdict'] =
      verdictRaw.includes('fake') ? 'Fake' : verdictRaw.includes('real') ? 'Real' : 'Real';

    return {
      verdict,
      newsVerdict,
      confidence: typeof data?.confidence === 'number' ? data.confidence : Number(data?.confidence ?? 0),
      reasoning: typeof data?.reasoning === 'string' ? data.reasoning : '',
      claim: typeof data?.claim === 'string' ? data.claim : undefined,
      evidence: Array.isArray(data?.evidence) ? data.evidence : undefined,
      timestamp: new Date().toISOString(),
      raw: data,
    };
  },
};
