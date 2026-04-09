/**
 * SentinelAI API Service
 * Handles communication with FastAPI backend.
 */

const BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_FASTAPI_URL ||
  'http://localhost:8080';
const N8N_WEBHOOK_URL = import.meta.env.VITE_N8N_WEBHOOK_URL;

export interface AnalysisResult {
  verdict: 'Real' | 'Fake';
  confidence: number;
  framesAnalyzed: number;
  suspiciousFrames: number;
  rawScore?: number;
  sourceUrl?: string;
  timestamp: string;
}

interface BackendAnalysisResult {
  verdict: 'Real' | 'Fake' | string;
  confidence: number;
  frames_analyzed?: number;
  suspicious_frames?: number;
  raw_score?: number;
  source_url?: string;
}

const toAnalysisResult = (payload: BackendAnalysisResult): AnalysisResult => ({
  verdict: payload.verdict === 'Fake' ? 'Fake' : 'Real',
  confidence: payload.confidence ?? 0,
  framesAnalyzed: payload.frames_analyzed ?? 0,
  suspiciousFrames: payload.suspicious_frames ?? 0,
  rawScore: payload.raw_score,
  sourceUrl: payload.source_url,
  timestamp: new Date().toISOString(),
});

export const apiService = {
  /**
   * Pathway A: Local Upload
   * Hits FastAPI server with multipart/form-data
   */
  analyzeFile: async (file: File): Promise<AnalysisResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const endpoint = file.type.startsWith('video/') ? '/analyze_video' : '/analyze_image';
    
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
      // Timeout handled by the caller or global config
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    const json = (await response.json()) as BackendAnalysisResult;
    return toAnalysisResult(json);
  },

  /**
   * Pathway B: URL Analysis
   * Hits n8n webhook first, then falls back to FastAPI if unavailable
   */
  analyzeUrl: async (url: string): Promise<AnalysisResult> => {
    const payload = JSON.stringify({ url });

    if (N8N_WEBHOOK_URL) {
      const n8nResponse = await fetch(N8N_WEBHOOK_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: payload,
      });

      if (!n8nResponse.ok) {
        throw new Error(`n8n request failed: ${n8nResponse.statusText}`);
      }

      const n8nJson = (await n8nResponse.json()) as BackendAnalysisResult;
      return toAnalysisResult(n8nJson);
    }

    const fastApiResponse = await fetch(`${BASE_URL}/analyze_url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: payload,
    });

    if (!fastApiResponse.ok) {
      throw new Error(`URL analysis failed: ${fastApiResponse.statusText}`);
    }

    const fastApiJson = (await fastApiResponse.json()) as BackendAnalysisResult;
    return toAnalysisResult(fastApiJson);
  },
};
