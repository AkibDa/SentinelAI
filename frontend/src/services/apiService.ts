/**
 * SentinelAI API Service
 * Handles communication with FastAPI and n8n backends.
 */

const FASTAPI_URL = import.meta.env.VITE_FASTAPI_URL || 'http://localhost:8000';
const N8N_WEBHOOK_URL = import.meta.env.VITE_N8N_WEBHOOK_URL || '';

export interface AnalysisResult {
  verdict: 'Real' | 'Fake';
  confidence: number;
  framesAnalyzed: number;
  suspiciousFrames: number;
  sourceUrl?: string;
  timestamp: string;
}

export const apiService = {
  /**
   * Pathway A: Local Upload
   * Hits FastAPI server with multipart/form-data
   */
  analyzeFile: async (file: File): Promise<AnalysisResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const endpoint = file.type.startsWith('video/') ? '/analyze_video' : '/analyze_image';
    
    const response = await fetch(`${FASTAPI_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
      // Timeout handled by the caller or global config
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Pathway B: URL Analysis
   * Hits n8n Webhook with JSON body
   */
  analyzeUrl: async (url: string): Promise<AnalysisResult> => {
    if (!N8N_WEBHOOK_URL) {
      throw new Error('n8n Webhook URL is not configured');
    }

    const response = await fetch(N8N_WEBHOOK_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      throw new Error(`URL analysis failed: ${response.statusText}`);
    }

    return response.json();
  },
};
