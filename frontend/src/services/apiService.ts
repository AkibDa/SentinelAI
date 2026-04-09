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
  confidence: number;
  framesAnalyzed?: number;
  suspiciousFrames?: number;
  sourceUrl?: string;
  timestamp: string;
  reasoning?: string;
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

    return response.json();
  },

  /**
   * Pathway B: URL Analysis
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

  /**
   * Pathway C: Text Analysis (Fake News Detection)
   * Uses Gemini API for semantic analysis
   */
  analyzeText: async (text: string): Promise<AnalysisResult> => {
    try {
      const geminiApiKey = process.env.GEMINI_API_KEY;
      if (!geminiApiKey) {
        throw new Error('GEMINI_API_KEY is missing. Set it in frontend env and restart Vite.');
      }
      const ai = new GoogleGenAI({ apiKey: geminiApiKey });

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: `Analyze the following headline or news content for potential misinformation or "fake news" characteristics.
        Provide a verdict (Real or Fake), a confidence score (0-100), and a brief reasoning.

        Content: "${text}"`,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              verdict: { type: Type.STRING, enum: ['Real', 'Fake'] },
              confidence: { type: Type.NUMBER },
              reasoning: { type: Type.STRING },
            },
            required: ['verdict', 'confidence', 'reasoning'],
          },
        },
      });

      const data = JSON.parse(response.text ?? '{}');

      return {
        verdict: data.verdict as 'Real' | 'Fake',
        confidence: data.confidence,
        reasoning: data.reasoning,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error('Gemini Analysis Error:', error);
      throw new Error('Failed to analyze text. Please try again.');
    }
  },
};
