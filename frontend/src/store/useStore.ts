import { create } from 'zustand';
import { AnalysisResult } from '../services/apiService';

interface AppState {
  isInitializing: boolean;
  hasEntered: boolean;
  isLoading: boolean;
  loadingMessage: string;
  result: AnalysisResult | null;
  error: string | null;
  
  setInitializing: (val: boolean) => void;
  setEntered: (val: boolean) => void;
  setLoading: (isLoading: boolean, message?: string) => void;
  setResult: (result: AnalysisResult | null) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useStore = create<AppState>((set) => ({
  isInitializing: true,
  hasEntered: false,
  isLoading: false,
  loadingMessage: '',
  result: null,
  error: null,

  setInitializing: (val) => set({ isInitializing: val }),
  setEntered: (val) => set({ hasEntered: val }),
  setLoading: (isLoading, message = '') => set({ isLoading, loadingMessage: message }),
  setResult: (result) => set({ result, error: null }),
  setError: (error) => set({ error, isLoading: false }),
  reset: () => set({ result: null, error: null, isLoading: false, loadingMessage: '' }),
}));
