import { create } from 'zustand';
import { AnalysisResult } from '../services/apiService';

interface AppState {
  isInitializing: boolean;
  hasEntered: boolean;
  isLoading: boolean;
  loadingMessage: string;
  result: AnalysisResult | null;
  textResult: AnalysisResult | null;
  error: string | null;
  currentPage: 'home' | 'text';
  
  setInitializing: (val: boolean) => void;
  setEntered: (val: boolean) => void;
  setLoading: (isLoading: boolean, message?: string) => void;
  setResult: (result: AnalysisResult | null) => void;
  setTextResult: (result: AnalysisResult | null) => void;
  setError: (error: string | null) => void;
  setCurrentPage: (page: 'home' | 'text') => void;
  reset: () => void;
}

export const useStore = create<AppState>((set) => ({
  isInitializing: true,
  hasEntered: false,
  isLoading: false,
  loadingMessage: '',
  result: null,
  textResult: null,
  error: null,
  currentPage: 'home',

  setInitializing: (val) => set({ isInitializing: val }),
  setEntered: (val) => set({ hasEntered: val }),
  setLoading: (isLoading, message = '') => set({ isLoading, loadingMessage: message }),
  setResult: (result) => set({ result, error: null }),
  setTextResult: (result) => set({ textResult: result, error: null }),
  setError: (error) => set({ error, isLoading: false }),
  setCurrentPage: (page) => set({ currentPage: page }),
  reset: () => set({ result: null, textResult: null, error: null, isLoading: false, loadingMessage: '' }),
}));
