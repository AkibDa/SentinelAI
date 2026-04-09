import { create } from 'zustand';
import { AnalysisResult } from '../services/apiService';

interface AppState {
  isInitializing: boolean;
  hasEntered: boolean;
  isLoading: boolean;
  loadingMessage: string;
  result: AnalysisResult | null;
  error: string | null;
  currentPage: 'home' | 'api';
  
  setInitializing: (val: boolean) => void;
  setEntered: (val: boolean) => void;
  setLoading: (isLoading: boolean, message?: string) => void;
  setResult: (result: AnalysisResult | null) => void;
  setError: (error: string | null) => void;
  setCurrentPage: (page: 'home' | 'api') => void;
  reset: () => void;
}

export const useStore = create<AppState>((set) => ({
  isInitializing: true,
  hasEntered: false,
  isLoading: false,
  loadingMessage: '',
  result: null,
  error: null,
  currentPage: 'home',

  setInitializing: (val) => set({ isInitializing: val }),
  setEntered: (val) => set({ hasEntered: val }),
  setLoading: (isLoading, message = '') => set({ isLoading, loadingMessage: message }),
  setResult: (result) => set({ result, error: null }),
  setError: (error) => set({ error, isLoading: false }),
  setCurrentPage: (page) => set({ currentPage: page }),
  reset: () => set({ result: null, error: null, isLoading: false, loadingMessage: '' }),
}));
