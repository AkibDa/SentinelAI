/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { AnimatePresence, motion } from 'motion/react';
import { useStore } from './store/useStore';
import { cn } from './lib/utils';
import ThreeBackground from './components/ThreeBackground';
import LoadingScreen from './components/LoadingScreen';
import IntroScreen from './components/IntroScreen';
import LandingPage from './components/LandingPage';
import Dashboard from './components/Dashboard';
import ApiPage from './components/ApiPage';
import { GlassFilter } from './components/ui/liquid-glass';
import Logo from './components/Logo';

export default function App() {
  const { isInitializing, hasEntered, result, error, setError, currentPage, setCurrentPage } = useStore();

  return (
    <div className="relative min-h-screen bg-black text-white selection:bg-emerald-500/30 overflow-x-hidden">
      <ThreeBackground />
      <GlassFilter />
      
      <AnimatePresence mode="wait">
        {isInitializing ? (
          <LoadingScreen key="loading" />
        ) : !hasEntered ? (
          <IntroScreen key="intro" />
        ) : (
          <main key={currentPage} className="relative z-10">
            {currentPage === 'api' ? (
              <ApiPage />
            ) : result ? (
              <Dashboard />
            ) : (
              <LandingPage />
            )}
          </main>
        )}
      </AnimatePresence>

      {/* Error Toast */}
      <AnimatePresence>
        {error && (
          <div className="fixed bottom-8 right-8 z-50">
            <div className="backdrop-blur-xl bg-rose-500/10 border border-rose-500/20 px-6 py-4 rounded-2xl flex items-center gap-4 shadow-2xl">
              <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
              <p className="text-sm font-medium text-rose-200">{error}</p>
              <button 
                onClick={() => setError(null)}
                className="text-white/20 hover:text-white transition-colors text-xs uppercase tracking-widest ml-4"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}
      </AnimatePresence>

      {/* Footer Info */}
      <div className="fixed bottom-12 left-12 hidden md:block">
        <p className="text-[10px] uppercase tracking-[0.3em] text-white/20 font-mono">
          Neural Core v4.2.0 // Secure Connection
        </p>
      </div>
    </div>
  );
}
