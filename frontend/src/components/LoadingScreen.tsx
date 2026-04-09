import { motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { useStore } from '../store/useStore';
import Logo from './Logo';

export default function LoadingScreen() {
  const [progress, setProgress] = useState(0);
  const setInitializing = useStore((state) => state.setInitializing);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => setInitializing(false), 500);
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 200);

    return () => clearInterval(interval);
  }, [setInitializing]);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black"
    >
      <div className="relative mb-8">
        <Logo size={80} />
      </div>

      <div className="w-64 space-y-4">
        <div className="flex justify-between text-[10px] uppercase tracking-[0.2em] text-emerald-500/60 font-mono">
          <span>Initializing Neural Core</span>
          <span>{Math.round(progress)}%</span>
        </div>
        
        <div className="h-[2px] w-full bg-white/5 overflow-hidden rounded-full">
          <motion.div
            className="h-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
          />
        </div>
      </div>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 1, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="mt-4 text-[10px] uppercase tracking-[0.3em] text-white/30 font-mono"
      >
        Authenticating Sentinel Protocols...
      </motion.p>
    </motion.div>
  );
}
