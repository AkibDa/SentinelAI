import { useEffect } from 'react';
import { motion } from 'motion/react';
import { useStore } from '../store/useStore';
import { SparklesCore } from './ui/sparkles';
import { SplineScene } from './ui/splite';
import { Spotlight } from './ui/spotlight';

export default function IntroScreen() {
  const { setEntered } = useStore();

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setEntered(true);
      }
    };

    const handleWheel = (e: WheelEvent) => {
      if (e.deltaY > 0) {
        setEntered(true);
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      // Simple touch detection for scroll
      setEntered(true);
    };

    window.addEventListener('scroll', handleScroll);
    window.addEventListener('wheel', handleWheel);
    window.addEventListener('touchmove', handleTouchMove);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('wheel', handleWheel);
      window.removeEventListener('touchmove', handleTouchMove);
    };
  }, [setEntered]);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, y: -100, filter: 'blur(20px)' }}
      transition={{ duration: 1.2, ease: "easeInOut" }}
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-black"
    >
      <Spotlight className="from-emerald-500/20 via-emerald-500/5 to-transparent" size={600} />

      {/* Background Spline Scene */}
      <div className="absolute inset-0 z-0 opacity-60">
        <SplineScene 
          scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
          className="w-full h-full"
        />
      </div>

      {/* Background Sparkles (Subtle) */}
      <div className="absolute inset-0 z-[1] pointer-events-none">
        <SparklesCore
          id="intro-sparkles"
          background="transparent"
          minSize={0.4}
          maxSize={1}
          particleDensity={20}
          className="w-full h-full"
          particleColor="#10b981"
          speed={0.3}
        />
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center text-center space-y-12 max-w-2xl px-6">
        <div className="space-y-4">
          <motion.h1 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.7, duration: 0.8 }}
            className="text-5xl md:text-7xl font-medium tracking-tighter text-white"
          >
            Sentinel<span className="text-emerald-500">AI</span>
          </motion.h1>
          <motion.p 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.9, duration: 0.8 }}
            className="text-white/40 text-sm md:text-base tracking-[0.3em] uppercase font-light"
          >
            Neural Forensics Protocol Active
          </motion.p>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
          className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-6"
        >
          <div className="flex flex-col items-center gap-2">
            <p className="text-[10px] uppercase tracking-[0.4em] text-white/30 animate-pulse">
              Scroll to Begin
            </p>
            <motion.div 
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              className="w-[1px] h-12 bg-gradient-to-b from-emerald-500/50 to-transparent"
            />
          </div>
          
          <p className="text-[10px] uppercase tracking-[0.5em] text-white/10">
            Aether Edition // v4.2.0
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
}
