import { motion } from 'motion/react';
import { Shield, ShieldAlert, ShieldCheck, Activity, Clock, Globe, RefreshCcw, ArrowLeft } from 'lucide-react';
import { useStore } from '../store/useStore';
import { cn } from '../lib/utils';
import { GlassEffect, GlassButton } from './ui/liquid-glass';

export default function Dashboard() {
  const { result, reset } = useStore();

  if (!result) return null;

  const isFake = result.verdict === 'Fake';
  const colorClass = isFake ? 'text-rose-500' : 'text-emerald-500';
  const bgClass = isFake ? 'bg-rose-500' : 'bg-emerald-500';
  const borderClass = isFake ? 'border-rose-500/30' : 'border-emerald-500/30';
  const glowClass = isFake ? 'shadow-[0_0_50px_rgba(244,63,94,0.2)]' : 'shadow-[0_0_50px_rgba(16,185,129,0.2)]';

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center px-6 py-20">
      {/* Top Navigation */}
      <div className="absolute top-12 left-12 flex items-center gap-8">
        <button 
          onClick={reset}
          className="flex items-center gap-2 text-white/40 hover:text-white transition-colors group"
        >
          <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
          <span className="text-xs uppercase tracking-widest">New Analysis</span>
        </button>
      </div>

      <div className="max-w-5xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
        {/* Left: Verdict Shield & Gauge */}
        <div className="lg:col-span-7 flex flex-col items-center space-y-12">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", damping: 15 }}
            className="relative"
          >
            {/* The Verdict Shield */}
            <div className={cn(
              "relative z-10 w-64 h-64 rounded-full flex items-center justify-center backdrop-blur-3xl bg-white/[0.02] border transition-all duration-1000",
              borderClass,
              glowClass
            )}>
              <motion.div
                initial={{ rotateY: 180 }}
                animate={{ rotateY: 0 }}
                transition={{ duration: 1, delay: 0.5 }}
              >
                {isFake ? (
                  <ShieldAlert size={120} strokeWidth={1} className={colorClass} />
                ) : (
                  <ShieldCheck size={120} strokeWidth={1} className={colorClass} />
                )}
              </motion.div>
              
              {/* Inner Glow */}
              <div className={cn("absolute inset-4 rounded-full blur-2xl opacity-10", bgClass)} />
            </div>
          </motion.div>

          <div className="text-center space-y-2">
            <motion.h2 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
              className={cn("text-6xl font-medium tracking-tighter uppercase", colorClass)}
            >
              {result.verdict}
            </motion.h2>
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.4 }}
              className="text-white/40 text-sm tracking-[0.2em] uppercase"
            >
              Confidence Score: {result.confidence}%
            </motion.p>
          </div>
        </div>

        {/* Right: Technical Context Card */}
        <div className="lg:col-span-5 space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1.6 }}
          >
            <GlassEffect className="rounded-3xl border border-white/10">
              <div className="p-8 space-y-8 w-full">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs uppercase tracking-[0.2em] text-white/60">Forensic Report</h3>
                  <Activity size={14} className="text-emerald-500/50" />
                </div>

                <div className="space-y-6">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-white/40">
                      <RefreshCcw size={18} />
                    </div>
                    <div>
                      <p className="text-[10px] uppercase tracking-widest text-white/20">Frames Analyzed</p>
                      <p className="text-lg font-mono text-white">{result.framesAnalyzed.toLocaleString()}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-white/40">
                      <ShieldAlert size={18} />
                    </div>
                    <div>
                      <p className="text-[10px] uppercase tracking-widest text-white/20">Suspicious Frames</p>
                      <p className={cn("text-lg font-mono", result.suspiciousFrames > 0 ? "text-rose-500" : "text-white/40")}>
                        {result.suspiciousFrames.toLocaleString()}
                      </p>
                    </div>
                  </div>

                  {result.sourceUrl && (
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-white/40">
                        <Globe size={18} />
                      </div>
                      <div className="overflow-hidden">
                        <p className="text-[10px] uppercase tracking-widest text-white/20">Source URL</p>
                        <p className="text-xs font-mono text-white/60 truncate">{result.sourceUrl}</p>
                      </div>
                    </div>
                  )}

                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-white/40">
                      <Clock size={18} />
                    </div>
                    <div>
                      <p className="text-[10px] uppercase tracking-widest text-white/20">Analysis Timestamp</p>
                      <p className="text-xs font-mono text-white/60">{new Date(result.timestamp).toLocaleString()}</p>
                    </div>
                  </div>
                </div>

                <div className="pt-6 border-t border-white/5">
                  <p className="text-[9px] leading-relaxed text-white/20 uppercase tracking-wider">
                    Neural core analysis complete. The verdict is based on frame-by-frame inconsistency detection and facial artifact mapping.
                  </p>
                </div>
              </div>
            </GlassEffect>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2 }}
          >
            <GlassButton
              onClick={reset}
              className="w-full py-4 border border-white/10 text-white/60 hover:text-white"
            >
              <span className="text-xs uppercase tracking-[0.2em]">Dismiss Report</span>
            </GlassButton>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
