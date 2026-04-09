import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, Link as LinkIcon, ArrowRight, FileVideo, FileImage, Mail, MessageSquare, Send, ShieldAlert, ShieldCheck, Activity } from 'lucide-react';
import { useStore } from '../store/useStore';
import { apiService } from '../services/apiService';
import { cn } from '../lib/utils';
import { GooeyText } from './ui/gooey-text-morphing';
import { GlassEffect, GlassButton } from './ui/liquid-glass';
import { SparklesCore } from './ui/sparkles';
import { SplineScene } from './ui/splite';
import { Spotlight } from './ui/spotlight';
import { Card } from './ui/card';

export default function LandingPage() {
  const [activeTab, setActiveTab] = useState<'upload' | 'link' | 'text'>('upload');
  const [url, setUrl] = useState('');
  const [text, setText] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { setLoading, setResult, setTextResult, setError, isLoading, loadingMessage, textResult, reset } = useStore();

  const handleFileUpload = async (file: File) => {
    try {
      setLoading(true, 'Uploading file...');
      const res = await apiService.analyzeFile(file);
      setLoading(true, 'Running deepfake analysis...');
      await new Promise((r) => setTimeout(r, 1500));
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    }
  };

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    try {
      setLoading(true, 'Fetching URL...');
      const res = await apiService.analyzeUrl(url);
      setLoading(true, 'Running deepfake analysis...');
      await new Promise((r) => setTimeout(r, 1500));
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'URL analysis failed');
    }
  };

  const handleTextSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!text.trim()) return;
    try {
      setLoading(true, 'Analyzing semantic patterns...');
      const res = await apiService.analyzeText(text);
      setLoading(true, 'Verifying cross-references...');
      await new Promise((r) => setTimeout(r, 1000));
      setTextResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Text analysis failed');
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center px-6 py-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        className="max-w-4xl w-full text-center space-y-12 relative"
      >
        <div className="absolute inset-0 -top-40 -z-10 w-full h-[120%]">
          <div className="absolute inset-x-20 top-0 bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent h-[2px] w-3/4 blur-sm" />
          <div className="absolute inset-x-20 top-0 bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent h-px w-3/4" />
          <div className="absolute inset-x-60 top-0 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent h-[5px] w-1/4 blur-sm" />
          <div className="absolute inset-x-60 top-0 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent h-px w-1/4" />

          <SparklesCore
            background="transparent"
            minSize={0.4}
            maxSize={1}
            particleDensity={1200}
            className="w-full h-full"
            particleColor="#10b981"
          />

          <div className="absolute inset-0 w-full h-full bg-black [mask-image:radial-gradient(50%_50%_at_50%_40%,transparent_20%,white)]" />
        </div>

        <div className="space-y-6 relative">
          <motion.div
            className="flex flex-col items-center justify-center relative z-20"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 1 }}
          >
            <h1 className="text-6xl md:text-8xl font-medium tracking-tighter text-white leading-[0.9] mb-12">
              Verify the
            </h1>
            <div className="h-[140px] flex items-center justify-center w-full">
              <GooeyText
                texts={['Sentinel AI', 'Visual Truth', 'Neural Core', 'Fake News']}
                morphTime={1.5}
                cooldownTime={1}
                className="font-medium tracking-tighter"
                textClassName="text-6xl md:text-8xl text-emerald-500"
              />
            </div>
          </motion.div>

          <div className="flex flex-col items-center gap-6 relative z-20">
            <p className="text-lg text-white/40 max-w-xl mx-auto font-light tracking-wide">
              Advanced neural forensics for real-time deepfake detection and semantic news verification.
              Protecting truth in the age of synthetic media.
            </p>
          </div>
        </div>

        <div className="relative max-w-2xl mx-auto">
          <div className="absolute inset-0 -z-10 blur-3xl bg-emerald-500/5 rounded-full" />

          <GlassEffect className="rounded-3xl overflow-hidden shadow-2xl border border-white/10">
            <div className="flex flex-col w-full">
              <div className="flex border-b border-white/5">
                <button
                  onClick={() => setActiveTab('upload')}
                  className={cn(
                    'flex-1 py-4 text-[10px] uppercase tracking-[0.2em] transition-all duration-300',
                    activeTab === 'upload' ? 'text-emerald-500 bg-white/[0.05]' : 'text-white/30 hover:text-white/60'
                  )}
                >
                  Local Media
                </button>
                <button
                  onClick={() => setActiveTab('link')}
                  className={cn(
                    'flex-1 py-4 text-[10px] uppercase tracking-[0.2em] transition-all duration-300',
                    activeTab === 'link' ? 'text-emerald-500 bg-white/[0.05]' : 'text-white/30 hover:text-white/60'
                  )}
                >
                  Remote Link
                </button>
                <button
                  onClick={() => setActiveTab('text')}
                  className={cn(
                    'flex-1 py-4 text-[10px] uppercase tracking-[0.2em] transition-all duration-300',
                    activeTab === 'text' ? 'text-emerald-500 bg-white/[0.05]' : 'text-white/30 hover:text-white/60'
                  )}
                >
                  Fake News
                </button>
              </div>

              <div className="p-8 min-h-[300px] flex flex-col justify-center">
                <AnimatePresence mode="wait">
                  {textResult ? (
                    <motion.div
                      key="text-result"
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="space-y-8"
                    >
                      <div className="flex flex-col items-center gap-4">
                        <div className={cn(
                          'w-20 h-20 rounded-full flex items-center justify-center border transition-all duration-500',
                          textResult.verdict === 'Fake'
                            ? 'border-rose-500/30 text-rose-500 shadow-[0_0_30px_rgba(244,63,94,0.1)]'
                            : 'border-emerald-500/30 text-emerald-500 shadow-[0_0_30px_rgba(16,185,129,0.1)]'
                        )}>
                          {textResult.verdict === 'Fake' ? <ShieldAlert size={40} /> : <ShieldCheck size={40} />}
                        </div>
                        <div className="text-center">
                          <h3 className={cn('text-3xl font-medium tracking-tighter uppercase', textResult.verdict === 'Fake' ? 'text-rose-500' : 'text-emerald-500')}>
                            {textResult.verdict}
                          </h3>
                          <p className="text-[10px] uppercase tracking-widest text-white/20">Confidence: {textResult.confidence}%</p>
                        </div>
                      </div>

                      <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6 space-y-3">
                        <div className="flex items-center justify-between">
                          <p className="text-[9px] uppercase tracking-widest text-white/20">Semantic Reasoning</p>
                          <Activity size={12} className="text-emerald-500/30" />
                        </div>
                        <p className="text-sm text-white/60 font-light italic leading-relaxed">
                          "{textResult.reasoning}"
                        </p>
                      </div>

                      <GlassButton onClick={reset} className="w-full py-3 border border-white/5 text-white/40 hover:text-white">
                        <span className="text-[10px] uppercase tracking-widest">New Text Analysis</span>
                      </GlassButton>
                    </motion.div>
                  ) : activeTab === 'upload' ? (
                    <motion.div
                      key="upload"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 10 }}
                      className={cn(
                        'relative group cursor-pointer border-2 border-dashed rounded-2xl transition-all duration-500 py-16 flex flex-col items-center justify-center gap-4',
                        isDragging ? 'border-emerald-500 bg-emerald-500/5' : 'border-white/10 hover:border-white/20 hover:bg-white/[0.01]'
                      )}
                      onDragOver={onDragOver}
                      onDragLeave={onDragLeave}
                      onDrop={onDrop}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                        accept="video/*,image/*"
                      />

                      <div className="relative">
                        <Upload className={cn('transition-colors duration-300', isDragging ? 'text-emerald-500' : 'text-white/20 group-hover:text-white/40')} size={48} strokeWidth={1} />
                        {isDragging && (
                          <motion.div
                            layoutId="glow"
                            className="absolute inset-0 blur-xl bg-emerald-500/40"
                          />
                        )}
                      </div>

                      <div className="text-center space-y-1">
                        <p className="text-sm text-white/60 font-medium">Drop media here or click to browse</p>
                        <p className="text-[10px] uppercase tracking-widest text-white/20">Supports MP4, MOV, JPG, PNG</p>
                      </div>

                      <div className="absolute bottom-4 flex gap-4 opacity-20">
                        <FileVideo size={16} />
                        <FileImage size={16} />
                      </div>
                    </motion.div>
                  ) : activeTab === 'link' ? (
                    <motion.form
                      key="link"
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                      onSubmit={handleUrlSubmit}
                      className="space-y-6"
                    >
                      <div className="relative group">
                        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-white/20 group-focus-within:text-emerald-500 transition-colors">
                          <LinkIcon size={20} />
                        </div>
                        <input
                          type="url"
                          placeholder="Paste YouTube, Instagram, or X link..."
                          value={url}
                          onChange={(e) => setUrl(e.target.value)}
                          className="w-full bg-white/[0.03] border border-white/10 rounded-2xl py-5 pl-12 pr-4 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-emerald-500/50 focus:bg-white/[0.05] transition-all"
                        />
                      </div>

                      <GlassButton
                        onClick={() => handleUrlSubmit({ preventDefault: () => {} } as any)}
                        className={cn(
                          'w-full bg-emerald-500 hover:bg-emerald-400 py-4',
                          (!url || isLoading) && 'opacity-50 pointer-events-none'
                        )}
                      >
                        <div className="flex items-center justify-center gap-2 text-black font-medium">
                          Analyze Source
                          <ArrowRight size={18} />
                        </div>
                      </GlassButton>

                      <p className="text-[10px] text-center uppercase tracking-widest text-white/20">
                        Processing via n8n neural pipeline
                      </p>
                    </motion.form>
                  ) : (
                    <motion.form
                      key="text"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      onSubmit={handleTextSubmit}
                      className="space-y-6"
                    >
                      <div className="relative group">
                        <textarea
                          placeholder="Paste a headline or news snippet here..."
                          value={text}
                          onChange={(e) => setText(e.target.value)}
                          className="w-full h-32 bg-white/[0.03] border border-white/10 rounded-2xl p-6 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-emerald-500/50 focus:bg-white/[0.05] transition-all resize-none"
                        />
                      </div>

                      <GlassButton
                        onClick={() => handleTextSubmit()}
                        className={cn(
                          'w-full bg-emerald-500 hover:bg-emerald-400 py-4',
                          (!text.trim() || isLoading) && 'opacity-50 pointer-events-none'
                        )}
                      >
                        <div className="flex items-center justify-center gap-2 text-black font-medium">
                          Run Semantic Check
                          <Send size={18} />
                        </div>
                      </GlassButton>

                      <p className="text-[10px] text-center uppercase tracking-widest text-white/20">
                        Powered by Gemini Neural Engine
                      </p>
                    </motion.form>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </GlassEffect>
        </div>

        <div className="pt-20 grid grid-cols-2 md:grid-cols-4 gap-8 w-full max-w-4xl">
          {[
            { label: 'Total Users', value: '128K+' },
            { label: 'Analyses', value: '2.4M+' },
            { label: 'Accuracy', value: '99.8%' },
            { label: 'Latency', value: '< 2.5s' },
          ].map((stat, i) => (
            <div
              key={i}
              className="text-center space-y-1"
            >
              <p className="text-4xl font-medium tracking-tighter text-white">{stat.value}</p>
              <p className="text-[10px] uppercase tracking-[0.2em] text-white/20">{stat.label}</p>
            </div>
          ))}
        </div>

        <div className="pt-32 w-full max-w-5xl">
          <Card className="w-full h-[500px] bg-white/[0.02] border-white/10 relative overflow-hidden rounded-3xl group shadow-2xl">
            <Spotlight
              className="from-emerald-500/20 via-emerald-500/5 to-transparent"
              size={600}
            />

            <div className="flex h-full">
              <div className="flex-1 p-12 relative z-10 flex flex-col justify-center text-left space-y-6">
                <div className="space-y-2">
                  <p className="text-[10px] uppercase tracking-[0.4em] text-emerald-500 font-medium">Core Technology</p>
                  <h2 className="text-5xl font-medium tracking-tighter text-white leading-tight">
                    Interactive <br />
                    <span className="text-emerald-500">Neural Core</span>
                  </h2>
                </div>
                <p className="text-white/40 text-base max-w-md font-light leading-relaxed">
                  Experience the depth of our forensic engine. SentinelAI utilizes
                  multi-layered neural networks to identify synthetic artifacts
                  invisible to the human eye.
                </p>
                <div className="pt-4">
                  <GlassButton className="px-8 py-3 bg-white/[0.03] border-white/10 hover:border-emerald-500/50 transition-all">
                    <span className="text-[10px] uppercase tracking-widest text-white/60">Explore Documentation</span>
                  </GlassButton>
                </div>
              </div>

              <div className="flex-1 relative h-full border-l border-white/5">
                <div className="absolute inset-0 bg-gradient-to-l from-black/40 to-transparent z-10 pointer-events-none" />
                <SplineScene
                  scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                  className="w-full h-full"
                />
              </div>
            </div>
          </Card>
        </div>

        <div className="pt-32 space-y-12 pb-20 w-full max-w-5xl">
          <div className="text-center space-y-4">
            <h2 className="text-3xl font-medium tracking-tighter text-white">Trusted by Forensics Experts</h2>
            <p className="text-white/40 text-sm max-w-md mx-auto font-light">
              Leading journalists and security agencies rely on SentinelAI for visual verification.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { name: 'Sarah Chen', role: 'Digital Forensic Lead', text: 'The accuracy of the neural core is unprecedented. It caught artifacts that our manual process missed entirely.' },
              { name: 'Marcus Thorne', role: 'Investigative Journalist', text: 'SentinelAI has become an essential tool in our newsroom for verifying viral social media content.' },
              { name: 'Dr. Elena Vance', role: 'AI Ethics Researcher', text: 'A vital layer of defense in the age of synthetic media. The transparency of the report is key.' },
            ].map((t, i) => (
              <GlassEffect key={i} className="p-8 rounded-3xl border border-white/5 text-left h-full space-y-6">
                <div className="space-y-6">
                  <p className="text-white/60 text-sm leading-relaxed italic">"{t.text}"</p>
                  <div className="space-y-1">
                    <p className="text-white text-xs font-medium">{t.name}</p>
                    <p className="text-white/20 text-[10px] uppercase tracking-widest">{t.role}</p>
                  </div>
                </div>
              </GlassEffect>
            ))}
          </div>
        </div>

        <div className="pt-32 grid grid-cols-1 md:grid-cols-2 gap-8 pb-20 w-full max-w-5xl">
          <GlassEffect className="p-8 rounded-3xl border border-white/5 flex flex-col items-center justify-center text-center space-y-6 group">
            <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:scale-110 transition-transform duration-500">
              <Mail size={32} strokeWidth={1.5} />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-medium text-white">Contact With Us</h3>
              <p className="text-white/40 text-sm font-light">Have questions? Reach out to our technical support team directly.</p>
            </div>
            <GlassButton
              onClick={() => {
                window.location.href = 'mailto:support@sentinelai.demo';
              }}
              className="px-8 py-3 bg-white/[0.03] border-white/10 hover:border-emerald-500/50 transition-all"
            >
              <span className="text-[10px] uppercase tracking-widest text-white/60">Send Email</span>
            </GlassButton>
          </GlassEffect>

          <GlassEffect className="p-8 rounded-3xl border border-white/5 text-left space-y-6">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-500">
                <MessageSquare size={20} strokeWidth={1.5} />
              </div>
              <h3 className="text-xl font-medium text-white">Give Your Feedback</h3>
            </div>
            <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest text-white/20 ml-1">Your Email</label>
                <input
                  type="email"
                  placeholder="name@gmail.com"
                  className="w-full bg-white/[0.02] border border-white/5 rounded-xl px-4 py-3 text-sm text-white placeholder:text-white/10 focus:outline-none focus:border-emerald-500/30 transition-colors"
                />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest text-white/20 ml-1">Message</label>
                <textarea
                  placeholder="How can we improve?"
                  rows={3}
                  className="w-full bg-white/[0.02] border border-white/5 rounded-xl px-4 py-3 text-sm text-white placeholder:text-white/10 focus:outline-none focus:border-emerald-500/30 transition-colors resize-none"
                />
              </div>
              <GlassButton className="w-full py-3 bg-emerald-500/10 border-emerald-500/20 text-emerald-500 hover:bg-emerald-500 hover:text-white transition-all">
                <span className="text-[10px] uppercase tracking-widest">Submit Feedback</span>
              </GlassButton>
            </form>
          </GlassEffect>
        </div>
      </motion.div>

      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 backdrop-blur-md bg-black/60 flex flex-col items-center justify-center gap-6"
          >
            <div className="relative">
              <div className="w-16 h-16 border-2 border-emerald-500/20 rounded-full" />
              <motion.div
                className="absolute inset-0 border-2 border-emerald-500 border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              />
            </div>
            <div className="text-center space-y-2">
              <p className="text-sm font-medium text-white tracking-wide">{loadingMessage}</p>
              <div className="flex gap-1 justify-center">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-1 h-1 bg-emerald-500 rounded-full"
                    animate={{ opacity: [0.2, 1, 0.2] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
