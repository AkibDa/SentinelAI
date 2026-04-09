import { motion } from 'motion/react';
import { Terminal, Code, Cpu, Shield, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { GlassEffect } from './ui/liquid-glass';
import { cn } from '../lib/utils';

export default function ApiPage() {
  const [copied, setCopied] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const codeSnippets = {
    curl: `curl -X GET http://localhost:8080/`,
    python: `import requests

url = "http://localhost:8080/analyze_url"
payload = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}

response = requests.post(url, json=payload)
print(response.json())`,
    node: `import { useEffect, useState } from 'react';

export default function HealthCheck() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8080/')
      .then((response) => response.json())
      .then((json) => setData(json))
      .catch((error) => console.error('Error fetching data:', error));
  }, []);

  return <pre>{JSON.stringify(data, null, 2)}</pre>;
}`
  };

  return (
    <div className="min-h-screen pt-32 pb-20 px-6 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-16"
      >
        {/* Header */}
        <div className="space-y-4">
          <h1 className="text-6xl md:text-8xl font-medium tracking-tighter text-white">
            Developer <span className="text-emerald-500">API</span>
          </h1>
          <p className="text-lg text-white/40 max-w-2xl font-light">
            Integrate our neural forensic engine directly into your workflow. 
            High-throughput, low-latency deepfake detection at scale.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { icon: <Cpu size={20} />, title: "Neural Core", desc: "Access our latest v4.2.0 detection models." },
            { icon: <Shield size={20} />, title: "Secure", desc: "Enterprise-grade encryption and data privacy." },
            { icon: <Terminal size={20} />, title: "RESTful", desc: "Simple, predictable API endpoints." }
          ].map((f, i) => (
            <GlassEffect key={i} className="p-6 rounded-2xl border border-white/5">
              <div className="space-y-4">
                <div className="text-emerald-500">{f.icon}</div>
                <h3 className="text-white font-medium">{f.title}</h3>
                <p className="text-white/30 text-sm leading-relaxed">{f.desc}</p>
              </div>
            </GlassEffect>
          ))}
        </div>

        {/* API Docs Section */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          {/* Left: Documentation */}
          <div className="lg:col-span-5 space-y-10">
            <div className="space-y-6">
              <h2 className="text-2xl font-medium text-white">Authentication</h2>
              <p className="text-white/40 text-sm leading-relaxed">
                For local development, call the FastAPI server directly at <code className="text-emerald-400 bg-emerald-500/10 px-1 rounded">http://localhost:8080</code>. 
                If you deploy publicly, add auth and TLS before exposing endpoints.
              </p>
            </div>

            <div className="space-y-6">
              <h2 className="text-2xl font-medium text-white">Endpoints</h2>
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-bold bg-emerald-500 text-black px-2 py-0.5 rounded">GET</span>
                    <span className="text-xs font-mono text-white/60">/</span>
                  </div>
                  <p className="text-[11px] text-white/30">Health check endpoint to verify backend connectivity from the frontend.</p>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-bold bg-emerald-500 text-black px-2 py-0.5 rounded">POST</span>
                    <span className="text-xs font-mono text-white/60">/analyze_url</span>
                  </div>
                  <p className="text-[11px] text-white/30">Analyze media from a remote URL (YouTube, X, etc).</p>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-bold bg-emerald-500 text-black px-2 py-0.5 rounded">POST</span>
                    <span className="text-xs font-mono text-white/60">/analyze_video | /analyze_image</span>
                  </div>
                  <p className="text-[11px] text-white/30">Analyze uploaded local files with multipart form data.</p>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <h2 className="text-2xl font-medium text-white">Communication Flow</h2>
              <div className="space-y-3 text-white/40 text-sm leading-relaxed">
                <p><span className="text-white/70">Request:</span> Frontend sends an HTTP request to <code className="text-emerald-400 bg-emerald-500/10 px-1 rounded">http://localhost:8080</code>.</p>
                <p><span className="text-white/70">CORS Check:</span> Browser preflight verifies the backend allows your frontend origin.</p>
                <p><span className="text-white/70">Processing:</span> FastAPI handles route logic in the backend API router and returns JSON.</p>
                <p><span className="text-white/70">Response:</span> Frontend receives the JSON and updates UI state.</p>
              </div>
            </div>
          </div>

          {/* Right: Code Samples */}
          <div className="lg:col-span-7">
            <GlassEffect className="rounded-2xl border border-white/10 overflow-hidden">
              <div className="flex flex-col h-full">
                <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-white/[0.02]">
                  <div className="flex items-center gap-2">
                    <Code size={14} className="text-emerald-500" />
                    <span className="text-[10px] uppercase tracking-widest text-white/40 font-medium">Request Sample</span>
                  </div>
                  <div className="flex gap-4">
                    {['curl', 'python', 'node'].map((lang) => (
                      <button key={lang} className="text-[10px] uppercase tracking-widest text-white/20 hover:text-white transition-colors">
                        {lang}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="p-4 md:p-6 bg-black/40 relative group">
                  <pre className="text-[10px] md:text-xs font-mono text-emerald-500/80 leading-relaxed overflow-x-auto pb-4">
                    {codeSnippets.curl}
                  </pre>
                  <button 
                    onClick={() => copyToClipboard(codeSnippets.curl, 'curl')}
                    className="absolute top-4 right-4 p-2 rounded-lg bg-white/5 text-white/20 hover:text-white hover:bg-white/10 transition-all opacity-0 group-hover:opacity-100"
                  >
                    {copied === 'curl' ? <Check size={14} /> : <Copy size={14} />}
                  </button>
                </div>
              </div>
            </GlassEffect>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
