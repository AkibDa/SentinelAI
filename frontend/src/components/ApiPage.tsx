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
    curl: `curl -X POST https://api.sentinelai.io/v1/analyze \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "file=@video.mp4"`,
    python: `import requests

url = "https://api.sentinelai.io/v1/analyze"
files = {"file": open("video.mp4", "rb")}
headers = {"Authorization": "Bearer YOUR_API_KEY"}

response = requests.post(url, files=files, headers=headers)
print(response.json())`,
    node: `const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('video.mp4'));

axios.post('https://api.sentinelai.io/v1/analyze', form, {
  headers: {
    ...form.getHeaders(),
    'Authorization': 'Bearer YOUR_API_KEY'
  }
}).then(res => console.log(res.data));`
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
                All API requests must include your API Key in the <code className="text-emerald-400 bg-emerald-500/10 px-1 rounded">Authorization</code> header. 
                You can manage your keys in the developer dashboard.
              </p>
            </div>

            <div className="space-y-6">
              <h2 className="text-2xl font-medium text-white">Endpoints</h2>
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-bold bg-emerald-500 text-black px-2 py-0.5 rounded">POST</span>
                    <span className="text-xs font-mono text-white/60">/v1/analyze</span>
                  </div>
                  <p className="text-[11px] text-white/30">Analyze a local video or image file for synthetic artifacts.</p>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-bold bg-emerald-500 text-black px-2 py-0.5 rounded">POST</span>
                    <span className="text-xs font-mono text-white/60">/v1/analyze-url</span>
                  </div>
                  <p className="text-[11px] text-white/30">Analyze media from a remote URL (YouTube, X, etc).</p>
                </div>
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
