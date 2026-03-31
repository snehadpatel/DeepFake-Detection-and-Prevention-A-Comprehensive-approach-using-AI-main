"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  ShieldCheck, 
  Search, 
  Upload, 
  AlertCircle, 
  Cpu, 
  BarChart3, 
  Download, 
  RefreshCcw,
  Fingerprint,
  Zap
} from "lucide-react";
import axios from "axios";
import { useDropzone } from "react-dropzone";

// --- Components ---

const Navbar = ({ activeTab, setActiveTab }: { activeTab: string, setActiveTab: (t: string) => void }) => (
  <nav className="fixed left-0 top-0 h-screen w-72 glass border-r border-white/5 p-8 flex flex-col gap-12 z-50">
    <div className="flex items-center gap-3">
      <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-2.5 rounded-2xl shadow-lg shadow-indigo-500/30">
        <ShieldCheck className="w-8 h-8 text-white" />
      </div>
      <div>
        <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">DeepShield AI</h1>
        <p className="text-[10px] uppercase font-bold tracking-widest text-indigo-400">Pro-Active Guardian</p>
      </div>
    </div>

    <div className="flex flex-col gap-3">
      <p className="text-[10px] uppercase font-bold tracking-widest text-slate-500 mb-2 ml-4">Workspace</p>
      <div 
        onClick={() => setActiveTab("detect")}
        className={`nav-item ${activeTab === "detect" ? "nav-item-active" : "glass-hover"}`}
      >
        <Search className="w-5 h-5" />
        <span className="font-semibold">Deepfake Detector</span>
      </div>
      <div 
        onClick={() => setActiveTab("protect")}
        className={`nav-item ${activeTab === "protect" ? "nav-item-active" : "glass-hover"}`}
      >
        <Zap className="w-5 h-5" />
        <span className="font-semibold">Digital Guardian</span>
      </div>
    </div>

    <div className="mt-auto pointer-events-none opacity-40">
      <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
        <BarChart3 className="w-6 h-6 mb-3 text-indigo-400" />
        <p className="text-sm font-medium mb-1">99.2% Accuracy</p>
        <p className="text-xs text-slate-400">Powered by EfficientNetB0</p>
      </div>
    </div>
  </nav>
);

// --- Detect Workspace ---

const DetectWorkspace = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const onDrop = (acceptedFiles: File[]) => {
    setFile(acceptedFiles[0]);
    setResult(null);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop, 
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false
  });

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8000/api/detect", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error analyzing image. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8 max-w-5xl mx-auto py-12">
      <div className="space-y-3">
        <h2 className="text-4xl font-bold tracking-tight">Deepfake Detection</h2>
        <p className="text-slate-400 max-w-2xl">Upload media to analyze inconsistencies, artifacts, and latent space anomalies using our neural explainability engine.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* Upload Area */}
        <div className="space-y-6">
          <div 
            {...getRootProps()} 
            className={`cursor-pointer transition-all duration-500 rounded-3xl border-2 border-dashed flex flex-col items-center justify-center p-12 gap-4 ${
              isDragActive ? "border-indigo-500 bg-indigo-500/10" : "border-white/10 hover:border-white/20 glass"
            }`}
          >
            <input {...getInputProps()} />
            {file ? (
              <div className="relative group overflow-hidden rounded-2xl">
                <img src={URL.createObjectURL(file)} alt="Uploaded" className="max-h-64 object-contain" />
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <RefreshCcw className="w-8 h-8 text-white" />
                </div>
              </div>
            ) : (
              <div className="p-6 rounded-full bg-white/5 border border-white/10">
                <Upload className="w-10 h-10 text-slate-400" />
              </div>
            )}
            <div className="text-center">
              <p className="text-lg font-semibold">{file ? file.name : "Select or drag file"}</p>
              <p className="text-sm text-slate-500">Supports JPG, PNG up to 10MB</p>
            </div>
          </div>

          <button 
            disabled={!file || loading}
            onClick={analyze}
            className="btn-primary w-full flex items-center justify-center gap-3 py-4"
          >
            {loading ? (
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 border-t-2 border-white rounded-full animate-spin"></div>
                Analyzing...
              </div>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Run Analysis
              </>
            )}
          </button>
        </div>

        {/* Results Area */}
        <AnimatePresence mode="wait">
          {result ? (
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="glass p-8 rounded-3xl border border-white/5 space-y-10 min-h-[500px]"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-1">AI Verdict</p>
                  <div className={`text-3xl font-black ${result.isReal ? "text-emerald-400" : "text-rose-500"}`}>
                    {result.isReal ? "Likely Real" : "Deepfake Detected"}
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-1">Confidence</p>
                  <div className="text-2xl font-bold">{(result.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>

              <div className="space-y-4">
                <p className="text-xs font-bold uppercase tracking-widest text-slate-500">Neural Heatmap (Grad-CAM)</p>
                <div className="relative rounded-2xl overflow-hidden glass border border-white/10 group">
                  <img src={result.heatmap} alt="Heatmap" className="w-full object-cover transition-all duration-500 group-hover:scale-105" />
                  <div className="absolute top-4 left-4 p-2 bg-black/60 backdrop-blur-md rounded-lg text-[10px] font-bold text-white border border-white/20">
                    EXPLAINABILITY OVERLAY
                  </div>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed italic">
                  {result.isReal 
                    ? "The model finds consistent facial geometry and organic lighting patterns across the source frame."
                    : "Anomalies detected in skin texture and ocular reflections. Red areas indicate strong structural inconsistencies common in generated media."}
                </p>
              </div>
            </motion.div>
          ) : (
            <div className="border-2 border-white/5 rounded-3xl border-dashed h-full min-h-[500px] flex flex-col items-center justify-center text-center p-12 opacity-50 grayscale">
              <Cpu className="w-12 h-12 text-slate-600 mb-4 animate-pulse-slow" />
              <p className="text-lg font-medium text-slate-500 italic">Neural Engine Waiting...</p>
              <p className="text-xs text-slate-600 max-w-xs mt-2">Upload an image to start the deep-layer artifact analysis.</p>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// --- Protect Workspace ---

const ProtectWorkspace = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const onDrop = (acceptedFiles: File[]) => {
    setFile(acceptedFiles[0]);
    setResult(null);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop, 
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false
  });

  const protect = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8000/api/protect", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error protecting image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8 max-w-5xl mx-auto py-12">
      <div className="space-y-3">
        <h2 className="text-4xl font-bold tracking-tight flex items-center gap-3">
          DeepShield Guardian <Zap className="w-8 h-8 text-indigo-400" />
        </h2>
        <p className="text-slate-400 max-w-2xl">Immunize your photos against deepfake manipulation using adversarial noise that disrupts GAN latent structures.</p>
      </div>

      <div className="grid grid-cols-1 gap-8">
        {!result ? (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-3xl p-12 flex flex-col gap-8 border border-white/10"
          >
            <div 
              {...getRootProps()} 
              className={`cursor-pointer transition-all duration-500 rounded-3xl border-2 border-dashed flex flex-col items-center justify-center p-20 gap-4 ${
                isDragActive ? "border-indigo-500 bg-indigo-500/10" : "border-white/10 hover:border-white/20 bg-white/5"
              }`}
            >
              <input {...getInputProps()} />
              {file ? (
                <div className="relative rounded-2xl overflow-hidden border border-white/20">
                  <img src={URL.createObjectURL(file)} alt="To Protect" className="max-h-64 object-contain" />
                </div>
              ) : (
                <div className="p-8 rounded-full bg-indigo-500/10 border border-indigo-500/20">
                  <Fingerprint className="w-12 h-12 text-indigo-400" />
                </div>
              )}
              <div className="text-center">
                <p className="text-xl font-bold">{file ? file.name : "Upload Private Photo"}</p>
                <p className="text-sm text-slate-500">Add a proactive shield before uploading to social media.</p>
              </div>
            </div>

            <button 
              disabled={!file || loading}
              onClick={protect}
              className="btn-primary py-5 text-lg"
            >
              {loading ? "Injecting Adversarial Shield..." : "Apply AI Protection"}
            </button>
          </motion.div>
        ) : (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-8"
          >
            <div className="glass rounded-3xl p-8 border border-white/10 flex flex-col lg:flex-row gap-8 items-center">
              <div className="flex-1 w-full space-y-3">
                <p className="text-xs font-bold uppercase tracking-widest text-slate-500">Secured Result</p>
                <div className="relative rounded-2xl overflow-hidden glass border border-white/20">
                  <img src={result.protectedImage} alt="Protected" className="w-full object-contain" />
                  <div className="absolute top-4 right-4 bg-emerald-500/20 text-emerald-400 backdrop-blur-md px-3 py-1.5 rounded-full text-[10px] font-bold border border-emerald-500/30 flex items-center gap-1.5 shadow-lg">
                    <ShieldCheck className="w-4 h-4" /> PROTECTED
                  </div>
                </div>
              </div>
              <div className="flex-1 w-full space-y-3">
                <p className="text-xs font-bold uppercase tracking-widest text-slate-500">Adversarial Map (Amplified)</p>
                <div className="relative rounded-2xl overflow-hidden glass border border-white/20">
                  <img src={result.noiseMap} alt="Noise Map" className="w-full object-contain" />
                  <div className="absolute top-4 right-4 bg-slate-900/60 text-slate-300 backdrop-blur-md px-3 py-1.5 rounded-full text-[10px] font-bold border border-white/10 uppercase">
                    Non-Visible Perturbation
                  </div>
                </div>
              </div>
            </div>

            <div className="flex gap-4">
              <button 
                onClick={() => setResult(null)}
                className="btn-secondary flex-1 flex items-center justify-center gap-2"
              >
                <RefreshCcw className="w-5 h-5" /> Protect Another
              </button>
              <a 
                href={result.protectedImage} 
                download="deepshield_protected.png"
                className="btn-primary flex-1 flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" /> Download Protected Image
              </a>
            </div>
            
            <div className="bg-indigo-500/10 border border-indigo-500/20 p-6 rounded-2xl flex gap-4">
              <AlertCircle className="w-6 h-6 text-indigo-400 shrink-0" />
              <div>
                <h4 className="font-bold text-indigo-300">How it works</h4>
                <p className="text-sm text-indigo-400/80 mt-1">
                  We've embedded high-frequency adversarial noise that is imperceptible to your eyes but completely confuses deepfake algorithms (GANs). If someone tries to swap your face, the result will be garbled or fail to process correctly.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

// --- Main App ---

export default function Home() {
  const [activeTab, setActiveTab] = useState("detect");

  return (
    <main className="min-h-screen">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <div className="pl-72 min-h-screen">
        <header className="px-12 py-8 flex justify-end">
          <div className="flex items-center gap-6">
            <div className="flex flex-col items-end">
              <p className="text-[10px] uppercase font-bold text-slate-500 tracking-[0.2em]">System Status</p>
              <p className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                Node Active
              </p>
            </div>
            <div className="w-px h-8 bg-white/10"></div>
            <div className="bg-white/5 border border-white/10 px-4 py-2 rounded-xl text-xs font-bold text-slate-400">
              V 2.0.4 - STABLE
            </div>
          </div>
        </header>

        <section className="px-12 pb-20">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              {activeTab === "detect" ? <DetectWorkspace /> : <ProtectWorkspace />}
            </motion.div>
          </AnimatePresence>
        </section>
      </div>
    </main>
  );
}
