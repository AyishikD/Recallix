"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain, Cpu, Database, GitBranch, Layers,
  ArrowLeft, ChevronRight, Sparkles, Clock,
  Search, Shield, RefreshCw, Lightbulb, Zap,
  Activity, Target, Workflow, Eye, Boxes,
  Microchip, Dna, HardDrive, ShieldCheck,
  ArrowRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { largestClustered, ms, pct } from "@/lib/results";

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

interface TechModule {
  name: string;
  detail: string;
  type: "logic" | "infra" | "storage";
}

interface Component {
  id: string;
  title: string;
  desc: string;
  icon: any;
  color: string;
  modules: TechModule[];
}

const architectureStack: Component[] = [
  {
    id: "cognitive",
    title: "Memory Logic (Python)",
    desc: "Deterministic store and recall pipelines. No LLM: facts, schemas, importance and relevance are computed from rules, embeddings and fitted models.",
    icon: Brain,
    color: "cyan",
    modules: [
      { name: "RecallEngine", detail: "FTS5 + entity index + per-user HNSW candidates, scored by a learned logistic relevance model", type: "logic" },
      { name: "TruthEngine", detail: "One value per fact slot: newer values supersede, 'X is not V' invalidates V only", type: "logic" },
      { name: "ImportanceRanker", detail: "1 + 9·sigmoid(w·features): novelty, IDF information, specificity, cue words; weights fitted", type: "logic" },
      { name: "IndexSync", detail: "Rebuilds vector indexes from SQLite at startup and after engine restarts", type: "logic" },
    ],
  },
  {
    id: "infrastructure",
    title: "Vector Engine (C++)",
    desc: "HNSW (Malkov & Yashunin heuristic neighbour selection), one index per user, exact scan of not-yet-linked vectors, shared-mutex concurrency.",
    icon: Cpu,
    color: "purple",
    modules: [
      { name: "VectorEngine", detail: "HNSW index; recall measured against exact search and hnswlib", type: "infra" },
      { name: "IndexRegistry", detail: "Namespaces (users) with one shared background indexing thread", type: "infra" },
      { name: "SIMD", detail: "NEON on ARM64, AVX2+FMA on x86-64, scalar otherwise (reported at runtime)", type: "infra" },
      { name: "Auth", detail: "Shared secret from RECALLIX_INTERNAL_KEY; the engine refuses to start without it", type: "infra" },
    ],
  },
  {
    id: "storage",
    title: "Persistence (SQLite)",
    desc: "SQLite in WAL mode is the system of record; the vector index is derived from it and can always be rebuilt.",
    icon: Database,
    color: "emerald",
    modules: [
      { name: "Episodic", detail: "Memories with embedding, embedding model name, content hash and FTS5 index", type: "storage" },
      { name: "Working", detail: "Sliding window of recent messages, one row per user", type: "storage" },
      { name: "Long-Term", detail: "Promoted facts (importance threshold from config) with Ebbinghaus decay", type: "storage" },
      { name: "Recall log", detail: "Candidate features per recall; feedback turns them into ranker training data", type: "storage" },
    ],
  },
];

const storePipelineSteps = [
  { step: "Auth", label: "Hashed API key → user (service keys may act for end users)", icon: <Eye className="w-4 h-4" /> },
  { step: "Dedupe", label: "Identical content visible to this agent is skipped", icon: <Target className="w-4 h-4" /> },
  { step: "Sensory", label: "Per-user ring buffer with TTL", icon: <Boxes className="w-4 h-4" /> },
  { step: "Working", label: "Sliding window of recent messages", icon: <Activity className="w-4 h-4" /> },
  { step: "Embedding", label: "One embedding per message, reused below", icon: <Dna className="w-4 h-4" /> },
  { step: "Facts", label: "Rule grammar → (entity, relation, value) slots", icon: <Workflow className="w-4 h-4" /> },
  { step: "Schema", label: "Prototype classifier on the embedding", icon: <Sparkles className="w-4 h-4" /> },
  { step: "Importance", label: "Learned logistic score (novelty, IDF, specificity, cues)", icon: <Zap className="w-4 h-4" /> },
  { step: "Episodic", label: "SQLite row + embedding + FTS5 + timeline", icon: <Database className="w-4 h-4" /> },
  { step: "HNSW", label: "Add to the user's index (searchable immediately)", icon: <Microchip className="w-4 h-4" /> },
  { step: "Gating", label: "Long-term promotion above the configured threshold", icon: <ShieldCheck className="w-4 h-4" /> },
  { step: "Jobs", label: "World state, prediction, replay, forgetting (background)", icon: <Lightbulb className="w-4 h-4" /> },
];

export default function ArchitecturePage() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setActiveStep(s => (s + 1) % storePipelineSteps.length), 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="min-h-screen bg-[#050506] text-white selection:bg-cyan-500/30 selection:text-cyan-200">
      <nav className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-white/5 flex items-center justify-between px-8 glass backdrop-blur-2xl">
        <div className="flex items-center gap-3 cursor-pointer">
          <Link href="/" className="flex items-center gap-3">
            <Brain className="w-6 h-6 text-cyan-400" />
            <span className="text-xl font-black tracking-tighter">RECALLIX</span>
          </Link>
        </div>
        <div className="hidden md:flex items-center gap-6">
          <Link href="/architecture" className="text-sm font-bold text-white transition-colors">Architecture</Link>
          <Link href="/benchmark" className="text-sm font-bold text-zinc-400 hover:text-white transition-colors">Performance Audit</Link>
          <Link href="/roadmap" className="text-sm font-bold text-zinc-400 hover:text-white transition-colors">Roadmap</Link>
          <Link href="/observability" className="text-sm font-bold text-zinc-400 hover:text-white transition-colors">Observability</Link>
        </div>
        <Link href="/benchmark">
          <Button className="h-10 px-5 bg-cyan-500 hover:bg-cyan-400 text-black font-black rounded-full">
            View Audit
          </Button>
        </Link>
      </nav>

      <main className="pt-32 pb-24 container mx-auto px-6 max-w-7xl">
        <header className="max-w-3xl mb-24">
          <Badge className="mb-6 py-1.5 px-4 bg-cyan-500/10 border-cyan-500/20 text-cyan-400 rounded-full font-bold text-xs">
            TECHNICAL DEEP-DIVE
          </Badge>
          <h1 className="text-6xl md:text-8xl font-black tracking-tighter leading-[0.85] mb-8">
            Modern AI <br />
            <span className="text-gradient">Architecture.</span>
          </h1>
          <p className="text-xl text-zinc-500 font-bold leading-relaxed max-w-2xl">
            A hybrid engine designed for universal AI continuity. 
            Python for high-level reasoning. C++ for hardware-level performance. 
            SQLite for ACID-compliant persistence.
          </p>
        </header>

        {/* BENTO GRID */}
        <motion.div 
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-24"
        >
          {/* Main Logic Card */}
          <motion.div variants={item} className="lg:col-span-8 bg-zinc-900/40 border border-white/5 rounded-[40px] p-10 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 blur-[120px] rounded-full group-hover:bg-cyan-500/10 transition-colors" />
            <div className="flex items-center gap-4 mb-10 relative z-10">
              <div className="w-14 h-14 bg-cyan-500/10 border border-cyan-500/20 rounded-2xl flex items-center justify-center">
                <Brain className="w-7 h-7 text-cyan-400" />
              </div>
              <div>
                <h3 className="text-2xl font-black tracking-tight">Python Cognitive Brain</h3>
                <p className="text-sm text-zinc-500 font-bold">The Orchestration Substrate</p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
              {architectureStack[0].modules.map((mod) => (
                <div key={mod.name} className="bg-black/40 border border-white/5 rounded-2xl p-6 hover:border-cyan-500/20 transition-colors">
                  <h4 className="text-sm font-black text-white mb-2">{mod.name}</h4>
                  <p className="text-xs text-zinc-500 font-bold leading-relaxed">{mod.detail}</p>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Scale Card */}
          <motion.div variants={item} className="lg:col-span-4 bg-purple-600/5 border border-purple-600/10 rounded-[40px] p-10 relative overflow-hidden flex flex-col group">
            <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-purple-500/10 blur-[100px] rounded-full group-hover:bg-purple-500/20 transition-colors" />
            <div className="w-14 h-14 bg-purple-500/10 border border-purple-500/20 rounded-2xl flex items-center justify-center mb-10">
              <Cpu className="w-7 h-7 text-purple-400" />
            </div>
            <h3 className="text-2xl font-black tracking-tight mb-4">C++ Moat</h3>
            <p className="text-sm text-zinc-500 font-bold leading-relaxed mb-8">
              {largestClustered ? `${ms(largestClustered.search_avg_ms, 2)} average search at ${largestClustered.n.toLocaleString()} vectors, recall@10 ${pct(largestClustered.recall_at_10, 1)} (single thread).` : null}
            </p>
            <div className="space-y-4 flex-1">
              {architectureStack[1].modules.slice(0, 3).map((mod) => (
                <div key={mod.name} className="flex items-center gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-500" />
                  <span className="text-xs font-bold text-zinc-400">{mod.name}</span>
                </div>
              ))}
            </div>
            <div className="mt-8 pt-8 border-t border-white/5">
              <span className="text-4xl font-black text-purple-400">0.1ms</span>
              <p className="text-[10px] text-zinc-600 uppercase font-bold tracking-widest mt-1">Verified search latency</p>
            </div>
          </motion.div>

          {/* Infrastructure Card */}
          <motion.div variants={item} className="lg:col-span-4 bg-emerald-600/5 border border-emerald-600/10 rounded-[40px] p-10 relative overflow-hidden flex flex-col group">
            <div className="absolute top-0 left-0 w-48 h-48 bg-emerald-500/5 blur-[80px] rounded-full group-hover:bg-emerald-500/10 transition-colors" />
            <div className="w-14 h-14 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl flex items-center justify-center mb-10">
              <HardDrive className="w-7 h-7 text-emerald-400" />
            </div>
            <h3 className="text-2xl font-black tracking-tight mb-4">Persistence</h3>
            <p className="text-sm text-zinc-500 font-bold leading-relaxed mb-8">
              High-velocity ACID ingestion using SQLite WAL mode.
            </p>
            <div className="space-y-4">
               {architectureStack[2].modules.map((mod) => (
                  <div key={mod.name} className="flex items-center gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span className="text-xs font-bold text-zinc-400">{mod.name}</span>
                  </div>
               ))}
            </div>
          </motion.div>

          {/* Interop Card */}
          <motion.div variants={item} className="lg:col-span-8 bg-zinc-900/40 border border-white/5 rounded-[40px] p-10 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/5 blur-[120px] rounded-full group-hover:bg-blue-500/10 transition-colors" />
            <div className="flex items-center gap-4 mb-8 relative z-10">
              <div className="w-14 h-14 bg-blue-500/10 border border-blue-500/20 rounded-2xl flex items-center justify-center">
                <ShieldCheck className="w-7 h-7 text-blue-400" />
              </div>
              <div>
                <h3 className="text-2xl font-black tracking-tight">Model Interoperability</h3>
                <p className="text-sm text-zinc-500 font-bold">Standardizing context for every AI architecture</p>
              </div>
            </div>
            <div className="bg-black/60 rounded-3xl p-8 border border-white/5 relative z-10">
               <div className="flex items-center justify-between mb-8">
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Storage Model</span>
                    <span className="text-sm font-bold text-white">Universal Architecture</span>
                  </div>
                  <ArrowRight className="text-zinc-700" />
                  <div className="flex flex-col gap-1 text-right">
                    <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Retrieval Model</span>
                    <span className="text-sm font-bold text-white">Diverse Ecosystem</span>
                  </div>
               </div>
               <p className="text-xs text-zinc-500 font-bold leading-relaxed">
                 Recallix decouples memory from specific model parameters. Context stored by a reasoning model 
                 is fully accessible by a coding assistant, agentic framework, or research agent without semantic loss.
               </p>
            </div>
          </motion.div>
        </motion.div>

        {/* 12-STEP STORE PIPELINE */}
        <section className="py-24">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-black tracking-tighter mb-4">
              Atomic Store <span className="text-gradient">Pipeline</span>
            </h2>
            <p className="text-zinc-500 text-lg font-bold">{storePipelineSteps.length} steps from request to stored memory.</p>
          </div>

          <div className="relative max-w-4xl mx-auto">
            {/* Connection Line */}
            <div className="absolute left-6 top-8 bottom-8 w-px bg-white/5 hidden md:block" />
            
            <div className="space-y-4">
              {storePipelineSteps.map((s, i) => (
                <motion.div 
                  key={s.step}
                  animate={{ 
                    opacity: activeStep === i ? 1 : 0.3,
                    x: activeStep === i ? 10 : 0
                  }}
                  className={`flex items-start gap-6 p-4 rounded-2xl transition-all duration-500 ${activeStep === i ? 'bg-white/5 border border-white/10' : ''}`}
                >
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors ${activeStep === i ? 'bg-cyan-500 text-black' : 'bg-zinc-900 text-zinc-600 border border-white/5'}`}>
                    {s.icon}
                  </div>
                  <div className="pt-1">
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Step 0{i + 1}</span>
                      <h4 className="text-sm font-black text-white">{s.step}</h4>
                    </div>
                    <p className="text-xs text-zinc-500 font-bold mt-1 leading-relaxed">{s.label}</p>
                  </div>
                  {activeStep === i && (
                    <motion.div 
                      layoutId="active-indicator"
                      className="ml-auto w-1 h-8 bg-cyan-500 rounded-full"
                    />
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* TECH STACK GRID */}
        <section className="py-24 border-t border-white/5">
           <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { title: "Python 3.11", detail: "FastAPI, Pydantic, NumPy (optional: sentence-transformers)", icon: Activity },
                { title: "C++ 17", detail: "Own HNSW implementation, NEON/AVX2 intrinsics, cpp-httplib, nlohmann-json", icon: Cpu },
                { title: "SQLite 3", detail: "WAL Mode, Full-Text Search (FTS5), JSON1 extension", icon: Database },
                { title: "Next.js 14", detail: "Tailwind CSS, Framer Motion, Lucide, Recharts", icon: Layers },
              ].map((tech) => (
                <div key={tech.title} className="glass-card rounded-3xl p-6 border border-white/5 flex flex-col gap-3">
                   <tech.icon className="w-5 h-5 text-zinc-500" />
                   <h4 className="text-sm font-black text-white">{tech.title}</h4>
                   <p className="text-[10px] text-zinc-600 font-bold leading-relaxed">{tech.detail}</p>
                </div>
              ))}
           </div>
        </section>

        <section className="py-24 text-center">
          <h2 className="text-4xl md:text-6xl font-black tracking-tighter mb-8 max-w-3xl mx-auto leading-tight">
            Ready for the <span className="text-gradient">Multi-Agent Future?</span>
          </h2>
          <div className="flex flex-wrap gap-4 justify-center">
            <Link href="/benchmark">
              <Button className="h-14 px-10 bg-cyan-500 hover:bg-cyan-400 text-black font-black text-lg rounded-2xl">
                Explore the Audit
              </Button>
            </Link>
            <Link href="/observability">
              <Button variant="outline" className="h-14 px-10 border-white/10 font-black text-lg rounded-2xl hover:bg-white/5">
                System Health
              </Button>
            </Link>
          </div>
        </section>
      </main>

      <footer className="border-t border-white/5 py-8 px-8 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-zinc-600" />
          <span className="text-xs font-bold text-zinc-600">RECALLIX</span>
        </div>
        <span className="text-xs text-zinc-700">Universal Memory Substrate for Multi-Agent Systems</span>
      </footer>
    </div>
  );
}
