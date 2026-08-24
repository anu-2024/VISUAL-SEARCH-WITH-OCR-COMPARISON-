"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import ImageUploader from "@/components/ImageUploader";
import CameraCapture from "@/components/CameraCapture";
import ImageWithSelection from "@/components/ImageWithSelection";
import ResultsPanel from "@/components/ResultsPanel";
import { analyzeImage, preloadModel } from "@/lib/analyzer";
import { AnalysisResult, OcrEngineStatus } from "@/types";
import { CropRegion } from "@/components/ImageWithSelection";

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [showCamera, setShowCamera] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ocrStatus, setOcrStatus] = useState<OcrEngineStatus>({
    tesseract: "idle",
    trocr: "idle",
  });

  // Ref to track abort so we can ignore stale results
  const analysisVersion = useRef(0);

  // Preload ML models in background on mount
  useEffect(() => {
    preloadModel();
  }, []);

  const handleImageSelected = useCallback((file: File) => {
    setError(null);
    setResults(null);
    setOcrStatus({ tesseract: "idle", trocr: "idle" });
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setShowCamera(false);
  }, []);

  const handleRegionSelected = useCallback(
    async (_region: CropRegion, croppedDataUrl: string) => {
      setError(null);
      setResults(null);
      setIsAnalyzing(true);

      // Mark both engines as running immediately
      setOcrStatus({ tesseract: "running", trocr: "running" });

      const version = ++analysisVersion.current;

      try {
        const data = await analyzeImage(croppedDataUrl);

        // Ignore stale results if a newer analysis was started
        if (version !== analysisVersion.current) return;

        setResults(data);

        // Mark individual engines done/error based on their result
        const tessResult = data.ocrResults.find((r) => r.id === "tesseract");
        const trocrResult = data.ocrResults.find((r) => r.id === "trocr");
        setOcrStatus({
          tesseract: tessResult?.error ? "error" : "done",
          trocr: trocrResult?.error ? "error" : "done",
        });
      } catch (err) {
        if (version !== analysisVersion.current) return;
        setError(err instanceof Error ? err.message : "Analysis failed");
        setOcrStatus({ tesseract: "error", trocr: "error" });
      } finally {
        if (version === analysisVersion.current) {
          setIsAnalyzing(false);
        }
      }
    },
    []
  );

  const handleReset = useCallback(() => {
    analysisVersion.current++;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setResults(null);
    setError(null);
    setShowCamera(false);
    setIsAnalyzing(false);
    setOcrStatus({ tesseract: "idle", trocr: "idle" });
  }, [imageUrl]);

  // ── Upload/Camera screen ───────────────────────────────────────────────────
  if (!imageUrl && !showCamera) {
    return (
      <main className="min-h-screen flex flex-col items-center justify-center px-4 py-8">
        <div className="w-full max-w-xl space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-[#4285f4] via-[#34a853] to-[#fbbc05]">
              <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <h1 className="text-2xl md:text-3xl font-medium text-[var(--foreground)]">
              Visual Search + OCR Compare
            </h1>
            <p className="text-[var(--text-secondary)]">
              Upload an image to run{" "}
              <span className="font-medium text-[var(--foreground)]">Tesseract.js</span> and{" "}
              <span className="font-medium text-[var(--foreground)]">TrOCR</span> simultaneously —
              compare speed, accuracy, and detected text side-by-side
            </p>
          </div>

          {/* Engine pills */}
          <div className="flex justify-center gap-3 flex-wrap">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--surface)] border border-[var(--border)] rounded-full text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              Tesseract.js · Classic LSTM
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--surface)] border border-[var(--border)] rounded-full text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-purple-500" />
              TrOCR · Vision Transformer
            </div>
          </div>

          {/* Upload */}
          <ImageUploader onImageSelected={handleImageSelected} />

          {/* Camera */}
          <div className="flex items-center justify-center gap-4">
            <div className="h-px flex-1 bg-[var(--border)]" />
            <span className="text-xs text-[var(--text-secondary)] uppercase">or</span>
            <div className="h-px flex-1 bg-[var(--border)]" />
          </div>

          <button
            onClick={() => setShowCamera(true)}
            className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-[var(--surface)] border border-[var(--border)] rounded-2xl hover:bg-[var(--border)]/50 transition-colors cursor-pointer"
          >
            <svg className="w-6 h-6 text-[var(--primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span className="font-medium">Take a photo</span>
          </button>

          <div className="text-center">
            <span className="inline-flex items-center gap-1.5 text-xs text-[var(--text-secondary)] px-3 py-1.5 bg-[var(--surface)] rounded-full">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
              100% private · All processing happens in your browser
            </span>
          </div>
        </div>
      </main>
    );
  }

  // ── Camera screen ──────────────────────────────────────────────────────────
  if (showCamera && !imageUrl) {
    return (
      <CameraCapture
        onCapture={handleImageSelected}
        onClose={() => setShowCamera(false)}
      />
    );
  }

  // ── Results screen ─────────────────────────────────────────────────────────
  return (
    <main className="min-h-screen flex flex-col">
      {/* Top bar */}
      <header className="sticky top-0 z-40 bg-[var(--background)] border-b border-[var(--border)] px-4 h-14 flex items-center gap-4">
        <button
          onClick={handleReset}
          className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-[var(--foreground)] transition-colors cursor-pointer"
          aria-label="Back"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>

        <div className="flex items-center gap-2 flex-1">
          <div className="w-6 h-6 rounded-full bg-gradient-to-br from-[#4285f4] via-[#34a853] to-[#fbbc05] flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <span className="text-sm font-medium">Visual Search</span>
        </div>

        {/* Live engine status pills in header */}
        {(isAnalyzing || results) && (
          <div className="flex gap-2">
            <HeaderStatusPill label="Tesseract" status={ocrStatus.tesseract} color="blue" />
            <HeaderStatusPill label="TrOCR" status={ocrStatus.trocr} color="purple" />
          </div>
        )}
      </header>

      {/* Main content */}
      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left: Image with selection */}
        <div className="lg:w-1/2 lg:max-w-[600px] lg:border-r border-[var(--border)] bg-[#202124] lg:sticky lg:top-14 lg:h-[calc(100vh-3.5rem)] flex items-center justify-center p-4">
          {imageUrl && (
            <ImageWithSelection
              imageUrl={imageUrl}
              onRegionSelected={handleRegionSelected}
              isAnalyzing={isAnalyzing}
            />
          )}
        </div>

        {/* Right: Results */}
        <div className="flex-1 lg:h-[calc(100vh-3.5rem)] lg:overflow-y-auto">
          <ResultsPanel
            results={results}
            isAnalyzing={isAnalyzing}
            error={error}
            tesseractStatus={ocrStatus.tesseract}
            trocrStatus={ocrStatus.trocr}
          />
        </div>
      </div>
    </main>
  );
}

// Small status pill shown in the top header bar
function HeaderStatusPill({
  label,
  status,
  color,
}: {
  label: string;
  status: "idle" | "running" | "done" | "error";
  color: "blue" | "purple";
}) {
  const colorMap = {
    blue: {
      running: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
      done: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
      error: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
      idle: "bg-[var(--surface)] text-[var(--text-secondary)]",
    },
    purple: {
      running: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
      done: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
      error: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
      idle: "bg-[var(--surface)] text-[var(--text-secondary)]",
    },
  };

  const cls = colorMap[color][status];
  const icon =
    status === "running" ? "●" : status === "done" ? "✓" : status === "error" ? "✗" : "○";

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
      <span className={status === "running" ? "animate-pulse" : ""}>{icon}</span>
      {label}
    </span>
  );
}
