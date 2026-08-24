"use client";

import { AnalysisResult } from "@/types";
import { useState } from "react";
import OcrComparisonPanel from "@/components/OcrComparisonPanel";

interface ResultsPanelProps {
  results: AnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;
  tesseractStatus: "idle" | "running" | "done" | "error";
  trocrStatus: "idle" | "running" | "done" | "error";
}

type Tab = "ocr" | "search" | "text";

export default function ResultsPanel({
  results,
  isAnalyzing,
  error,
  tesseractStatus,
  trocrStatus,
}: ResultsPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>("ocr");

  // ── Loading skeleton ───────────────────────────────────────────────────────
  if (isAnalyzing && !results) {
    return (
      <div className="h-full flex flex-col">
        <ResultsTabs activeTab={activeTab} onTabChange={setActiveTab} />
        <div className="flex-1 p-4 space-y-4 overflow-y-auto">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="animate-pulse space-y-2">
              <div className="h-5 bg-[var(--surface)] rounded w-1/3" />
              <div className="h-28 bg-[var(--surface)] rounded-xl" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── Error state ────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="text-center space-y-3">
          <div className="w-12 h-12 mx-auto rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
            <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-sm text-[var(--text-secondary)]">{error}</p>
        </div>
      </div>
    );
  }

  // ── Empty state ────────────────────────────────────────────────────────────
  if (!results && tesseractStatus === "idle" && trocrStatus === "idle") {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="text-center space-y-3">
          <svg className="w-16 h-16 mx-auto text-[var(--text-secondary)] opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
              d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
          <p className="text-sm font-medium text-[var(--text-secondary)]">Draw a selection on the image</p>
          <p className="text-xs text-[var(--text-secondary)]">Both OCR engines will run in parallel and compare results</p>
        </div>
      </div>
    );
  }

  // ── Results ────────────────────────────────────────────────────────────────
  const engines = results?.ocrResults ?? [];
  const isRunning = tesseractStatus === "running" || trocrStatus === "running";

  return (
    <div className="h-full flex flex-col">
      <ResultsTabs activeTab={activeTab} onTabChange={setActiveTab} hasOcr={engines.length > 0 || isRunning} />

      <div className="flex-1 overflow-y-auto">
        {activeTab === "ocr" && (
          <OcrComparisonPanel
            engines={engines}
            tesseractStatus={tesseractStatus}
            trocrStatus={trocrStatus}
          />
        )}
        {activeTab === "search" && results && <SearchResults results={results} />}
        {activeTab === "text" && results && <RawTextResults results={results} />}
      </div>
    </div>
  );
}

// ─── Tabs ──────────────────────────────────────────────────────────────────────

function ResultsTabs({
  activeTab,
  onTabChange,
  hasOcr,
}: {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  hasOcr?: boolean;
}) {
  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    {
      id: "ocr",
      label: "OCR Compare",
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
      ),
    },
    {
      id: "search",
      label: "Search",
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      ),
    },
    {
      id: "text",
      label: "Raw Text",
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
  ];

  return (
    <div className="flex border-b border-[var(--border)] px-2">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`flex items-center gap-1.5 px-4 py-3 text-sm font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === tab.id
              ? "border-[var(--primary)] text-[var(--primary)]"
              : "border-transparent text-[var(--text-secondary)] hover:text-[var(--foreground)]"
          }`}
        >
          {tab.icon}
          {tab.label}
          {tab.id === "ocr" && hasOcr && (
            <span className="ml-1 w-1.5 h-1.5 rounded-full bg-[var(--primary)] inline-block" />
          )}
        </button>
      ))}
    </div>
  );
}

// ─── Search Results tab ────────────────────────────────────────────────────────

function SearchResults({ results }: { results: AnalysisResult }) {
  const topLabel = results.labels[0];
  const alternates = results.labels.slice(1, 6);
  const isLowConfidence = !topLabel || topLabel.confidence < 0.4;

  return (
    <div className="p-4 space-y-5">
      {topLabel && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-full bg-[var(--primary)]/10 flex items-center justify-center">
              <svg className="w-3 h-3 text-[var(--primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="text-xs text-[var(--text-secondary)] uppercase tracking-wider font-medium">
              {isLowConfidence ? "Best guess" : "Identified"}
            </span>
          </div>

          <a
            href={`https://www.google.com/search?q=${encodeURIComponent(topLabel.name)}`}
            target="_blank"
            rel="noopener noreferrer"
            className="block p-4 bg-[var(--surface)] rounded-xl border border-[var(--border)] hover:border-[var(--primary)] transition-all group"
          >
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-lg group-hover:text-[var(--primary)] transition-colors">
                {topLabel.name}
              </h3>
              <span
                className={`text-xs px-2 py-1 rounded-full font-medium ${
                  topLabel.confidence > 0.6
                    ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                    : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                }`}
              >
                {Math.round(topLabel.confidence * 100)}%
              </span>
            </div>
          </a>
        </div>
      )}

      {alternates.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs text-[var(--text-secondary)] uppercase tracking-wider font-medium">
            Other possibilities
          </h4>
          <div className="space-y-2">
            {alternates.map((label, i) => (
              <a
                key={i}
                href={`https://www.google.com/search?q=${encodeURIComponent(label.name)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between p-3 bg-[var(--surface)]/60 border border-[var(--border)] rounded-lg hover:border-[var(--primary)] hover:text-[var(--primary)] transition-colors"
              >
                <span className="text-sm font-medium">{label.name}</span>
                <span className="text-xs text-[var(--text-secondary)]">
                  {Math.round(label.confidence * 100)}%
                </span>
              </a>
            ))}
          </div>
        </div>
      )}

      {results.searchResults.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs text-[var(--text-secondary)] uppercase tracking-wider font-medium">
            Search links
          </h4>
          <div className="space-y-2">
            {results.searchResults.map((result, i) => (
              <a
                key={i}
                href={result.link}
                target="_blank"
                rel="noopener noreferrer"
                className="block p-3 border border-[var(--border)] rounded-xl hover:border-[var(--primary)] hover:shadow-sm transition-all"
              >
                <p className="font-medium text-sm">{result.title}</p>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5 truncate">{result.snippet}</p>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Raw Text tab ──────────────────────────────────────────────────────────────

function RawTextResults({ results }: { results: AnalysisResult }) {
  if (!results.textContent) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="text-center space-y-3">
          <svg className="w-12 h-12 mx-auto text-[var(--text-secondary)] opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-sm text-[var(--text-secondary)]">No text detected in this image</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => navigator.clipboard.writeText(results.textContent || "")}
          className="inline-flex items-center gap-1.5 px-4 py-2 bg-[var(--primary)] text-white rounded-full text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Copy best result
        </button>
        <a
          href={`https://www.google.com/search?q=${encodeURIComponent(results.textContent)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 px-4 py-2 border border-[var(--border)] rounded-full text-sm font-medium hover:bg-[var(--surface)] transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          Search
        </a>
      </div>

      <div className="p-4 bg-[var(--surface)] rounded-xl border border-[var(--border)]">
        <h4 className="text-sm font-semibold mb-2">Best detected text</h4>
        <pre className="text-sm whitespace-pre-wrap font-mono leading-relaxed text-[var(--foreground)]">
          {results.textContent}
        </pre>
      </div>
    </div>
  );
}
