"use client";

import { useState } from "react";
import { OcrEngineResult } from "@/types";

interface OcrComparisonPanelProps {
  engines: OcrEngineResult[];
  tesseractStatus: "idle" | "running" | "done" | "error";
  trocrStatus: "idle" | "running" | "done" | "error";
}

function StatusBadge({ status }: { status: "idle" | "running" | "done" | "error" }) {
  if (status === "running") {
    return (
      <span className="ocr-badge ocr-badge-running">
        <span className="ocr-spinner" />
        Running…
      </span>
    );
  }
  if (status === "done") {
    return <span className="ocr-badge ocr-badge-done">✓ Done</span>;
  }
  if (status === "error") {
    return <span className="ocr-badge ocr-badge-error">✗ Error</span>;
  }
  return <span className="ocr-badge ocr-badge-idle">Idle</span>;
}

export default function OcrComparisonPanel({
  engines,
  tesseractStatus,
  trocrStatus,
}: OcrComparisonPanelProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const statuses: Record<string, "idle" | "running" | "done" | "error"> = {
    tesseract: tesseractStatus,
    trocr: trocrStatus,
  };

  const copyText = (id: string, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1800);
    });
  };

  return (
    <div className="ocr-panel">
      <div className="ocr-header">
        <div className="ocr-header-title">
          <svg className="ocr-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <h2>OCR Engine Comparison</h2>
        </div>
        <p className="ocr-header-sub">Both models run simultaneously — results update as each finishes</p>
      </div>

      <div className="ocr-cards-grid">
        {engines.map((engine) => {
          const status = statuses[engine.id];
          const isRunningThis = status === "running";

          return (
            <div
              key={engine.id}
              className={`ocr-card ${isRunningThis ? "ocr-card-running" : ""} ${
                engine.id === "tesseract" ? "ocr-card-trocr" : "ocr-card-tesseract"
              }`}
            >
              <div className="ocr-card-header">
                <div className="ocr-card-title-row">
                  <div className={`ocr-engine-icon ${engine.id === "tesseract" ? "ocr-engine-icon-trocr" : "ocr-engine-icon-tesseract"}`}>
                    {engine.id === "tesseract" ? (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    )}
                  </div>
                  <div>
                    <p className="ocr-engine-name">{engine.name}</p>
                    <p className="ocr-engine-desc">{engine.description}</p>
                  </div>
                </div>
                <StatusBadge status={status} />
              </div>

              <div className={`ocr-text-box ${isRunningThis ? "ocr-text-box-running" : ""}`}>
                {isRunningThis ? (
                  <div className="ocr-text-shimmer">
                    <div className="ocr-shimmer-line ocr-shimmer-line-wide" />
                    <div className="ocr-shimmer-line ocr-shimmer-line-mid" />
                    <div className="ocr-shimmer-line ocr-shimmer-line-short" />
                  </div>
                ) : engine.error ? (
                  <p className="ocr-text-error">{engine.error}</p>
                ) : engine.text ? (
                  <pre className="ocr-text-content">{engine.text}</pre>
                ) : (
                  <p className="ocr-text-empty">No text detected</p>
                )}
              </div>

              {engine.text && !isRunningThis && (
                <button
                  className="ocr-copy-btn"
                  onClick={() => copyText(engine.id, engine.text!)}
                >
                  {copiedId === engine.id ? (
                    <>
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy
                    </>
                  )}
                </button>
              )}
            </div>
          );
        })}
      </div>

      {!engines.some((e) => e.text) && tesseractStatus !== "running" && trocrStatus !== "running" && (
        <div className="ocr-empty">
          <svg className="ocr-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="ocr-empty-title">No text detected</p>
          <p className="ocr-empty-sub">Try selecting a region with visible text in the image</p>
        </div>
      )}
    </div>
  );
}
