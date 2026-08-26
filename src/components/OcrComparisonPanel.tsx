"use client";

import { useMemo, useState } from "react";
import { OcrEngineResult } from "@/types";

interface OcrComparisonPanelProps {
  engines: OcrEngineResult[];
  tesseractStatus: "idle" | "running" | "done" | "error";
  trocrStatus: "idle" | "running" | "done" | "error";
}

// ─── helpers ──────────────────────────────────────────────────────────────────

function normalize(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
}

function levenshtein(a: string, b: string): number {
  const prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(
        current[j - 1] + 1,
        prev[j] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
    for (let j = 0; j <= b.length; j++) prev[j] = current[j];
  }
  return prev[b.length];
}

function cer(truth: string, predicted: string): number {
  if (!truth) return 0;
  const dist = levenshtein(truth, predicted);
  return Math.max(0, Math.round((1 - dist / Math.max(truth.length, 1)) * 100));
}

// ─── sub-components ───────────────────────────────────────────────────────────

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

function MetricBar({
  label,
  value,
  max,
  unit,
  highlight,
}: {
  label: string;
  value: number;
  max: number;
  unit: string;
  highlight: boolean;
}) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="ocr-metric-bar-row">
      <span className="ocr-metric-label">{label}</span>
      <div className="ocr-metric-track">
        <div
          className={`ocr-metric-fill ${highlight ? "ocr-metric-fill-best" : ""}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`ocr-metric-value ${highlight ? "ocr-metric-value-best" : ""}`}>
        {value.toLocaleString()} {unit}
      </span>
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

export default function OcrComparisonPanel({
  engines,
  tesseractStatus,
  trocrStatus,
}: OcrComparisonPanelProps) {
  const [groundTruth, setGroundTruth] = useState("");
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

  // Derived comparison metrics
  const hasAnyText = engines.some((e) => e.text);

  const fastestId = useMemo(() => {
    const withText = engines.filter((e) => e.text);
    if (!withText.length) return null;
    return withText.sort((a, b) => a.timeMs - b.timeMs)[0].id;
  }, [engines]);

  const mostWordsId = useMemo(() => {
    const withText = engines.filter((e) => e.text);
    if (!withText.length) return null;
    return withText.sort((a, b) => b.wordCount - a.wordCount)[0].id;
  }, [engines]);

  const highestConfId = useMemo(() => {
    const withConf = engines.filter((e) => e.confidence >= 0);
    if (!withConf.length) return null;
    return withConf.sort((a, b) => b.confidence - a.confidence)[0].id;
  }, [engines]);

  const maxTime = useMemo(() => Math.max(...engines.map((e) => e.timeMs), 1), [engines]);
  const maxWords = useMemo(() => Math.max(...engines.map((e) => e.wordCount), 1), [engines]);
  const maxChars = useMemo(() => Math.max(...engines.map((e) => e.charCount), 1), [engines]);

  const cerScores = useMemo(() => {
    const truth = normalize(groundTruth);
    if (!truth) return null;
    return Object.fromEntries(
      engines.map((e) => [e.id, cer(truth, normalize(e.text ?? ""))])
    );
  }, [groundTruth, engines]);

  const bestCerId = useMemo(() => {
    if (!cerScores) return null;
    return Object.entries(cerScores).sort(([, a], [, b]) => b - a)[0]?.[0] ?? null;
  }, [cerScores]);

  const isRunning = tesseractStatus === "running" || trocrStatus === "running";

  return (
    <div className="ocr-panel">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="ocr-header">
        <div className="ocr-header-title">
          <svg className="ocr-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <h2>OCR Engine Comparison</h2>
          {isRunning && <span className="ocr-live-badge">● LIVE</span>}
        </div>
        <p className="ocr-header-sub">Both models run simultaneously — results update as each finishes</p>
      </div>

      {/* ── Engine Cards ───────────────────────────────────────────────── */}
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
              {/* Card header */}
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

              {/* Badges row */}
              <div className="ocr-badges-row">
                {engine.id === fastestId && (
                  <span className="ocr-winner-badge ocr-winner-speed">⚡ Fastest</span>
                )}
                {engine.id === mostWordsId && (
                  <span className="ocr-winner-badge ocr-winner-words">📝 Most Text</span>
                )}
                {engine.id === highestConfId && engine.confidence >= 0 && (
                  <span className="ocr-winner-badge ocr-winner-conf">🎯 High Confidence</span>
                )}
                {cerScores && engine.id === bestCerId && (
                  <span className="ocr-winner-badge ocr-winner-cer">🏆 Best Accuracy</span>
                )}
              </div>

              {/* Metrics chips */}
              <div className="ocr-chips">
                <div className="ocr-chip">
                  <span className="ocr-chip-label">⏱ Time</span>
                  <span className="ocr-chip-value">{engine.timeMs.toLocaleString()} ms</span>
                </div>
                <div className="ocr-chip">
                  <span className="ocr-chip-label">📝 Words</span>
                  <span className="ocr-chip-value">{engine.wordCount}</span>
                </div>
                <div className="ocr-chip">
                  <span className="ocr-chip-label">🔤 Chars</span>
                  <span className="ocr-chip-value">{engine.charCount}</span>
                </div>
                {engine.confidence >= 0 && (
                  <div className="ocr-chip">
                    <span className="ocr-chip-label">🎯 Conf.</span>
                    <span className="ocr-chip-value">{engine.confidence}%</span>
                  </div>
                )}
                {cerScores && (
                  <div className={`ocr-chip ${engine.id === bestCerId ? "ocr-chip-highlight" : ""}`}>
                    <span className="ocr-chip-label">✅ CER Acc.</span>
                    <span className="ocr-chip-value">{cerScores[engine.id]}%</span>
                  </div>
                )}
              </div>

              {/* Detected text output */}
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

              {/* Copy button */}
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

      {/* ── Comparison Bar Chart ────────────────────────────────────────── */}
      {hasAnyText && !isRunning && (
        <div className="ocr-chart-section">
          <h3 className="ocr-section-title">Performance Comparison</h3>
          <div className="ocr-chart-block">
            <p className="ocr-chart-subtitle">Processing Time (lower is better)</p>
            {engines.map((e) => (
              <MetricBar
                key={e.id}
                label={e.name}
                value={e.timeMs}
                max={maxTime}
                unit="ms"
                highlight={e.id === fastestId}
              />
            ))}
          </div>
          <div className="ocr-chart-block">
            <p className="ocr-chart-subtitle">Words Detected (higher is better)</p>
            {engines.map((e) => (
              <MetricBar
                key={e.id}
                label={e.name}
                value={e.wordCount}
                max={maxWords}
                unit="words"
                highlight={e.id === mostWordsId}
              />
            ))}
          </div>
          <div className="ocr-chart-block">
            <p className="ocr-chart-subtitle">Characters Detected</p>
            {engines.map((e) => (
              <MetricBar
                key={e.id}
                label={e.name}
                value={e.charCount}
                max={maxChars}
                unit="chars"
                highlight={e.id === mostWordsId}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Ground Truth / CER Section ──────────────────────────────────── */}
      {hasAnyText && !isRunning && (
        <div className="ocr-cer-section">
          <div className="ocr-cer-header">
            <div>
              <h3 className="ocr-section-title">Accuracy Calculator (CER)</h3>
              <p className="ocr-cer-desc">
                Paste the actual text from the image to calculate Character Error Rate accuracy for both engines.
              </p>
            </div>
          </div>
          <textarea
            value={groundTruth}
            onChange={(e) => setGroundTruth(e.target.value)}
            placeholder="Type or paste the actual text that appears in the image…"
            rows={3}
            className="ocr-cer-textarea"
          />
          {cerScores && (
            <div className="ocr-cer-results">
              {engines.map((e) => (
                <div
                  key={e.id}
                  className={`ocr-cer-card ${e.id === bestCerId ? "ocr-cer-card-best" : ""}`}
                >
                  <p className="ocr-cer-engine-name">{e.name}</p>
                  <div className="ocr-cer-score-row">
                    <span className="ocr-cer-score">{cerScores[e.id]}%</span>
                    <span className="ocr-cer-label">accuracy</span>
                  </div>
                  <div className="ocr-cer-bar-track">
                    <div
                      className={`ocr-cer-bar-fill ${e.id === bestCerId ? "ocr-cer-bar-best" : ""}`}
                      style={{ width: `${cerScores[e.id]}%` }}
                    />
                  </div>
                  {e.id === bestCerId && (
                    <span className="ocr-cer-winner-label">🏆 More Accurate</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Winner Summary ──────────────────────────────────────────────── */}
      {hasAnyText && !isRunning && (
        <div className="ocr-summary">
          <h3 className="ocr-section-title">Summary</h3>
          <div className="ocr-summary-grid">
            {fastestId && (
              <div className="ocr-summary-card">
                <span className="ocr-summary-emoji">⚡</span>
                <div>
                  <p className="ocr-summary-metric">Fastest Engine</p>
                  <p className="ocr-summary-winner">
                    {engines.find((e) => e.id === fastestId)?.name}
                  </p>
                  <p className="ocr-summary-detail">
                    {engines.find((e) => e.id === fastestId)?.timeMs.toLocaleString()} ms
                  </p>
                </div>
              </div>
            )}
            {mostWordsId && (
              <div className="ocr-summary-card">
                <span className="ocr-summary-emoji">📝</span>
                <div>
                  <p className="ocr-summary-metric">Most Text</p>
                  <p className="ocr-summary-winner">
                    {engines.find((e) => e.id === mostWordsId)?.name}
                  </p>
                  <p className="ocr-summary-detail">
                    {engines.find((e) => e.id === mostWordsId)?.wordCount} words detected
                  </p>
                </div>
              </div>
            )}
            {highestConfId && (
              <div className="ocr-summary-card">
                <span className="ocr-summary-emoji">🎯</span>
                <div>
                  <p className="ocr-summary-metric">Highest Confidence</p>
                  <p className="ocr-summary-winner">
                    {engines.find((e) => e.id === highestConfId)?.name}
                  </p>
                  <p className="ocr-summary-detail">
                    {engines.find((e) => e.id === highestConfId)?.confidence}% avg confidence
                  </p>
                </div>
              </div>
            )}
            {bestCerId && (
              <div className="ocr-summary-card ocr-summary-card-highlight">
                <span className="ocr-summary-emoji">🏆</span>
                <div>
                  <p className="ocr-summary-metric">Best CER Accuracy</p>
                  <p className="ocr-summary-winner">
                    {engines.find((e) => e.id === bestCerId)?.name}
                  </p>
                  <p className="ocr-summary-detail">
                    {cerScores?.[bestCerId]}% character accuracy
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!hasAnyText && !isRunning && (
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
