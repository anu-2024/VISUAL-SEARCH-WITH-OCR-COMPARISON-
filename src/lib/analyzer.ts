import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import Tesseract from "tesseract.js";
import type { ImageToTextPipeline } from "@huggingface/transformers";
import { AnalysisResult, Label, OcrEngineResult, SearchResult } from "@/types";

// ─── Image Preprocessing Utilities ───────────────────────────────────────────
// TrOCR expects clean, high-contrast, single-line text images.
// These canvas helpers prepare the image before inference.

/** Convert to grayscale using luminance weights. */
function toGrayscale(imageData: ImageData): ImageData {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    d[i] = gray;
    d[i + 1] = gray;
    d[i + 2] = gray;
  }
  return imageData;
}

/** Otsu's method: compute optimal threshold for binarization. */
function otsuThreshold(imageData: ImageData): number {
  const hist = new Array(256).fill(0);
  const d = imageData.data;
  const total = d.length / 4;
  for (let i = 0; i < d.length; i += 4) hist[d[i]]++;

  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];

  let sumB = 0;
  let wB = 0;
  let maxVariance = 0;
  let threshold = 128;

  for (let i = 0; i < 256; i++) {
    wB += hist[i];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const variance = wB * wF * (mB - mF) * (mB - mF);
    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = i;
    }
  }
  return threshold;
}

/** Apply Otsu binarization. */
function binarize(imageData: ImageData): ImageData {
  const threshold = otsuThreshold(imageData);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] >= threshold ? 255 : 0;
    d[i] = v;
    d[i + 1] = v;
    d[i + 2] = v;
  }
  return imageData;
}

/** Simple 3x3 sharpen convolution. */
function sharpen(imageData: ImageData): ImageData {
  const w = imageData.width;
  const h = imageData.height;
  const src = new Uint8ClampedArray(imageData.data);
  const dst = imageData.data;
  const kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      for (let c = 0; c < 3; c++) {
        let val = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx = ((y + ky) * w + (x + kx)) * 4 + c;
            val += src[idx] * kernel[(ky + 1) * 3 + (kx + 1)];
          }
        }
        dst[(y * w + x) * 4 + c] = Math.min(255, Math.max(0, val));
      }
    }
  }
  return imageData;
}

/** Full preprocessing: grayscale → sharpen → binarize → clean up. */
function preprocessForOCR(
  imageSource: string,
  targetWidth = 384
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const scale = targetWidth / img.naturalWidth;
      const w = targetWidth;
      const h = Math.max(1, Math.round(img.naturalHeight * scale));

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d")!;

      // White background for binarization
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);

      let imageData = ctx.getImageData(0, 0, w, h);
      imageData = toGrayscale(imageData);
      imageData = sharpen(imageData);
      imageData = binarize(imageData);

      // Add 20px white padding around the text (helps TrOCR)
      const pad = 20;
      const padded = document.createElement("canvas");
      padded.width = w + pad * 2;
      padded.height = h + pad * 2;
      const pCtx = padded.getContext("2d")!;
      pCtx.fillStyle = "#ffffff";
      pCtx.fillRect(0, 0, padded.width, padded.height);
      pCtx.putImageData(imageData, pad, pad);

      resolve(padded.toDataURL("image/png"));
    };
    img.onerror = () => resolve(imageSource);
    img.src = imageSource;
  });
}

// ─── Text Line Segmentation ──────────────────────────────────────────────────
// TrOCR processes one line at a time. This splits multi-line regions into
// individual lines using horizontal projection profiling.

function segmentTextLines(
  imageSource: string
): Promise<string[]> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0);

      const imageData = ctx.getImageData(0, 0, w, h);
      const gray = toGrayscale(new ImageData(new Uint8ClampedArray(imageData.data), w, h));
      const bin = binarize(gray);
      const d = bin.data;

      // Horizontal projection: sum dark pixels per row
      const projection = new Array(h).fill(0);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          if (d[(y * w + x) * 4] === 0) projection[y]++;
        }
      }

      // Find line bands (rows with above-threshold dark pixels)
      const threshold = w * 0.01;
      const lines: { start: number; end: number }[] = [];
      let inLine = false;
      let lineStart = 0;

      for (let y = 0; y < h; y++) {
        if (!inLine && projection[y] > threshold) {
          inLine = true;
          lineStart = y;
        } else if (inLine && projection[y] <= threshold) {
          inLine = false;
          lines.push({ start: lineStart, end: y });
        }
      }
      if (inLine) lines.push({ start: lineStart, end: h });

      // Merge lines too close together (< 8px gap = same line)
      const merged: { start: number; end: number }[] = [];
      for (const line of lines) {
        if (merged.length > 0 && line.start - merged[merged.length - 1].end < 8) {
          merged[merged.length - 1].end = line.end;
        } else {
          merged.push({ ...line });
        }
      }

      // Need at least 2 lines to segment; otherwise return the whole image
      if (merged.length < 2) {
        resolve([imageSource]);
        return;
      }

      // Crop each line with padding
      const pad = 10;
      const croppedLines = merged.map((line) => {
        const cropY = Math.max(0, line.start - pad);
        const cropH = Math.min(h, line.end + pad) - cropY;
        const lineCanvas = document.createElement("canvas");
        lineCanvas.width = w;
        lineCanvas.height = cropH;
        const lCtx = lineCanvas.getContext("2d")!;
        lCtx.fillStyle = "#ffffff";
        lCtx.fillRect(0, 0, w, cropH);
        lCtx.drawImage(img, 0, cropY, w, cropH, 0, 0, w, cropH);
        return lineCanvas.toDataURL("image/png");
      });

      resolve(croppedLines);
    };
    img.onerror = () => resolve([imageSource]);
    img.src = imageSource;
  });
}

let model: mobilenet.MobileNet | null = null;
let modelLoading = false;
let modelLoadPromise: Promise<mobilenet.MobileNet> | null = null;

let detector: cocoSsd.ObjectDetection | null = null;
let detectorLoading = false;
let detectorLoadPromise: Promise<cocoSsd.ObjectDetection> | null = null;

// Load MobileNet model (cached after first load).
async function loadModel(): Promise<mobilenet.MobileNet> {
  if (model) return model;
  if (modelLoadPromise) return modelLoadPromise;

  modelLoading = true;
  modelLoadPromise = (async () => {
    await tf.ready();
    const loadedModel = await mobilenet.load({ version: 2, alpha: 1.0 });
    model = loadedModel;
    modelLoading = false;
    return loadedModel;
  })();

  return modelLoadPromise;
}

// Load COCO-SSD (cached after first load).
async function loadDetector(): Promise<cocoSsd.ObjectDetection> {
  if (detector) return detector;
  if (detectorLoadPromise) return detectorLoadPromise;

  detectorLoading = true;
  detectorLoadPromise = (async () => {
    await tf.ready();
    const loadedDetector = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    detector = loadedDetector;
    detectorLoading = false;
    return loadedDetector;
  })();

  return detectorLoadPromise;
}

export function isModelLoading(): boolean {
  return modelLoading || detectorLoading;
}

export function preloadModel(): void {
  loadDetector().catch(console.error);
  loadModel().catch(console.error);
  loadTrOCR().catch(console.error);
}

async function detectObjects(imageElement: HTMLImageElement): Promise<Label[]> {
  const objectDetector = await loadDetector();
  const predictions = await objectDetector.detect(imageElement, 20);

  const merged = new Map<string, number>();
  for (const pred of predictions) {
    const name = cleanLabel(pred.class);
    const existing = merged.get(name) ?? 0;
    merged.set(name, Math.max(existing, pred.score));
  }

  return Array.from(merged.entries())
    .map(([name, confidence]) => ({ name, confidence }))
    .sort((a, b) => b.confidence - a.confidence);
}

const DETECTION_CONFIDENCE_THRESHOLD = 0.4;

async function identifyLabels(imageElement: HTMLImageElement): Promise<Label[]> {
  const detected = await detectObjects(imageElement);
  const confidentDetections = detected.filter(
    (l) => l.confidence >= DETECTION_CONFIDENCE_THRESHOLD
  );

  if (confidentDetections.length > 0) {
    return confidentDetections.slice(0, 8);
  }

  return classifyImage(imageElement);
}

async function classifyImage(imageElement: HTMLImageElement): Promise<Label[]> {
  const net = await loadModel();
  const predictions = await net.classify(imageElement, 20);
  return mergeAndCleanPredictions(predictions);
}

function mergeAndCleanPredictions(
  predictions: { className: string; probability: number }[]
): Label[] {
  const merged = new Map<string, number>();

  for (const pred of predictions) {
    const names = pred.className.split(",").map((n) => cleanLabel(n.trim()));
    const canonical = names[0];
    const existing = merged.get(canonical) ?? 0;
    merged.set(canonical, Math.max(existing, pred.probability));
  }

  const sorted = Array.from(merged.entries())
    .map(([name, confidence]) => ({ name, confidence }))
    .sort((a, b) => b.confidence - a.confidence);

  return sorted.filter((l) => l.confidence > 0.01).slice(0, 8);
}

function cleanLabel(label: string): string {
  return label
    .replace(/_/g, " ")
    .trim()
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

// ---------------------------------------------------------------------------
// OCR Engine 1: Tesseract.js — classic LSTM-based OCR engine
// ---------------------------------------------------------------------------

async function extractTextTesseract(imageSource: string): Promise<OcrEngineResult> {
  const start = performance.now();
  const base: Omit<OcrEngineResult, "text" | "timeMs" | "wordCount" | "charCount" | "confidence" | "error"> = {
    id: "tesseract",
    name: "Tesseract.js",
    description: "Classic LSTM-based OCR engine",
  };

  try {
    const result = await Tesseract.recognize(imageSource, "eng", {
      logger: () => {},
    });

    const text = result.data.text.trim();

    // Tesseract gives per-word confidence (0–100). Average it across all words.
    // Cast to any because some @types/tesseract.js versions omit `.words` on Page.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const words: { confidence: number }[] = (result.data as any).words ?? [];
    const avgConf =
      words.length > 0
        ? Math.round(words.reduce((sum: number, w: { confidence: number }) => sum + w.confidence, 0) / words.length)
        : -1;

    const finalText = text.length > 2 ? text : null;
    const wordCount = finalText ? finalText.split(/\s+/).filter(Boolean).length : 0;
    const charCount = finalText ? finalText.replace(/\s/g, "").length : 0;

    return {
      ...base,
      text: finalText,
      timeMs: Math.round(performance.now() - start),
      wordCount,
      charCount,
      confidence: avgConf,
    };
  } catch (err) {
    return {
      ...base,
      text: null,
      timeMs: Math.round(performance.now() - start),
      wordCount: 0,
      charCount: 0,
      confidence: -1,
      error: err instanceof Error ? err.message : "Tesseract OCR failed",
    };
  }
}

// ---------------------------------------------------------------------------
// OCR Engine 2: TrOCR — transformer-based OCR (ViT encoder + text decoder)
// Uses trocr-base-printed (333M params) — significantly better than the
// small variant. Image is preprocessed (grayscale → sharpen → binarize)
// and multi-line text is segmented before inference.
// ---------------------------------------------------------------------------

const TROCR_MODEL_ID = "Xenova/trocr-base-printed";

let trocrPipeline: ImageToTextPipeline | null = null;
let trocrLoadPromise: Promise<ImageToTextPipeline> | null = null;

async function loadTrOCR(): Promise<ImageToTextPipeline> {
  if (trocrPipeline) return trocrPipeline;
  if (trocrLoadPromise) return trocrLoadPromise;

  trocrLoadPromise = (async () => {
    const { pipeline } = await import("@huggingface/transformers");
    const extractor = await pipeline("image-to-text", TROCR_MODEL_ID, {
      dtype: "q8",
    });
    trocrPipeline = extractor;
    return extractor;
  })();

  return trocrLoadPromise;
}

/** Run TrOCR on a single (preprocessed) image and return extracted text. */
async function runTrOCRSingle(
  extractor: ImageToTextPipeline,
  imageSource: string
): Promise<string> {
  const output = await extractor(imageSource);
  const first = Array.isArray(output) ? output[0] : output;
  const generated =
    first && typeof first === "object" && "generated_text" in first
      ? String((first as { generated_text: unknown }).generated_text ?? "")
      : "";
  return generated.trim();
}

/** Character entropy: lower = more random/gibberish. */
function textEntropy(text: string): number {
  if (text.length === 0) return 0;
  const freq = new Map<string, number>();
  for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1);
  let entropy = 0;
  for (const count of freq.values()) {
    const p = count / text.length;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

/** Estimate TrOCR confidence from output quality heuristics. */
function estimateTrOCRConfidence(text: string | null): number {
  if (!text || text.length === 0) return -1;
  if (text.length < 3) return 15;

  const entropy = textEntropy(text);
  const hasSpaces = /\s/.test(text);
  const letterRatio = (text.match(/[a-zA-Z]/g)?.length ?? 0) / text.length;

  let score = 50;

  // Longer text with reasonable letter ratio → more confident
  if (text.length >= 10) score += 15;
  if (text.length >= 20) score += 10;
  if (letterRatio > 0.5) score += 10;

  // Entropy between 2.5–4.5 is typical for English; too high = gibberish
  if (entropy >= 2.5 && entropy <= 4.5) score += 10;
  else if (entropy > 5.0) score -= 20;

  // Spaces suggest real words
  if (hasSpaces) score += 5;

  return Math.max(10, Math.min(95, Math.round(score)));
}

async function extractTextTrOCR(imageSource: string): Promise<OcrEngineResult> {
  const start = performance.now();
  const base: Omit<OcrEngineResult, "text" | "timeMs" | "wordCount" | "charCount" | "confidence" | "error"> = {
    id: "trocr",
    name: "TrOCR (Transformer)",
    description: "Vision-Transformer encoder–decoder OCR model (333M params)",
  };

  try {
    const extractor = await loadTrOCR();

    // Segment image into text lines
    const lines = await segmentTextLines(imageSource);

    // Process each line: preprocess → TrOCR inference
    const lineTexts: string[] = [];
    for (const lineImg of lines) {
      const preprocessed = await preprocessForOCR(lineImg, 384);
      const text = await runTrOCRSingle(extractor, preprocessed);
      if (text.length > 0) lineTexts.push(text);
    }

    const text = lineTexts.join("\n").trim();
    const finalText = text.length > 0 ? text : null;

    const wordCount = finalText ? finalText.split(/\s+/).filter(Boolean).length : 0;
    const charCount = finalText ? finalText.replace(/\s/g, "").length : 0;
    const confidence = estimateTrOCRConfidence(finalText);

    return {
      ...base,
      text: finalText,
      timeMs: Math.round(performance.now() - start),
      wordCount,
      charCount,
      confidence,
    };
  } catch (err) {
    return {
      ...base,
      text: null,
      timeMs: Math.round(performance.now() - start),
      wordCount: 0,
      charCount: 0,
      confidence: -1,
      error: err instanceof Error ? err.message : "TrOCR failed",
    };
  }
}

function pickPrimaryText(results: OcrEngineResult[]): string | null {
  const candidates = results.filter((r) => r.text && r.text.length > 0);
  if (candidates.length === 0) return null;
  if (candidates.length === 1) return candidates[0].text;

  // Score each engine: confidence × wordCount (favors accurate + complete output)
  // A positive confidence is required; -1 means unavailable.
  const scored = candidates.map((r) => {
    const conf = r.confidence >= 0 ? r.confidence : 50;
    return { text: r.text!, score: conf * Math.max(1, r.wordCount) };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored[0].text;
}

function generateSearchResults(labels: Label[], textContent: string | null): SearchResult[] {
  const results: SearchResult[] = [];
  const topLabels = labels.filter((l) => l.confidence > 0.05).slice(0, 5);

  if (topLabels.length === 0) return results;

  const primary = topLabels[0].name;

  results.push({
    title: primary,
    link: `https://www.google.com/search?q=${encodeURIComponent(primary)}`,
    snippet: `See Google results for "${primary}"`,
  });

  results.push({
    title: `${primary} - Images`,
    link: `https://www.google.com/search?tbm=isch&q=${encodeURIComponent(primary)}`,
    snippet: `Find similar images on Google`,
  });

  results.push({
    title: `${primary} - Wikipedia`,
    link: `https://en.wikipedia.org/wiki/${encodeURIComponent(primary.replace(/ /g, "_"))}`,
    snippet: `Learn more on Wikipedia`,
  });

  results.push({
    title: `Buy ${primary}`,
    link: `https://www.google.com/search?tbm=shop&q=${encodeURIComponent(primary)}`,
    snippet: `Shop for ${primary}`,
  });

  for (const label of topLabels.slice(1, 4)) {
    results.push({
      title: label.name,
      link: `https://www.google.com/search?q=${encodeURIComponent(label.name)}`,
      snippet: `Related: ${label.name} (${Math.round(label.confidence * 100)}% match)`,
    });
  }

  if (topLabels.length > 1) {
    const combined = topLabels.slice(0, 3).map((l) => l.name).join(" ");
    results.push({
      title: `"${combined}"`,
      link: `https://www.google.com/search?q=${encodeURIComponent(combined)}`,
      snippet: `Combined search for all detected objects`,
    });
  }

  if (textContent) {
    const shortText = textContent.substring(0, 80).trim();
    results.push({
      title: `"${shortText}"`,
      link: `https://www.google.com/search?q=${encodeURIComponent(shortText)}`,
      snippet: `Search for detected text`,
    });
  }

  return results;
}

// Main analysis function — runs entirely in the browser.
// Both OCR engines are fired simultaneously via Promise.all.
export async function analyzeImage(imageUrl: string): Promise<AnalysisResult> {
  const img = new Image();
  img.crossOrigin = "anonymous";

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = imageUrl;
  });

  const [labels, tesseractResult, trocrResult] = await Promise.all([
    identifyLabels(img),
    extractTextTesseract(imageUrl),
    extractTextTrOCR(imageUrl),
  ]);

  const ocrResults: OcrEngineResult[] = [tesseractResult, trocrResult];
  const textContent = pickPrimaryText(ocrResults);
  const searchResults = generateSearchResults(labels, textContent);

  return {
    labels,
    textContent,
    ocrResults,
    searchResults,
    visualMatches: [],
  };
}
