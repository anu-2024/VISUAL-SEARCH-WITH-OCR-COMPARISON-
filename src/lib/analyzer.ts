import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import Tesseract from "tesseract.js";
import type { ImageToTextPipeline } from "@huggingface/transformers";
import { AnalysisResult, Label, OcrEngineResult, SearchResult } from "@/types";

// ─── Image Preprocessing Utilities ───────────────────────────────────────────
// TrOCR expects clean, high-contrast, single-line text images.
// We provide two strategies and pick whichever gives better output.

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

/** Contrast-limited adaptive histogram stretch. */
function stretchContrast(imageData: ImageData): ImageData {
  const d = imageData.data;
  const n = d.length / 4;
  // Clip 2% from each end of histogram
  const hist = new Array(256).fill(0);
  for (let i = 0; i < d.length; i += 4) hist[d[i]]++;
  const clip = Math.floor(n * 0.02);
  let lo = 0, hi = 255, count = 0;
  for (let i = 0; i < 256; i++) { count += hist[i]; if (count >= clip) { lo = i; break; } }
  count = 0;
  for (let i = 255; i >= 0; i--) { count += hist[i]; if (count >= clip) { hi = i; break; } }
  if (hi <= lo) return imageData;
  const range = hi - lo;
  for (let i = 0; i < d.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      d[i + c] = Math.max(0, Math.min(255, Math.round(((d[i + c] - lo) / range) * 255)));
    }
  }
  return imageData;
}

// ─── Preprocessing Strategies ────────────────────────────────────────────────
// Strategy A: Full binarize (best for noisy/low-contrast photos)
// Strategy B: Contrast stretch + sharpen (best for clean/high-contrast images)
// We try both and pick whichever TrOCR rates higher.

type PreprocessResult = { dataUrl: string; strategy: string };

function canvasToDataUrl(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL("image/png");
}

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = source;
  });
}

function drawToCanvas(
  img: HTMLImageElement,
  targetWidth: number
): HTMLCanvasElement {
  const scale = targetWidth / img.naturalWidth;
  const w = targetWidth;
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  return canvas;
}

function addPadding(canvas: HTMLCanvasElement, pad: number): HTMLCanvasElement {
  const padded = document.createElement("canvas");
  padded.width = canvas.width + pad * 2;
  padded.height = canvas.height + pad * 2;
  const ctx = padded.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, padded.width, padded.height);
  ctx.drawImage(canvas, pad, pad);
  return padded;
}

/** Strategy A: grayscale → sharpen → binarize → pad */
async function preprocessBinarized(
  img: HTMLImageElement,
  targetWidth: number
): Promise<PreprocessResult> {
  const canvas = drawToCanvas(img, targetWidth);
  const ctx = canvas.getContext("2d")!;
  let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  imageData = toGrayscale(imageData);
  imageData = sharpen(imageData);
  imageData = binarize(imageData);
  ctx.putImageData(imageData, 0, 0);
  return { dataUrl: canvasToDataUrl(addPadding(canvas, 20)), strategy: "binarized" };
}

/** Strategy B: grayscale → contrast stretch → sharpen → pad */
async function preprocessContrast(
  img: HTMLImageElement,
  targetWidth: number
): Promise<PreprocessResult> {
  const canvas = drawToCanvas(img, targetWidth);
  const ctx = canvas.getContext("2d")!;
  let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  imageData = toGrayscale(imageData);
  imageData = stretchContrast(imageData);
  imageData = sharpen(imageData);
  ctx.putImageData(imageData, 0, 0);
  return { dataUrl: canvasToDataUrl(addPadding(canvas, 20)), strategy: "contrast" };
}

// ─── Text Line Segmentation ──────────────────────────────────────────────────
// TrOCR processes one line at a time. This splits multi-line regions into
// individual lines using horizontal projection profiling.

function segmentTextLines(
  imageSource: string
): Promise<string[]> {
  return new Promise((resolve) => {
    loadImage(imageSource)
      .then((img) => {
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        const canvas = drawToCanvas(img, w);
        const ctx = canvas.getContext("2d")!;
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
          return canvasToDataUrl(lineCanvas);
        });

        resolve(croppedLines);
      })
      .catch(() => resolve([imageSource]));
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
// Uses trocr-base-printed (333M params). Runs dual preprocessing strategies
// (binarized vs contrast-stretched) and picks the best result per line.
// Cross-validates against Tesseract output for final primary text selection.
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
      dtype: "fp16",
    });
    trocrPipeline = extractor;
    return extractor;
  })();

  return trocrLoadPromise;
}

/** Run TrOCR on a single image and return extracted text. */
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

/** Score text quality (higher = better OCR output). */
function scoreTextQuality(text: string): number {
  if (!text || text.length === 0) return 0;
  if (text.length < 3) return 5;

  let score = 0;

  // Length bonus — longer text (up to a point) means more was detected
  score += Math.min(text.length * 2, 40);

  // Letter ratio — real text is mostly letters
  const letters = (text.match(/[a-zA-Z]/g)?.length ?? 0) / text.length;
  score += letters * 30;

  // Entropy — English text is 2.5–4.5; too high = garbage
  const entropy = textEntropy(text);
  if (entropy >= 2.0 && entropy <= 4.5) score += 20;
  else if (entropy > 5.0) score -= 30;

  // Spaces indicate real words
  if (/\s/.test(text)) score += 10;

  // Penalties for common garbage patterns
  if (/[^\w\s.,;:!?'"()\-+/]/g.test(text)) score -= 10;
  if (/(.)\1{4,}/.test(text)) score -= 20; // 5+ repeated chars

  return Math.max(0, Math.round(score));
}

/** Estimate TrOCR confidence from text quality score. */
function estimateTrOCRConfidence(text: string | null): number {
  if (!text || text.length === 0) return -1;
  const q = scoreTextQuality(text);
  // Map quality score 0–100 to confidence 10–95
  return Math.max(10, Math.min(95, Math.round(50 + q * 0.45)));
}

/** Choose the better of two texts from different preprocessing strategies. */
function pickBetterText(a: string, b: string): string {
  const sa = scoreTextQuality(a);
  const sb = scoreTextQuality(b);
  if (sb > sa) return b;
  return a;
}

/** Run TrOCR on a single line with dual preprocessing, return best text. */
async function runTrOCRLine(
  extractor: ImageToTextPipeline,
  lineImage: string
): Promise<string> {
  const img = await loadImage(lineImage);

  // Strategy A: binarized
  const binarized = await preprocessBinarized(img, 384);
  const textA = await runTrOCRSingle(extractor, binarized.dataUrl);

  // Strategy B: contrast-stretched
  const contrast = await preprocessContrast(img, 384);
  const textB = await runTrOCRSingle(extractor, contrast.dataUrl);

  // Pick whichever scored higher
  return pickBetterText(textA, textB);
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

    // Process each line with dual preprocessing, pick best
    const lineTexts: string[] = [];
    for (const lineImg of lines) {
      const text = await runTrOCRLine(extractor, lineImg);
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

  const trocr = candidates.find((r) => r.id === "trocr");
  const tesseract = candidates.find((r) => r.id === "tesseract");

  if (!trocr || !tesseract) return candidates[0].text;

  const trocrConf = trocr.confidence >= 0 ? trocr.confidence : 0;
  const tessConf = tesseract.confidence >= 0 ? tesseract.confidence : 0;

  // Cross-validation: check if the two engines agree on any words
  const trocrWords = new Set(
    (trocr.text ?? "").toLowerCase().split(/\s+/).filter(Boolean)
  );
  const tessWords = new Set(
    (tesseract.text ?? "").toLowerCase().split(/\s+/).filter(Boolean)
  );
  let overlap = 0;
  for (const w of trocrWords) if (tessWords.has(w)) overlap++;
  const overlapRatio =
    Math.max(trocrWords.size, tessWords.size) > 0
      ? overlap / Math.max(trocrWords.size, tessWords.size)
      : 0;

  // If both agree (>50% word overlap), prefer TrOCR (more sophisticated model)
  if (overlapRatio > 0.5) {
    return trocrConf >= 40 ? trocr.text : tesseract.text;
  }

  // If they disagree significantly, use confidence-weighted quality
  const trocrScore = trocrConf * Math.max(1, trocr.wordCount);
  const tessScore = tessConf * Math.max(1, tesseract.wordCount);

  // TrOCR gets a bonus (transformer model is generally more accurate)
  const trocrBonus = 1.2;

  return trocrScore * trocrBonus >= tessScore ? trocr.text : tesseract.text;
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
