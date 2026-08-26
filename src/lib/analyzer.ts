import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import Tesseract from "tesseract.js";
import type { ImageToTextPipeline } from "@huggingface/transformers";
import { AnalysisResult, Label, OcrEngineResult, SearchResult } from "@/types";

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
// Hybrid approach: Tesseract detects text lines → TrOCR recognizes each line.
// This leverages Tesseract's superior layout analysis + TrOCR's superior
// character recognition. Also tries raw image directly for single-line images.
// ---------------------------------------------------------------------------

const TROCR_MODEL_ID = "Xenova/trocr-small-printed";

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
  return Math.max(10, Math.min(95, Math.round(50 + q * 0.45)));
}

/**
 * Use Tesseract's word-level bounding boxes to crop individual lines,
 * then run TrOCR on each crop. This gives TrOCR exactly what it needs:
 * clean, single-line text images.
 */
async function trocrWithTesseractCrops(
  extractor: ImageToTextPipeline,
  imageSource: string
): Promise<string> {
  const result = await Tesseract.recognize(imageSource, "eng", {
    logger: () => {},
  });

  // Group Tesseract words into lines by their y-coordinate proximity
  type TesseractWord = { bbox: { x0: number; y0: number; x1: number; y1: number } };
  // Tesseract.js types omit `.words`; cast page data to access it.
  const words: TesseractWord[] = (result.data as unknown as { words?: TesseractWord[] }).words ?? [];
  if (words.length === 0) return "";

  // Sort words by position: top-to-bottom, left-to-right
  const sorted = [...words].sort((a, b) => {
    const yDiff = a.bbox.y0 - b.bbox.y0;
    return Math.abs(yDiff) > 15 ? yDiff : a.bbox.x0 - b.bbox.x0;
  });

  // Group into lines (words within 15px vertical distance)
  const lines: { words: typeof sorted; bbox: { x0: number; y0: number; x1: number; y1: number } }[] = [];
  let currentLine: typeof sorted = [];

  for (const word of sorted) {
    if (currentLine.length === 0) {
      currentLine.push(word);
      continue;
    }
    const lastY = currentLine[currentLine.length - 1].bbox.y0;
    if (Math.abs(word.bbox.y0 - lastY) <= 15) {
      currentLine.push(word);
    } else {
      lines.push({ words: currentLine, bbox: computeBBox(currentLine) });
      currentLine = [word];
    }
  }
  if (currentLine.length > 0) {
    lines.push({ words: currentLine, bbox: computeBBox(currentLine) });
  }

  // Crop each line and run TrOCR
  const lineTexts: string[] = [];
  for (const line of lines) {
    const { bbox } = line;
    const pad = 8;
    const cropCanvas = await cropRegion(imageSource, {
      x: Math.max(0, bbox.x0 - pad),
      y: Math.max(0, bbox.y0 - pad),
      width: bbox.x1 - bbox.x0 + pad * 2,
      height: bbox.y1 - bbox.y0 + pad * 2,
    });
    const text = await runTrOCRSingle(extractor, cropCanvas);
    if (text.length > 0) lineTexts.push(text);
  }

  return lineTexts.join("\n");
}

function computeBBox(
  words: { bbox: { x0: number; y0: number; x1: number; y1: number } }[]
): { x0: number; y0: number; x1: number; y1: number } {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const w of words) {
    x0 = Math.min(x0, w.bbox.x0);
    y0 = Math.min(y0, w.bbox.y0);
    x1 = Math.max(x1, w.bbox.x1);
    y1 = Math.max(y1, w.bbox.y1);
  }
  return { x0, y0, x1, y1 };
}

/** Crop a region from an image and return as data URL. */
function cropRegion(
  imageSource: string,
  region: { x: number; y: number; width: number; height: number }
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(region.width));
      canvas.height = Math.max(1, Math.round(region.height));
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(
        img,
        region.x, region.y, region.width, region.height,
        0, 0, canvas.width, canvas.height
      );
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = () => resolve(imageSource);
    img.src = imageSource;
  });
}

async function extractTextTrOCR(imageSource: string): Promise<OcrEngineResult> {
  const start = performance.now();
  const base: Omit<OcrEngineResult, "text" | "timeMs" | "wordCount" | "charCount" | "confidence" | "error"> = {
    id: "trocr",
    name: "TrOCR (Transformer)",
    description: "Vision-Transformer encoder–decoder OCR model",
  };

  try {
    const extractor = await loadTrOCR();

    // Strategy 1: Send raw image directly (best for single-line images)
    const rawText = await runTrOCRSingle(extractor, imageSource);

    // Strategy 2: Use Tesseract to detect lines, then TrOCR to recognize each
    const hybridText = await trocrWithTesseractCrops(extractor, imageSource);

    // Pick whichever got more/better text
    const rawScore = scoreTextQuality(rawText);
    const hybridScore = scoreTextQuality(hybridText);
    const bestText = hybridScore >= rawScore ? hybridText : rawText;

    const finalText = bestText.length > 0 ? bestText : null;

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

  // Prefer TrOCR whenever it produced meaningful output (>= 3 chars),
  // since the hybrid pipeline already validates via Tesseract word crops.
  if (trocr.wordCount >= 1 && (trocr.charCount ?? 0) >= 3) {
    return trocr.text;
  }

  return tesseract.text;
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
