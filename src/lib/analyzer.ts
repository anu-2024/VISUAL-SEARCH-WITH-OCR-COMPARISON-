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

async function extractTextTrOCR(imageSource: string): Promise<OcrEngineResult> {
  const start = performance.now();
  const base: Omit<OcrEngineResult, "text" | "timeMs" | "wordCount" | "charCount" | "confidence" | "error"> = {
    id: "trocr",
    name: "TrOCR (Transformer)",
    description: "Vision-Transformer encoder–decoder OCR model",
  };

  try {
    const extractor = await loadTrOCR();
    const output = await extractor(imageSource);
    const first = Array.isArray(output) ? output[0] : output;
    const generated =
      first && typeof first === "object" && "generated_text" in first
        ? String((first as { generated_text: unknown }).generated_text ?? "")
        : "";
    const text = generated.trim();
    const finalText = text.length > 0 ? text : null;

    const wordCount = finalText ? finalText.split(/\s+/).filter(Boolean).length : 0;
    const charCount = finalText ? finalText.replace(/\s/g, "").length : 0;

    // TrOCR doesn't expose a confidence score. Estimate based on
    // output density: longer, non-trivially short outputs are more reliable.
    const confidence =
      finalText === null
        ? -1
        : finalText.length < 3
        ? 20
        : finalText.length < 10
        ? 50
        : 75;

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

  return candidates.reduce((best, current) =>
    (current.text?.length ?? 0) > (best.text?.length ?? 0) ? current : best
  ).text;
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
