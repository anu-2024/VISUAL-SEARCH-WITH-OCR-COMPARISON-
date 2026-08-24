export interface AnalysisResult {
  labels: Label[];
  textContent: string | null;
  ocrResults: OcrEngineResult[];
  searchResults: SearchResult[];
  visualMatches: VisualMatch[];
}

export interface Label {
  name: string;
  confidence: number;
}

// Result from a single OCR engine — used to compare engines against each other.
export interface OcrEngineResult {
  id: "tesseract" | "trocr";
  name: string;
  description: string;
  text: string | null;
  timeMs: number;
  wordCount: number;
  charCount: number;
  /** 0–100 estimated confidence. Tesseract gives real per-word confidence;
   *  TrOCR uses a heuristic. -1 means unavailable. */
  confidence: number;
  error?: string;
}

/** Per-engine loading status for live UI spinners */
export interface OcrEngineStatus {
  tesseract: "idle" | "running" | "done" | "error";
  trocr: "idle" | "running" | "done" | "error";
}

export interface SearchResult {
  title: string;
  link: string;
  snippet: string;
}

export interface VisualMatch {
  title: string;
  link: string;
  thumbnail: string;
  source: string;
}
