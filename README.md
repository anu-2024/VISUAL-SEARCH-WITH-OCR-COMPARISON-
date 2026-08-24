# Visual Search + OCR Comparison — Google Lens Clone

A free, open-source, **fully client-side** Google Lens alternative. Upload a photo or take one with your camera to identify objects, extract text with **two OCR engines running in parallel**, and compare their results — all without any API keys or server processing.

---

## Features

- **Image Upload** — Drag & drop, file picker, or paste from clipboard
- **Camera Capture** — Take photos directly (mobile & desktop with camera switching)
- **Object Detection** — Identifies everyday objects using TensorFlow.js **COCO-SSD** (80 common classes: person, cell phone, laptop, cup, chair, bottle, etc.), with MobileNet's ImageNet classifier as a fallback for scenes outside those 80 classes (landscapes, artwork, food close-ups). All in-browser.
- **Dual OCR Engines — run in parallel, compared side by side**:
  - **Tesseract.js** — Classic LSTM-based OCR engine
  - **TrOCR** — Vision Transformer encoder + autoregressive text decoder, run fully in-browser via [transformers.js](https://github.com/huggingface/transformers.js)
  - **OCR Compare tab** shows: detected text per engine, processing time, word count, character count, confidence score, performance bar charts, CER accuracy calculator, and a winner summary
- **Search Links** — Direct Google, Google Images, Wikipedia, and Shopping links
- **Mobile Responsive** — Full mobile support with front/back camera
- **Dark Mode** — Automatic theme based on system preference
- **100% Private** — No data leaves your browser. Zero APIs. Zero tracking.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router, Turbopack) |
| Language | TypeScript |
| Styling | Tailwind CSS 4 + custom CSS |
| Object Detection | TensorFlow.js + COCO-SSD (80 everyday classes) |
| Scene Classification | TensorFlow.js + MobileNet v2 (ImageNet fallback) |
| OCR Engine 1 | Tesseract.js (LSTM, WASM) |
| OCR Engine 2 | @huggingface/transformers + TrOCR (ONNX Runtime Web) |

---

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

**No API keys needed. No environment variables. Just install and run.**

---

## Deploy

### Vercel (recommended)
```bash
npx vercel login   # one-time login
npx vercel --yes   # deploy
```

Or click:
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-repo/lens)

### Temporary link (no account needed)
```bash
npx vercel deploy --temporary
```

---

## How It Works

1. User uploads/captures an image and draws a selection box on it
2. **COCO-SSD** scans the crop for 80 everyday object classes; if nothing is confidently detected, **MobileNet** runs as a fallback classifier for scenes outside that vocabulary
3. Both OCR engines fire **simultaneously** via `Promise.all`:
   - Tesseract.js (classical pipeline)
   - TrOCR via transformers.js (transformer pipeline, weights cached after first load)
4. The **OCR Compare** tab populates as results arrive — showing detected text, timing, confidence, and metrics for both engines
5. The best transcription drives the search links; all raw outputs are also shown in the **Raw Text** tab
6. Results generate direct links to Google, Google Images, Wikipedia, Shopping
7. **Everything runs in the user's browser** — no server, no uploads, no tracking

---

## OCR Models — Deep Explanation

### Engine 1 — Tesseract.js (Classical LSTM OCR)

#### What is Tesseract?
Tesseract is one of the **oldest and most battle-tested open-source OCR engines**, originally developed at HP Labs in the 1980s and later open-sourced by Google in 2006. Tesseract.js is a WebAssembly port that runs the full engine directly in the browser.

#### Architecture
Tesseract uses a **multi-stage classical pipeline**:

```
Image Input
    ↓
1. Binarisation       — Convert to black & white using adaptive thresholding
    ↓
2. Layout Analysis    — Detect text blocks, columns, paragraphs, lines, words
    ↓
3. Line Normalisation — Deskew, scale, and baseline-correct each text line
    ↓
4. LSTM Network       — A bidirectional LSTM reads each line character by character
    ↓
5. Language Model     — A word-level n-gram model refines the LSTM output
    ↓
6. Output             — Unicode text + per-word confidence scores (0–100)
```

The **LSTM (Long Short-Term Memory)** at its core is a recurrent neural network trained to translate sequences of image pixels (a horizontal scan of a text line) into sequences of characters. It can handle variable-width characters and overlapping glyphs better than the older character-segmentation approach Tesseract 3 used.

#### Strengths
- ⚡ **Very fast** — no large model download, WASM binary is ~10 MB
- ✅ **Excellent on clean, printed documents** — business cards, receipts, typed text, book pages
- 📊 **Returns real confidence scores** — per-word confidence (0–100) from its language model
- 🌍 **Multi-language** — supports 100+ languages via `.traineddata` files
- 🔧 **Fine-tunable** — can be retrained on domain-specific fonts

#### Weaknesses
- ❌ Struggles with **handwriting** (it's not trained for it by default)
- ❌ Sensitive to **image quality** — blurry, low-contrast, or skewed images drop accuracy significantly
- ❌ Poor on **stylised fonts**, watermarks, or text mixed with complex backgrounds
- ❌ Requires relatively **clean line segmentation** — fails when text is curved, rotated, or overlapping

#### In this app
```
File: src/lib/analyzer.ts → extractTextTesseract()
Model: Tesseract.js v7 (eng traineddata)
Confidence: Averaged from result.data.words[].confidence (real Tesseract scores)
```

---

### Engine 2 — TrOCR (Transformer-based OCR)

#### What is TrOCR?
TrOCR is a **state-of-the-art OCR model** introduced by Microsoft Research in 2021 ([paper](https://arxiv.org/abs/2109.10282)). Unlike classical OCR pipelines, TrOCR uses a pure **encoder-decoder transformer architecture** — the same family of models behind GPT and BERT — applied end-to-end to image-to-text conversion.

In this app, it runs fully **in-browser** via [transformers.js](https://github.com/huggingface/transformers.js) (Hugging Face's JavaScript port) using **ONNX Runtime Web** compiled to WebAssembly.

#### Architecture

```
Image Input (any size)
    ↓
1. Image Preprocessing  — Resize to 384×384, normalise pixel values
    ↓
2. ViT Encoder (Vision Transformer)
   — Splits image into 16×16 pixel patches (like words in NLP)
   — Each patch becomes an embedding vector
   — 12 layers of multi-head self-attention across all patches
   — Learns global spatial relationships across the entire image at once
    ↓
3. Cross-Attention Bridge
   — Encoder outputs are fed as "context" to the decoder
    ↓
4. Autoregressive Text Decoder (GPT-2 style)
   — Generates text token by token, left to right
   — Each new token attends to: all encoder (image) outputs + previously generated tokens
   — Stops at [EOS] (end-of-sequence) token
    ↓
5. Output — Unicode text string
```

The key insight is the **ViT (Vision Transformer) encoder**: instead of scanning image pixels line by line like Tesseract's LSTM, it processes the **entire image as a 2D grid of patches simultaneously**, using self-attention to understand the global context. This means it can handle rotated text, multi-column layouts, and complex backgrounds much better than LSTM-based systems.

#### Model used: `Xenova/trocr-small-printed`
This is the **small, quantized (q8)** variant of TrOCR optimised for the browser:

| Variant | Parameters | Size | Use Case |
|---|---|---|---|
| `trocr-base-printed` | 334M | ~1.3 GB | High accuracy, needs GPU |
| `trocr-small-printed` | 62M | ~250 MB | Balanced, server-side |
| `Xenova/trocr-small-printed` (q8) | 62M (quantized) | ~80 MB | **This app — browser WASM** |

**Quantization (q8)**: model weights are compressed from 32-bit floats to 8-bit integers, reducing model size ~4× with minimal accuracy loss. This makes it feasible to download and run in a browser.

#### Strengths
- 🔎 **Better on complex images** — handles varied fonts, slight rotation, mixed backgrounds
- 🧠 **Global context understanding** — the attention mechanism sees the whole image at once
- ✅ **End-to-end trained** — no separate line detection, binarisation, or layout analysis steps needed
- 🚀 **Keeps improving** — transformer architecture benefits from more data/compute, unlike LSTM which plateaus

#### Weaknesses
- 🐢 **Slow on first run** — ~80 MB model download + WASM compile (subsequent runs use browser cache)
- ❌ **Single-line focused** — `trocr-small-printed` is optimised for single text lines, not full documents
- ⚠️ **No native confidence scores** — unlike Tesseract, TrOCR's decoder doesn't expose token probabilities in this inference setup (app uses a length-based heuristic)
- 💾 **Higher memory usage** — 80 MB+ resident in browser memory vs ~10 MB for Tesseract

#### In this app
```
File: src/lib/analyzer.ts → extractTextTrOCR()
Model: Xenova/trocr-small-printed (HuggingFace Hub, quantized q8)
Runtime: @huggingface/transformers → ONNX Runtime Web → WebAssembly
Confidence: Estimated heuristic (output length-based), not raw logits
```

---

## Side-by-Side Comparison

| Feature | Tesseract.js | TrOCR (Transformer) |
|---|---|---|
| **Architecture** | Binarize → Layout → LSTM → LM | ViT Encoder → Autoregressive Decoder |
| **Era** | Classical (1985, modernised 2017) | Modern deep learning (2021) |
| **Model size** | ~10 MB (WASM binary) | ~80 MB (q8 ONNX weights) |
| **First-run load** | < 2 seconds | 10–60s (download + compile) |
| **Subsequent runs** | Instant | Fast (cached model) |
| **Clean printed text** | ✅ Excellent | ✅ Good |
| **Handwriting** | ❌ Poor | ✅ Good (with right variant) |
| **Stylised fonts** | ❌ Struggles | ✅ More robust |
| **Low quality / blurry** | ⚠️ Degrades quickly | ⚠️ More resilient |
| **Multi-line documents** | ✅ Excellent (layout analysis) | ⚠️ Better per-line |
| **Confidence scores** | ✅ Real (per-word, 0–100) | ⚠️ Heuristic estimate |
| **Language support** | 100+ languages | English-focused (this model) |
| **Runs in browser** | ✅ Yes (WASM) | ✅ Yes (ONNX Runtime Web) |
| **Server required** | ❌ No | ❌ No |

---

## Metrics Explained (OCR Compare Tab)

| Metric | Meaning |
|---|---|
| **⏱ Time (ms)** | Wall-clock time from start of recognition to final text output |
| **📝 Words** | Number of whitespace-separated tokens in the detected text |
| **🔤 Chars** | Number of non-whitespace characters in the detected text |
| **🎯 Confidence** | For Tesseract: real average per-word confidence (0–100) from its language model. For TrOCR: heuristic estimate based on output length |
| **✅ CER Accuracy** | Character Error Rate accuracy — paste the actual ground-truth text and the app computes `1 - (Levenshtein distance / ground-truth length)` × 100. Higher = better. |

---

## Privacy

All image analysis happens locally in the browser using WebAssembly and WebGL. Your images are never uploaded to any server. The only network requests are:
- Loading ML model files on first use (cached permanently by the browser afterward)
- Opening search links (Google, Wikipedia) when you click them

---

## License

MIT
