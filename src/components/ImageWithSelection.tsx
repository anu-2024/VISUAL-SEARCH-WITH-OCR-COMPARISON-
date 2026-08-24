"use client";

import { useRef, useState, useCallback, useEffect } from "react";

export interface CropRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface ImageWithSelectionProps {
  imageUrl: string;
  onRegionSelected: (region: CropRegion, croppedDataUrl: string) => void;
  isAnalyzing: boolean;
}

// Computes the actual rendered bounds of an <img> using object-contain
// inside its container (accounts for letterboxing on the sides/top-bottom).
function getRenderedImageBounds(img: HTMLImageElement, container: HTMLElement) {
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  const naturalWidth = img.naturalWidth;
  const naturalHeight = img.naturalHeight;

  if (!naturalWidth || !naturalHeight) {
    return { left: 0, top: 0, width: containerWidth, height: containerHeight };
  }

  const containerRatio = containerWidth / containerHeight;
  const imageRatio = naturalWidth / naturalHeight;

  let renderedWidth: number;
  let renderedHeight: number;

  if (imageRatio > containerRatio) {
    // Image is wider than container -> letterboxed top/bottom
    renderedWidth = containerWidth;
    renderedHeight = containerWidth / imageRatio;
  } else {
    // Image is taller than container -> letterboxed left/right
    renderedHeight = containerHeight;
    renderedWidth = containerHeight * imageRatio;
  }

  const left = (containerWidth - renderedWidth) / 2;
  const top = (containerHeight - renderedHeight) / 2;

  return { left, top, width: renderedWidth, height: renderedHeight };
}

export default function ImageWithSelection({
  imageUrl,
  onRegionSelected,
  isAnalyzing,
}: ImageWithSelectionProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentRegion, setCurrentRegion] = useState<CropRegion | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [showHint, setShowHint] = useState(true);

  // Hide hint after a few seconds
  useEffect(() => {
    const timer = setTimeout(() => setShowHint(false), 4000);
    return () => clearTimeout(timer);
  }, []);

  const getRelativeCoords = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      const container = containerRef.current;
      if (!container) return { x: 0, y: 0 };

      const rect = container.getBoundingClientRect();
      let clientX: number, clientY: number;

      if ("touches" in e) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      } else {
        clientX = e.clientX;
        clientY = e.clientY;
      }

      return {
        x: Math.max(0, Math.min(clientX - rect.left, rect.width)),
        y: Math.max(0, Math.min(clientY - rect.top, rect.height)),
      };
    },
    []
  );

  const handleStart = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (isAnalyzing) return;
      e.preventDefault();
      const coords = getRelativeCoords(e);
      setStartPoint(coords);
      setIsDrawing(true);
      setCurrentRegion(null);
      setShowHint(false);
    },
    [getRelativeCoords, isAnalyzing]
  );

  const handleMove = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (!isDrawing || !startPoint) return;
      e.preventDefault();
      const coords = getRelativeCoords(e);

      const x = Math.min(startPoint.x, coords.x);
      const y = Math.min(startPoint.y, coords.y);
      const width = Math.abs(coords.x - startPoint.x);
      const height = Math.abs(coords.y - startPoint.y);

      setCurrentRegion({ x, y, width, height });
    },
    [isDrawing, startPoint, getRelativeCoords]
  );

  const cropRegion = useCallback(
    (region: CropRegion) => {
      const img = imgRef.current;
      const container = containerRef.current;
      if (!img || !container) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      // Get the actual rendered bounds of the image within the container
      // (accounts for object-contain letterboxing)
      const bounds = getRenderedImageBounds(img, container);

      // Convert selection (relative to container) into coordinates
      // relative to the rendered image, then clamp to image bounds
      const selLeft = Math.max(region.x, bounds.left);
      const selTop = Math.max(region.y, bounds.top);
      const selRight = Math.min(region.x + region.width, bounds.left + bounds.width);
      const selBottom = Math.min(region.y + region.height, bounds.top + bounds.height);

      const selWidth = Math.max(0, selRight - selLeft);
      const selHeight = Math.max(0, selBottom - selTop);

      if (selWidth < 4 || selHeight < 4) return;

      // Scale from rendered image size to natural image size
      const scale = img.naturalWidth / bounds.width;

      const cropX = (selLeft - bounds.left) * scale;
      const cropY = (selTop - bounds.top) * scale;
      const cropW = selWidth * scale;
      const cropH = selHeight * scale;

      // Upscale small crops so the classifier gets enough detail
      // (MobileNet works on 224x224 input; tiny crops lose detail)
      const MIN_OUTPUT_SIZE = 320;
      const outputScale = Math.max(1, MIN_OUTPUT_SIZE / Math.max(cropW, cropH));

      canvas.width = cropW * outputScale;
      canvas.height = cropH * outputScale;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, cropX, cropY, cropW, cropH, 0, 0, canvas.width, canvas.height);

      const croppedDataUrl = canvas.toDataURL("image/jpeg", 0.95);
      onRegionSelected(region, croppedDataUrl);
    },
    [onRegionSelected]
  );

  const handleEnd = useCallback(() => {
    if (!isDrawing || !currentRegion) {
      setIsDrawing(false);
      return;
    }

    setIsDrawing(false);

    // Minimum size check (at least 20x20px)
    if (currentRegion.width < 20 || currentRegion.height < 20) {
      setCurrentRegion(null);
      return;
    }

    // Crop the selected region and pass it up
    cropRegion(currentRegion);
  }, [isDrawing, currentRegion, cropRegion]);

  // Auto-select entire image on first load
  useEffect(() => {
    if (imageLoaded && containerRef.current && imgRef.current && !currentRegion) {
      const container = containerRef.current;
      const img = imgRef.current;
      const bounds = getRenderedImageBounds(img, container);

      // Select the full rendered image (no padding — avoids cropping into
      // the letterbox background which would confuse the classifier)
      const autoRegion: CropRegion = {
        x: bounds.left,
        y: bounds.top,
        width: bounds.width,
        height: bounds.height,
      };
      setCurrentRegion(autoRegion);
      // Auto-analyze the full image
      setTimeout(() => cropRegion(autoRegion), 100);
    }
  }, [imageLoaded, cropRegion, currentRegion]);

  return (
    <div className="relative select-none">
      {/* Image container */}
      <div
        ref={containerRef}
        className="relative rounded-2xl overflow-hidden bg-[#202124] cursor-crosshair"
        onMouseDown={handleStart}
        onMouseMove={handleMove}
        onMouseUp={handleEnd}
        onMouseLeave={handleEnd}
        onTouchStart={handleStart}
        onTouchMove={handleMove}
        onTouchEnd={handleEnd}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Uploaded image"
          className="w-full h-auto max-h-[60vh] object-contain"
          onLoad={() => setImageLoaded(true)}
          draggable={false}
        />

        {/* Dark overlay outside selection */}
        {currentRegion && (
          <div className="absolute inset-0 pointer-events-none">
            {/* Semi-transparent overlay */}
            <div className="absolute inset-0 bg-black/50" />
            {/* Clear cutout for selected region */}
            <div
              className="absolute bg-transparent"
              style={{
                left: currentRegion.x,
                top: currentRegion.y,
                width: currentRegion.width,
                height: currentRegion.height,
                boxShadow: "0 0 0 9999px rgba(0,0,0,0.5)",
              }}
            />
            {/* Selection border */}
            <div
              className="absolute border-2 border-white rounded-sm"
              style={{
                left: currentRegion.x,
                top: currentRegion.y,
                width: currentRegion.width,
                height: currentRegion.height,
              }}
            >
              {/* Corner handles */}
              <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-white rounded-tl-sm" />
              <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-white rounded-tr-sm" />
              <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-white rounded-bl-sm" />
              <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-white rounded-br-sm" />
            </div>
          </div>
        )}

        {/* Analyzing overlay */}
        {isAnalyzing && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="bg-black/60 rounded-full px-4 py-2 flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span className="text-white text-sm">Searching...</span>
            </div>
          </div>
        )}

        {/* Hint */}
        {showHint && !isAnalyzing && !isDrawing && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-white text-xs px-3 py-2 rounded-full backdrop-blur-sm">
            Drag to select a region to search
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
