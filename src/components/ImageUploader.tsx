"use client";

import { useCallback, useState, useEffect } from "react";

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
}

export default function ImageUploader({ onImageSelected }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onImageSelected(file);
      }
    },
    [onImageSelected]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onImageSelected(file);
      }
    },
    [onImageSelected]
  );

  // Global paste handler
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith("image/")) {
          const file = items[i].getAsFile();
          if (file) {
            onImageSelected(file);
            break;
          }
        }
      }
    };

    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [onImageSelected]);

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`
        relative border-2 border-dashed rounded-2xl p-10 text-center
        transition-all duration-200 ease-out
        ${
          isDragging
            ? "border-[var(--primary)] bg-blue-50/5 scale-[1.01]"
            : "border-[var(--border)] hover:border-[var(--primary)]/50 hover:bg-[var(--surface)]/50"
        }
      `}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        aria-label="Upload image"
      />

      <div className="space-y-4 pointer-events-none">
        <div className="mx-auto w-14 h-14 rounded-2xl bg-[var(--surface)] border border-[var(--border)] flex items-center justify-center">
          {isDragging ? (
            <svg className="w-7 h-7 text-[var(--primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          ) : (
            <svg className="w-7 h-7 text-[var(--primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          )}
        </div>

        <div>
          <p className="text-base font-medium">
            {isDragging ? "Drop your image" : "Drag an image here or "}
            {!isDragging && (
              <span className="text-[var(--primary)]">upload a file</span>
            )}
          </p>
          <p className="text-[var(--text-secondary)] text-sm mt-1.5">
            Or paste an image from clipboard (Ctrl+V)
          </p>
        </div>
      </div>
    </div>
  );
}
