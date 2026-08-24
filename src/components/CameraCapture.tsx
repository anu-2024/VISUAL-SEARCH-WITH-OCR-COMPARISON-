"use client";

import { useRef, useState, useEffect } from "react";

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  onClose: () => void;
}

export default function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment");

  useEffect(() => {
    let cancelled = false;

    async function startCamera() {
      // Stop any existing stream before requesting a new one
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode,
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
        });

        if (cancelled) {
          mediaStream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = mediaStream;
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
        setError(null);
      } catch {
        if (!cancelled) {
          setError("Unable to access camera. Please check permissions.");
        }
      }
    }

    startCamera();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [facingMode]);

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], "camera-capture.jpg", {
            type: "image/jpeg",
          });
          // Stop camera
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
          }
          onCapture(file);
        }
      },
      "image/jpeg",
      0.9
    );
  };

  const switchCamera = () => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col">
      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 z-10 p-4 flex justify-between items-center">
        <button
          onClick={onClose}
          className="w-10 h-10 rounded-full bg-black/50 flex items-center justify-center text-white cursor-pointer"
          aria-label="Close camera"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <button
          onClick={switchCamera}
          className="w-10 h-10 rounded-full bg-black/50 flex items-center justify-center text-white cursor-pointer"
          aria-label="Switch camera"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Camera view */}
      {error ? (
        <div className="flex-1 flex items-center justify-center text-white text-center p-4">
          <div className="space-y-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-16 h-16 mx-auto text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
            <p className="text-lg">{error}</p>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white/20 rounded-lg cursor-pointer"
            >
              Go Back
            </button>
          </div>
        </div>
      ) : (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="flex-1 object-cover"
        />
      )}

      {/* Capture button */}
      {!error && (
        <div className="absolute bottom-0 left-0 right-0 p-8 flex justify-center">
          <button
            onClick={capturePhoto}
            className="w-18 h-18 rounded-full border-4 border-white flex items-center justify-center cursor-pointer"
            aria-label="Take photo"
          >
            <div className="w-14 h-14 rounded-full bg-white hover:bg-gray-200 transition-colors" />
          </button>
        </div>
      )}

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
