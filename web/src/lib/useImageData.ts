"use client";

import { useEffect, useState } from "react";

export function useDecodedRgba(src: string | null, size = 224): ImageData | null {
  const [data, setData] = useState<ImageData | null>(null);

  useEffect(() => {
    if (!src) {
      setData(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, size, size);
      setData(ctx.getImageData(0, 0, size, size));
    };
    img.src = src;
    return () => {
      cancelled = true;
    };
  }, [src, size]);

  return data;
}

export function sampleLumaAt(data: ImageData | null, x: number, y: number, size = 224): number | null {
  if (!data) return null;
  const cx = Math.max(0, Math.min(size - 1, Math.floor(x)));
  const cy = Math.max(0, Math.min(size - 1, Math.floor(y)));
  const i = (cy * size + cx) * 4;
  return (0.299 * data.data[i] + 0.587 * data.data[i + 1] + 0.114 * data.data[i + 2]) / 255.0;
}
