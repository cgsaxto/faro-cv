"use client";

import { useEffect, useRef, useState } from "react";
import {
  clientPointToImagePixel,
  computeObjectContainBox,
  imagePixelToContainerPercent,
} from "@/lib/imageCoords";
import type { StudyBoxROI } from "@/types/workbench";

export interface StudyCursor {
  x: number;
  y: number;
  active: boolean;
  locked: boolean;
}

interface StudyColumnCardProps {
  indexLabel: string;
  title: string;
  subtitle: string;
  imageUrl: string;
  accentHex: string;
  accentClass: string;
  refRoi: StudyBoxROI;
  detailRoi: StudyBoxROI;
  activeFocus: "detail" | "ref";
  cursor: StudyCursor;
  onSelectFocus: (focus: "detail" | "ref") => void;
  onCursorMove: (x: number, y: number) => void;
  onCursorLeave: () => void;
  onCursorClick: (x: number, y: number) => void;
}

const IMG_SIZE = 224;

function boxToContainerStyle(
  roi: StudyBoxROI,
  containerW: number,
  containerH: number,
): React.CSSProperties {
  const box = computeObjectContainBox(containerW, containerH, IMG_SIZE, IMG_SIZE);
  if (containerW <= 0 || containerH <= 0 || box.renderedW <= 0) {
    return {
      left: `${(roi.x0 / IMG_SIZE) * 100}%`,
      top: `${(roi.y0 / IMG_SIZE) * 100}%`,
      width: `${((roi.x1 - roi.x0) / IMG_SIZE) * 100}%`,
      height: `${((roi.y1 - roi.y0) / IMG_SIZE) * 100}%`,
    };
  }
  return {
    left: `${((box.offsetX + (roi.x0 / IMG_SIZE) * box.renderedW) / containerW) * 100}%`,
    top: `${((box.offsetY + (roi.y0 / IMG_SIZE) * box.renderedH) / containerH) * 100}%`,
    width: `${(((roi.x1 - roi.x0) / IMG_SIZE) * box.renderedW / containerW) * 100}%`,
    height: `${(((roi.y1 - roi.y0) / IMG_SIZE) * box.renderedH / containerH) * 100}%`,
  };
}

export function StudyColumnCard({
  indexLabel,
  title,
  subtitle,
  imageUrl,
  accentHex,
  accentClass,
  refRoi,
  detailRoi,
  activeFocus,
  cursor,
  onSelectFocus,
  onCursorMove,
  onCursorLeave,
  onCursorClick,
}: StudyColumnCardProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 224, h: 224 });

  useEffect(() => {
    const el = frameRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const obs = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setSize({ w: entry.contentRect.width, h: entry.contentRect.height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const hitPixel = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const hit = clientPointToImagePixel(e.clientX, e.clientY, rect, IMG_SIZE, IMG_SIZE);
    return hit && hit.inside ? { x: hit.x, y: hit.y } : null;
  };

  const cross = imagePixelToContainerPercent(cursor.x, cursor.y, size.w, size.h, IMG_SIZE, IMG_SIZE);

  return (
    <div className="flex flex-col min-w-0 min-h-0 bg-lab-bg">
      <div className="h-8 shrink-0 px-2.5 border-b border-lab-border bg-lab-surface flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="font-mono text-[10px] text-lab-subtle">{indexLabel}</span>
          <span className="font-sans text-[11px] font-semibold tracking-[0.06em] text-lab-text truncate">
            {title}
          </span>
        </div>
        <span className={`font-mono text-[10px] tabular-nums shrink-0 ${accentClass}`}>
          {subtitle}
        </span>
      </div>

      <div className="lg:flex-1 lg:min-h-0 py-3 px-2 lg:p-2 flex items-center justify-center overflow-hidden">
        <div
          ref={frameRef}
          onMouseMove={(e) => {
            const p = hitPixel(e);
            if (p) {
              onCursorMove(p.x, p.y);
            } else {
              onCursorLeave();
            }
          }}
          onMouseLeave={onCursorLeave}
          onClick={(e) => {
            const p = hitPixel(e);
            if (p) onCursorClick(p.x, p.y);
          }}
          className="relative aspect-square w-[224px] h-[224px] lg:w-auto lg:h-full lg:max-h-[260px] max-w-full border border-lab-borderStrong bg-black cursor-crosshair select-none overflow-hidden"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl}
            alt={title}
            draggable={false}
            className="w-full h-full object-contain block"
            style={{ imageRendering: "pixelated" }}
          />

          {/* Clickable Green Reference ROI box */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectFocus("ref");
            }}
            style={boxToContainerStyle(refRoi, size.w, size.h)}
            title="Click to inspect Reference ROI (luminance residual σ) across all 4 columns"
            className={`absolute border transition-all cursor-pointer ${
              activeFocus === "ref"
                ? "border-lab-signal border-2 bg-lab-signal/10"
                : "border-lab-signal/70 hover:border-lab-signal"
            }`}
          />

          {/* Clickable Amber Detail ROI box */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectFocus("detail");
            }}
            style={boxToContainerStyle(detailRoi, size.w, size.h)}
            title="Click to zoom Target Detail ROI across all 4 columns and compare PSNR vs. Contrast Retained"
            className={`absolute border transition-all cursor-pointer ${
              activeFocus === "detail"
                ? "border-lab-warn border-2 bg-lab-warn/10"
                : "border-lab-warn/80 hover:border-lab-warn"
            }`}
          />

          {cursor.active && (
            <>
              <div
                className="pointer-events-none absolute top-0 bottom-0 w-px"
                style={{ left: `${cross.leftPct}%`, backgroundColor: accentHex, opacity: 0.6 }}
              />
              <div
                className="pointer-events-none absolute left-0 right-0 h-px"
                style={{ top: `${cross.topPct}%`, backgroundColor: accentHex, opacity: 0.6 }}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
