"use client";

import { useEffect, useRef } from "react";
import type { StudyBoxROI, StudyMeasurementRow } from "@/types/workbench";

export interface ZoomColumnItem {
  key: "clean" | "noisy" | "svd" | "fft";
  label: string;
  imageData: ImageData | null;
  row: StudyMeasurementRow | null;
  cleanContrastLuma: number;
  accentClass: string;
}

interface StudyZoomDeckProps {
  columns: ZoomColumnItem[];
  roi: StudyBoxROI;
  focusMode: "detail" | "ref";
  featureLabel: string;
  selectedSeed: number;
}

const ZOOM_PX = 144; // 36x36 ROI rendered at crisp 4x nearest-neighbor scale

function RoiZoomCanvas({
  imageData,
  roi,
  borderClass,
}: {
  imageData: ImageData | null;
  roi: StudyBoxROI;
  borderClass: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const roiW = roi.x1 - roi.x0;
    const roiH = roi.y1 - roi.y0;
    const tmp = document.createElement("canvas");
    tmp.width = 224;
    tmp.height = 224;
    const tctx = tmp.getContext("2d");
    if (!tctx) return;
    tctx.putImageData(imageData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, ZOOM_PX, ZOOM_PX);
    ctx.drawImage(tmp, roi.x0, roi.y0, roiW, roiH, 0, 0, ZOOM_PX, ZOOM_PX);
  }, [imageData, roi]);

  return (
    <canvas
      ref={canvasRef}
      width={ZOOM_PX}
      height={ZOOM_PX}
      className={`w-[120px] h-[120px] xl:w-[144px] xl:h-[144px] shrink-0 border bg-black block ${borderClass}`}
      style={{ imageRendering: "pixelated" }}
    />
  );
}

export function StudyZoomDeck({
  columns,
  roi,
  focusMode,
  featureLabel,
  selectedSeed,
}: StudyZoomDeckProps) {
  const isDetail = focusMode === "detail";
  const borderClass = isDetail ? "border-lab-warn" : "border-lab-signal";

  return (
    <section className="flex-1 lg:min-h-0 flex flex-col bg-lab-bg lg:overflow-hidden">
      <div className="h-8 shrink-0 px-3 border-b border-lab-border bg-lab-surface flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`w-2 h-2 shrink-0 ${isDetail ? "bg-lab-warn" : "bg-lab-signal"}`}
          />
          <span className="font-sans text-[11px] font-semibold tracking-[0.06em] text-lab-text truncate">
            {isDetail
              ? `SYNCHRONIZED 4-COLUMN ZOOM · ${featureLabel.toUpperCase()}`
              : "SYNCHRONIZED 4-COLUMN ZOOM · REFERENCE ROI [16:52, 16:52]"}
          </span>
        </div>
        <span className="font-mono text-[10px] text-lab-muted shrink-0 hidden md:inline tabular-nums">
          {isDetail
            ? "COMPARE OVERALL PSNR vs. LOCAL FEATURE CONTRAST RETAINED"
            : "COMPARE OVERALL PSNR vs. REFERENCE ROI LUMINANCE RESIDUAL σ"}
          {` · SEED=${selectedSeed}`}
        </span>
      </div>

      <div className="flex-1 lg:min-h-0 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-lab-border lg:overflow-hidden">
        {columns.map((col) => {
          const r = col.row;
          return (
            <div
              key={col.key}
              className="p-3 flex items-center gap-3 min-w-0 bg-lab-bg"
            >
              <RoiZoomCanvas imageData={col.imageData} roi={roi} borderClass={borderClass} />

              <div className="flex-1 min-w-0 space-y-2 font-mono tabular-nums">
                <div className={`text-[11px] font-semibold truncate ${col.accentClass}`}>
                  {col.label}
                </div>

                {/* Paired Headline Metrics: Overall PSNR vs. Focused ROI Metric */}
                <div className="grid grid-cols-1 gap-1.5">
                  <div className="px-2 py-1.5 border border-lab-border bg-lab-surface">
                    <div className="text-[9px] text-lab-subtle">OVERALL PSNR</div>
                    <div className="text-sm font-semibold text-lab-text">
                      {r ? `${r.psnr_db.toFixed(2)} dB` : "∞ (Clean)"}
                    </div>
                  </div>

                  {isDetail ? (
                    <div className="px-2 py-1.5 border border-lab-warn/40 bg-lab-warn/5">
                      <div className="text-[9px] text-lab-warn">CONTRAST RETAINED</div>
                      <div className="text-sm font-semibold text-lab-text">
                        {r ? `${r.feature_contrast_retained_pct.toFixed(1)}%` : "100.0%"}
                      </div>
                      <div className="text-[9px] text-lab-muted">
                        ΔL = {r ? r.feature_contrast_luma.toFixed(4) : col.cleanContrastLuma.toFixed(4)}
                      </div>
                    </div>
                  ) : (
                    <div className="px-2 py-1.5 border border-lab-signal/40 bg-lab-signal/5">
                      <div className="text-[9px] text-lab-signal">REF ROI RESIDUAL σ</div>
                      <div className="text-sm font-semibold text-lab-text">
                        {r ? r.ref_roi_residual_std.toFixed(4) : "0.0000"}
                      </div>
                      <div className="text-[9px] text-lab-muted">
                        SSIM {r ? r.ssim.toFixed(4) : "1.0000"}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between text-[9px] text-lab-subtle pt-0.5">
                  <span>SSIM {r ? r.ssim.toFixed(3) : "1.000"}</span>
                  <span>{r ? `${r.pipeline_latency_ms.toFixed(1)} ms*` : "0.0 ms"}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
