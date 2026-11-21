"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Crosshair, Lock, Unlock } from "lucide-react";
import {
  clientPointToImagePixel,
  imagePixelToContainerPercent,
} from "@/lib/imageCoords";
import type { ImageBundle, InferenceSummary } from "@/types/workbench";

interface SynchronizedViewportProps {
  images: ImageBundle | null;
  cleanInference: InferenceSummary | null;
  adversarialInference: InferenceSummary | null;
  purifiedInference: InferenceSummary | null;
}

interface CursorState {
  x: number;
  y: number;
  active: boolean;
  locked: boolean;
}

interface RgbTriple {
  r: number;
  g: number;
  b: number;
}

const IMG_SIZE = 224;
const LOUPE_GRID = 11; // 11x11 pixels around cursor
const LOUPE_SCALE = 8; // 88x88 px crisp pixelated canvas

function useDecodedImageData(dataUrl: string | null): ImageData | null {
  const [imageData, setImageData] = useState<ImageData | null>(null);

  useEffect(() => {
    if (!dataUrl) {
      setImageData(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const offscreen = document.createElement("canvas");
      offscreen.width = IMG_SIZE;
      offscreen.height = IMG_SIZE;
      const ctx = offscreen.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
      setImageData(ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE));
    };
    img.src = dataUrl;
    return () => {
      cancelled = true;
    };
  }, [dataUrl]);

  return imageData;
}

function samplePixel(data: ImageData | null, x: number, y: number): RgbTriple | null {
  if (!data) return null;
  const cx = Math.max(0, Math.min(IMG_SIZE - 1, Math.floor(x)));
  const cy = Math.max(0, Math.min(IMG_SIZE - 1, Math.floor(y)));
  const offset = (cy * IMG_SIZE + cx) * 4;
  return {
    r: data.data[offset],
    g: data.data[offset + 1],
    b: data.data[offset + 2],
  };
}

function formatDelta(v: number): string {
  if (v > 0) return `+${String(v).padStart(2, "0")}`;
  if (v < 0) return `-${String(Math.abs(v)).padStart(2, "0")}`;
  return " 00";
}

interface LoupeCanvasProps {
  imageData: ImageData | null;
  x: number;
  y: number;
  accentColor: string;
}

function PixelLoupeInset({ imageData, x, y, accentColor }: LoupeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const half = Math.floor(LOUPE_GRID / 2);
    const size = LOUPE_GRID * LOUPE_SCALE;
    ctx.clearRect(0, 0, size, size);

    for (let dy = -half; dy <= half; dy++) {
      for (let dx = -half; dx <= half; dx++) {
        const sx = x + dx;
        const sy = y + dy;
        const px = (dx + half) * LOUPE_SCALE;
        const py = (dy + half) * LOUPE_SCALE;

        if (sx >= 0 && sx < IMG_SIZE && sy >= 0 && sy < IMG_SIZE) {
          const offset = (sy * IMG_SIZE + sx) * 4;
          ctx.fillStyle = `rgb(${imageData.data[offset]}, ${imageData.data[offset + 1]}, ${imageData.data[offset + 2]})`;
        } else {
          ctx.fillStyle = "#090A0C";
        }
        ctx.fillRect(px, py, LOUPE_SCALE, LOUPE_SCALE);
      }
    }

    // Subtle pixel gridlines
    ctx.strokeStyle = "rgba(9, 10, 12, 0.45)";
    ctx.lineWidth = 1;
    for (let i = 1; i < LOUPE_GRID; i++) {
      const pos = i * LOUPE_SCALE + 0.5;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, size);
      ctx.moveTo(0, pos);
      ctx.lineTo(size, pos);
      ctx.stroke();
    }

    // Center pixel reticle highlight
    const centerPos = half * LOUPE_SCALE;
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(centerPos + 0.5, centerPos + 0.5, LOUPE_SCALE - 1, LOUPE_SCALE - 1);
  }, [imageData, x, y, accentColor]);

  if (!imageData) return null;

  const side = LOUPE_GRID * LOUPE_SCALE;
  return (
    <canvas
      ref={canvasRef}
      width={side}
      height={side}
      className="block border border-lab-borderStrong shadow-2xl bg-lab-bg"
      style={{ imageRendering: "pixelated" }}
    />
  );
}

interface ViewportColumnProps {
  indexLabel: string;
  title: string;
  dataUrl: string | null;
  imageData: ImageData | null;
  inference: InferenceSummary | null;
  accentHex: string;
  accentClass: string;
  cursor: CursorState;
  headerRight?: React.ReactNode;
  emptyHint: string;
  onCursorMove: (x: number, y: number) => void;
  onCursorLeave: () => void;
  onCursorClick: (x: number, y: number) => void;
}

function ViewportColumn({
  indexLabel,
  title,
  dataUrl,
  imageData,
  inference,
  accentHex,
  accentClass,
  cursor,
  headerRight,
  emptyHint,
  onCursorMove,
  onCursorLeave,
  onCursorClick,
}: ViewportColumnProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [frameSize, setFrameSize] = useState<{ w: number; h: number }>({ w: 224, h: 224 });

  useEffect(() => {
    const el = frameRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const obs = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      setFrameSize({ w: entry.contentRect.width, h: entry.contentRect.height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [dataUrl]);

  const computeCoords = (e: React.MouseEvent<HTMLDivElement>): { x: number; y: number } | null => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    const hit = clientPointToImagePixel(e.clientX, e.clientY, rect, IMG_SIZE, IMG_SIZE);
    if (!hit || !hit.inside) return null;
    return { x: hit.x, y: hit.y };
  };

  const crosshairPos = imagePixelToContainerPercent(
    cursor.x,
    cursor.y,
    frameSize.w,
    frameSize.h,
    IMG_SIZE,
    IMG_SIZE,
  );

  // Auto-reposition loupe inset so it never occludes the cursor
  const loupeInTopLeft = cursor.x > IMG_SIZE * 0.55 && cursor.y > IMG_SIZE * 0.55;
  const top1 = inference?.top_k[0];

  return (
    <div className="flex flex-col min-w-0 min-h-0 bg-lab-bg">
      {/* Column Instrument Header */}
      <div className="h-8 shrink-0 px-2.5 border-b border-lab-border bg-lab-surface flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="font-mono text-[10px] text-lab-subtle">{indexLabel}</span>
          <span className="font-sans text-[11px] font-semibold tracking-[0.07em] text-lab-text truncate">
            {title}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {headerRight}
          {top1 ? (
            <span
              className={`font-mono text-[10px] tabular-nums truncate max-w-[130px] ${accentClass}`}
              title={`${top1.label} (${(top1.probability * 100).toFixed(2)}%) · ${inference?.latency_ms}ms`}
            >
              {top1.label} · {(top1.probability * 100).toFixed(1)}%
            </span>
          ) : (
            <span className="font-mono text-[10px] text-lab-subtle">—</span>
          )}
        </div>
      </div>

      {/* Square Pixel Viewport */}
      <div className="flex-1 min-h-0 p-2.5 flex items-center justify-center overflow-hidden">
        {dataUrl ? (
          <div
            ref={frameRef}
            onMouseMove={(e) => {
              const coords = computeCoords(e);
              if (coords) {
                onCursorMove(coords.x, coords.y);
              } else {
                onCursorLeave();
              }
            }}
            onMouseLeave={onCursorLeave}
            onClick={(e) => {
              const coords = computeCoords(e);
              if (coords) onCursorClick(coords.x, coords.y);
            }}
            className="relative aspect-square h-full max-h-full max-w-full border border-lab-borderStrong bg-black cursor-crosshair select-none overflow-hidden"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={dataUrl}
              alt={title}
              draggable={false}
              className="w-full h-full object-contain block"
              style={{ imageRendering: "pixelated" }}
            />

            {/* Synchronized Hairline Crosshair (object-contain calibrated) */}
            {cursor.active && (
              <>
                <div
                  className="pointer-events-none absolute top-0 bottom-0 w-px"
                  style={{
                    left: `${crosshairPos.leftPct}%`,
                    backgroundColor: accentHex,
                    opacity: 0.7,
                  }}
                />
                <div
                  className="pointer-events-none absolute left-0 right-0 h-px"
                  style={{
                    top: `${crosshairPos.topPct}%`,
                    backgroundColor: accentHex,
                    opacity: 0.7,
                  }}
                />
                <div
                  className="pointer-events-none absolute w-2.5 h-2.5 -translate-x-1/2 -translate-y-1/2 border"
                  style={{
                    left: `${crosshairPos.leftPct}%`,
                    top: `${crosshairPos.topPct}%`,
                    borderColor: accentHex,
                  }}
                />
              </>
            )}

            {/* Synchronized 11x11 Pixel Loupe Inset */}
            {cursor.active && imageData && (
              <div
                className={`pointer-events-none absolute z-20 ${
                  loupeInTopLeft ? "top-2 left-2" : "bottom-2 right-2"
                }`}
              >
                <PixelLoupeInset
                  imageData={imageData}
                  x={cursor.x}
                  y={cursor.y}
                  accentColor={accentHex}
                />
              </div>
            )}
          </div>
        ) : (
          <div className="aspect-square h-full max-h-full max-w-full border border-dashed border-lab-border flex flex-col items-center justify-center gap-1.5 text-center px-4">
            <Crosshair className="w-4 h-4 text-lab-subtle" />
            <span className="font-mono text-[10px] text-lab-subtle tracking-wide">
              {emptyHint}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export function SynchronizedViewport({
  images,
  cleanInference,
  adversarialInference,
  purifiedInference,
}: SynchronizedViewportProps) {
  const [showResidual, setShowResidual] = useState(false);
  const [showGradCam, setShowGradCam] = useState(false);
  const [cursor, setCursor] = useState<CursorState>({
    x: 112,
    y: 112,
    active: false,
    locked: false,
  });

  const cleanData = useDecodedImageData(images?.clean_png ?? null);
  const advData = useDecodedImageData(images?.adversarial_png ?? null);
  const residualData = useDecodedImageData(images?.residual_png ?? null);
  const purifiedData = useDecodedImageData(images?.purified_png ?? null);
  const cleanCamData = useDecodedImageData(images?.clean_cam_png ?? null);
  const advCamData = useDecodedImageData(images?.adversarial_cam_png ?? null);
  const purifiedCamData = useDecodedImageData(images?.purified_cam_png ?? null);

  const handleMove = useCallback((x: number, y: number) => {
    setCursor((prev) => (prev.locked ? prev : { x, y, active: true, locked: false }));
  }, []);

  const handleLeave = useCallback(() => {
    setCursor((prev) => (prev.locked ? prev : { ...prev, active: false }));
  }, []);

  const handleClick = useCallback((x: number, y: number) => {
    setCursor((prev) =>
      prev.locked
        ? { x, y, active: true, locked: false }
        : { x, y, active: true, locked: true },
    );
  }, []);

  // Arrow-key 1px micro-stepping when loupe is locked
  useEffect(() => {
    if (!cursor.locked) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setCursor((prev) => ({ ...prev, locked: false, active: false }));
        return;
      }
      const deltaMap: Record<string, [number, number]> = {
        ArrowUp: [0, -1],
        ArrowDown: [0, 1],
        ArrowLeft: [-1, 0],
        ArrowRight: [1, 0],
      };
      const delta = deltaMap[e.key];
      if (!delta) return;
      e.preventDefault();
      setCursor((prev) => ({
        ...prev,
        x: Math.max(0, Math.min(IMG_SIZE - 1, prev.x + delta[0])),
        y: Math.max(0, Math.min(IMG_SIZE - 1, prev.y + delta[1])),
      }));
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [cursor.locked]);

  const cleanPx = cursor.active ? samplePixel(cleanData, cursor.x, cursor.y) : null;
  const advPx = cursor.active ? samplePixel(advData, cursor.x, cursor.y) : null;
  const purPx = cursor.active ? samplePixel(purifiedData, cursor.x, cursor.y) : null;

  const hasCam = Boolean(images?.clean_cam_png);

  return (
    <section className="flex-[57] min-h-0 flex flex-col border-b border-lab-border">
      {/* Three Equal Columns */}
      <div className="flex-1 min-h-0 grid grid-cols-3 divide-x divide-lab-border">
        <ViewportColumn
          indexLabel="01"
          title={showGradCam ? "ORIGINAL · CAM" : "ORIGINAL (CLEAN)"}
          dataUrl={showGradCam ? (images?.clean_cam_png ?? images?.clean_png ?? null) : (images?.clean_png ?? null)}
          imageData={showGradCam ? (cleanCamData ?? cleanData) : cleanData}
          inference={cleanInference}
          accentHex="#10B981"
          accentClass="text-lab-signal"
          cursor={cursor}
          emptyHint="LOAD MODEL + TARGET IMAGE"
          onCursorMove={handleMove}
          onCursorLeave={handleLeave}
          onCursorClick={handleClick}
        />

        <ViewportColumn
          indexLabel="02"
          title={
            showResidual
              ? "RESIDUAL ×12"
              : showGradCam
                ? "ADVERSARIAL · CAM"
                : "ADVERSARIAL (ATTACKED)"
          }
          dataUrl={
            showResidual
              ? (images?.residual_png ?? null)
              : showGradCam
                ? (images?.adversarial_cam_png ?? images?.adversarial_png ?? null)
                : (images?.adversarial_png ?? null)
          }
          imageData={
            showResidual
              ? residualData
              : showGradCam
                ? (advCamData ?? advData)
                : advData
          }
          inference={adversarialInference}
          accentHex="#EF4444"
          accentClass="text-lab-alert"
          cursor={cursor}
          emptyHint="RUN FGSM / PGD / PATCH ATTACK"
          headerRight={
            images?.residual_png ? (
              <div className="flex border border-lab-border bg-lab-bg">
                <button
                  type="button"
                  onClick={() => setShowResidual(false)}
                  className={`px-1.5 py-0.5 font-mono text-[9px] ${
                    !showResidual ? "bg-lab-elevated text-white" : "text-lab-subtle"
                  }`}
                >
                  {showGradCam ? "CAM" : "IMG"}
                </button>
                <button
                  type="button"
                  onClick={() => setShowResidual(true)}
                  className={`px-1.5 py-0.5 font-mono text-[9px] ${
                    showResidual ? "bg-lab-alert/20 text-lab-alert" : "text-lab-subtle"
                  }`}
                >
                  Δ×12
                </button>
              </div>
            ) : null
          }
          onCursorMove={handleMove}
          onCursorLeave={handleLeave}
          onCursorClick={handleClick}
        />

        <ViewportColumn
          indexLabel="03"
          title={showGradCam ? "PURIFIED · CAM" : "PURIFIED (RECONSTRUCTED)"}
          dataUrl={
            showGradCam
              ? (images?.purified_cam_png ?? images?.purified_png ?? null)
              : (images?.purified_png ?? null)
          }
          imageData={showGradCam ? (purifiedCamData ?? purifiedData) : purifiedData}
          inference={purifiedInference}
          accentHex="#38BDF8"
          accentClass="text-lab-cutoff"
          cursor={cursor}
          emptyHint="APPLY SVD / 2D-FFT PURIFICATION"
          onCursorMove={handleMove}
          onCursorLeave={handleLeave}
          onCursorClick={handleClick}
        />
      </div>

      {/* 28px Synchronized Pixel Loupe Oscilloscope Telemetry Strip */}
      <div className="h-7 shrink-0 px-3 border-t border-lab-border bg-lab-surface flex items-center justify-between gap-4 font-mono text-[10px] tabular-nums select-none overflow-hidden">
        <div className="flex items-center gap-3 min-w-0">
          <span className="inline-flex items-center gap-1 text-lab-muted shrink-0">
            {cursor.locked ? (
              <Lock className="w-3 h-3 text-lab-cutoff" />
            ) : (
              <Unlock className="w-3 h-3 text-lab-subtle" />
            )}
            <span>LOUPE 11×11</span>
          </span>

          {cursor.active ? (
            <>
              <span className="text-lab-text shrink-0">
                POS({String(cursor.x).padStart(3, "0")},{String(cursor.y).padStart(3, "0")})
              </span>

              {cleanPx && (
                <span className="text-lab-signal shrink-0">
                  CLEAN({String(cleanPx.r).padStart(3, "0")},{String(cleanPx.g).padStart(3, "0")},
                  {String(cleanPx.b).padStart(3, "0")})
                </span>
              )}

              {advPx && cleanPx && (
                <span className="text-lab-alert shrink-0">
                  ΔADV({formatDelta(advPx.r - cleanPx.r)},{formatDelta(advPx.g - cleanPx.g)},
                  {formatDelta(advPx.b - cleanPx.b)})
                </span>
              )}

              {purPx && cleanPx && (
                <span className="text-lab-cutoff shrink-0">
                  ΔPUR({formatDelta(purPx.r - cleanPx.r)},{formatDelta(purPx.g - cleanPx.g)},
                  {formatDelta(purPx.b - cleanPx.b)})
                </span>
              )}
            </>
          ) : (
            <span className="text-lab-subtle">
              HOVER ANY IMAGE TO INSPECT SYNCHRONIZED PIXELS · CLICK TO LOCK + ARROW KEYS TO STEP 1PX
            </span>
          )}
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          <div className="flex border border-lab-border bg-lab-bg">
            <button
              type="button"
              onClick={() => setShowGradCam(false)}
              className={`px-1.5 py-0.5 font-mono text-[9px] ${
                !showGradCam ? "bg-lab-elevated text-white" : "text-lab-subtle hover:text-lab-muted"
              }`}
            >
              RGB
            </button>
            <button
              type="button"
              disabled={!hasCam}
              onClick={() => {
                setShowResidual(false);
                setShowGradCam(true);
              }}
              className={`px-1.5 py-0.5 font-mono text-[9px] disabled:opacity-35 ${
                showGradCam ? "bg-lab-warn/20 text-lab-warn" : "text-lab-subtle hover:text-lab-muted"
              }`}
              title="Toggle synchronized Grad-CAM attention overlay across all 3 columns"
            >
              GRAD-CAM
            </button>
          </div>
          <span className="text-lab-subtle hidden xl:inline">224×224 · PNG</span>
        </div>
      </div>
    </section>
  );
}
