"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { SpectralSummary } from "@/types/workbench";

interface SvdCutoffChartProps {
  spectral: SpectralSummary | null;
  currentRank: number;
  interactive: boolean;
  onPreviewRankChange: (rank: number) => void;
  onCommitRank: (rank: number) => void;
}

const CHART_W = 540;
const CHART_H = 196;
const PAD_L = 42;
const PAD_R = 16;
const PAD_T = 14;
const PAD_B = 26;
const PLOT_W = CHART_W - PAD_L - PAD_R;
const PLOT_H = CHART_H - PAD_T - PAD_B;

export function SvdCutoffChart({
  spectral,
  currentRank,
  interactive,
  onPreviewRankChange,
  onCommitRank,
}: SvdCutoffChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [dragRank, setDragRank] = useState<number>(currentRank);

  useEffect(() => {
    if (!dragging) {
      setDragRank(currentRank);
    }
  }, [currentRank, dragging]);

  const svals = spectral?.singular_values ?? [];
  const cumEnergy = spectral?.cumulative_energy ?? [];
  const maxRank = Math.max(svals.length, 1);
  const activeRank = Math.max(1, Math.min(maxRank, dragging ? dragRank : currentRank));

  const geometry = useMemo(() => {
    if (svals.length === 0) return null;
    const eps = 1e-3;
    const logVals = svals.map((v) => Math.log10(Math.max(v, eps)));
    const minLog = Math.min(...logVals);
    const maxLog = Math.max(...logVals);
    const span = Math.max(maxLog - minLog, 1e-3);

    const xForIndex = (idx: number) =>
      PAD_L + (idx / Math.max(svals.length - 1, 1)) * PLOT_W;
    const yForLog = (lv: number) =>
      PAD_T + (1 - (lv - minLog) / span) * PLOT_H;
    const yForEnergy = (e: number) => PAD_T + (1 - e) * PLOT_H;

    const fullPath = logVals
      .map((lv, i) => `${i === 0 ? "M" : "L"}${xForIndex(i).toFixed(1)},${yForLog(lv).toFixed(1)}`)
      .join(" ");

    const retainedCount = Math.max(1, Math.min(svals.length, activeRank));
    const retainedPts = logVals
      .slice(0, retainedCount)
      .map((lv, i) => `${xForIndex(i).toFixed(1)},${yForLog(lv).toFixed(1)}`);

    const cutoffX = xForIndex(retainedCount - 1);
    const baselineY = PAD_T + PLOT_H;
    const areaPath = `M${PAD_L},${baselineY} L${retainedPts.join(" L")} L${cutoffX.toFixed(1)},${baselineY} Z`;

    const energyPath = cumEnergy
      .map((e, i) => `${i === 0 ? "M" : "L"}${xForIndex(i).toFixed(1)},${yForEnergy(e).toFixed(1)}`)
      .join(" ");

    return {
      fullPath,
      areaPath,
      energyPath,
      cutoffX,
      cutoffY: yForLog(logVals[retainedCount - 1]),
      minLog,
      maxLog,
      retainedSigma: svals[retainedCount - 1],
      retainedEnergy: cumEnergy[retainedCount - 1] ?? 1.0,
    };
  }, [svals, cumEnergy, activeRank]);

  const rankFromClientX = (clientX: number): number => {
    const svg = svgRef.current;
    if (!svg || svals.length === 0) return activeRank;
    const rect = svg.getBoundingClientRect();
    const relX = ((clientX - rect.left) / rect.width) * CHART_W;
    const t = Math.max(0, Math.min(1, (relX - PAD_L) / PLOT_W));
    return Math.max(2, Math.min(svals.length, Math.round(t * (svals.length - 1)) + 1));
  };

  const handlePointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!interactive || svals.length === 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const next = rankFromClientX(e.clientX);
    setDragging(true);
    setDragRank(next);
    onPreviewRankChange(next);
  };

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging) return;
    const next = rankFromClientX(e.clientX);
    setDragRank(next);
    onPreviewRankChange(next);
  };

  const handlePointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging) return;
    e.currentTarget.releasePointerCapture(e.pointerId);
    const finalRank = rankFromClientX(e.clientX);
    setDragging(false);
    setDragRank(finalRank);
    onCommitRank(finalRank);
  };

  return (
    <div className="flex flex-col min-w-0 min-h-0 bg-lab-bg">
      {/* Header */}
      <div className="h-7 shrink-0 px-2.5 border-b border-lab-border bg-lab-surface flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-sans text-[11px] font-semibold tracking-[0.07em] text-lab-text">
            SVD SINGULAR VALUE SPECTRUM
          </span>
          <span className="font-mono text-[9px] text-lab-subtle hidden xl:inline">
            DRAG CUTOFF LINE TO RE-PURIFY
          </span>
        </div>
        {geometry ? (
          <div className="flex items-center gap-2.5 font-mono text-[10px] tabular-nums">
            <span className="text-lab-cutoff">
              k={activeRank}/{svals.length}
            </span>
            <span className="text-lab-signal">
              E={(geometry.retainedEnergy * 100).toFixed(2)}%
            </span>
            <span className="text-lab-muted hidden sm:inline">
              σ_k={geometry.retainedSigma.toFixed(2)}
            </span>
          </div>
        ) : (
          <span className="font-mono text-[10px] text-lab-subtle">NO SPECTRUM</span>
        )}
      </div>

      {/* Interactive Scree Plot SVG */}
      <div className="flex-1 min-h-0 p-2 flex items-center justify-center">
        {geometry ? (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            preserveAspectRatio="none"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            className={`w-full h-full select-none ${
              interactive ? "cursor-ew-resize" : "cursor-default"
            }`}
          >
            {/* Horizontal log gridlines */}
            {[0, 0.25, 0.5, 0.75, 1].map((t) => {
              const y = PAD_T + t * PLOT_H;
              const logTick = geometry.maxLog - t * (geometry.maxLog - geometry.minLog);
              return (
                <g key={t}>
                  <line
                    x1={PAD_L}
                    y1={y}
                    x2={CHART_W - PAD_R}
                    y2={y}
                    stroke="#1D212A"
                    strokeWidth="1"
                  />
                  <text
                    x={PAD_L - 6}
                    y={y + 3}
                    textAnchor="end"
                    fill="#596070"
                    fontSize="9"
                    fontFamily='"Fragment Mono", ui-monospace, monospace'
                  >
                    10^{logTick.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Vertical rank ticks */}
            {[1, 56, 112, 168, svals.length].map((tick) => {
              const x = PAD_L + ((tick - 1) / Math.max(svals.length - 1, 1)) * PLOT_W;
              return (
                <g key={tick}>
                  <line
                    x1={x}
                    y1={PAD_T}
                    x2={x}
                    y2={PAD_T + PLOT_H}
                    stroke="#171A22"
                    strokeWidth="1"
                  />
                  <text
                    x={x}
                    y={CHART_H - 8}
                    textAnchor="middle"
                    fill="#596070"
                    fontSize="9"
                    fontFamily='"Fragment Mono", ui-monospace, monospace'
                  >
                    {tick}
                  </text>
                </g>
              );
            })}

            {/* Retained Subspace Shaded Area */}
            <path d={geometry.areaPath} fill="rgba(56, 189, 248, 0.14)" />

            {/* Cumulative Energy Dashed Curve (0..100%) */}
            <path
              d={geometry.energyPath}
              fill="none"
              stroke="rgba(16, 185, 129, 0.55)"
              strokeWidth="1.2"
              strokeDasharray="3 3"
            />

            {/* Full Singular Value Decay Curve */}
            <path
              d={geometry.fullPath}
              fill="none"
              stroke="#E6E8EC"
              strokeWidth="1.5"
            />

            {/* Draggable Vertical Cutoff Line */}
            <line
              x1={geometry.cutoffX}
              y1={PAD_T}
              x2={geometry.cutoffX}
              y2={PAD_T + PLOT_H}
              stroke="#38BDF8"
              strokeWidth={dragging ? "2" : "1.5"}
            />

            {/* Cutoff Intersection Marker & Handle */}
            <circle
              cx={geometry.cutoffX}
              cy={geometry.cutoffY}
              r="3.5"
              fill="#090A0C"
              stroke="#38BDF8"
              strokeWidth="1.5"
            />
            <rect
              x={geometry.cutoffX - 15}
              y={PAD_T - 2}
              width="30"
              height="13"
              fill="#38BDF8"
              rx="1"
            />
            <text
              x={geometry.cutoffX}
              y={PAD_T + 7.5}
              textAnchor="middle"
              fill="#090A0C"
              fontSize="8.5"
              fontWeight="700"
              fontFamily='"Fragment Mono", ui-monospace, monospace'
            >
              k={activeRank}
            </text>

            {/* Axis border */}
            <rect
              x={PAD_L}
              y={PAD_T}
              width={PLOT_W}
              height={PLOT_H}
              fill="none"
              stroke="#262B36"
              strokeWidth="1"
            />
          </svg>
        ) : (
          <div className="w-full h-full border border-dashed border-lab-border flex items-center justify-center font-mono text-[10px] text-lab-subtle">
            APPLY PURIFICATION TO PLOT SINGULAR VALUE DECAY log₁₀(σᵢ)
          </div>
        )}
      </div>
    </div>
  );
}
