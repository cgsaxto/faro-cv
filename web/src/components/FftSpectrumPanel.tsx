"use client";

import { useEffect, useState } from "react";
import type { DefenseConfig, ImageBundle, SpectralSummary } from "@/types/workbench";

interface FftSpectrumPanelProps {
  images: ImageBundle | null;
  spectral: SpectralSummary | null;
  appliedDefenseConfig: DefenseConfig | null;
}

export function FftSpectrumPanel({
  images,
  spectral,
  appliedDefenseConfig,
}: FftSpectrumPanelProps) {
  const hasBefore = Boolean(images?.fft_before_png);
  const hasAfter = Boolean(images?.fft_after_png);
  const [view, setView] = useState<"before" | "after">("before");

  // Auto-select PUR once post-purification spectrum becomes ready; revert to ADV when cleared
  useEffect(() => {
    if (hasAfter) {
      setView("after");
    } else {
      setView("before");
    }
  }, [hasAfter]);

  const effectiveView: "before" | "after" = view === "after" && hasAfter ? "after" : "before";
  const activeUrl =
    effectiveView === "after"
      ? (images?.fft_after_png ?? null)
      : (images?.fft_before_png ?? null);

  // Read strictly from the committed FSM defense_config that produced the current spectrum
  const activeFftParams =
    effectiveView === "after" &&
    appliedDefenseConfig &&
    (appliedDefenseConfig.kind === "fft" || appliedDefenseConfig.kind === "pipeline")
      ? appliedDefenseConfig.fft
      : null;

  const isBandstop = activeFftParams?.filter_type === "bandstop";
  const svgRadius = activeFftParams
    ? Math.min(48, Math.max(4, activeFftParams.cutoff_radius * 50))
    : null;
  const innerStop = activeFftParams
    ? Math.max(2, (activeFftParams.cutoff_radius - activeFftParams.band_width) * 50)
    : 0;
  const outerStop = activeFftParams
    ? Math.min(49, (activeFftParams.cutoff_radius + activeFftParams.band_width) * 50)
    : 0;

  const hfBefore = spectral ? spectral.high_freq_energy_before * 100 : null;
  const hfAfter = spectral ? spectral.high_freq_energy_after * 100 : null;

  return (
    <div className="flex flex-col min-w-0 min-h-0 bg-lab-bg">
      <div className="h-7 shrink-0 px-2.5 border-b border-lab-border bg-lab-surface flex items-center justify-between gap-2">
        <span className="font-sans text-[11px] font-semibold tracking-[0.07em] text-lab-text truncate">
          2D-FFT LOG SPECTRUM
        </span>

        <div className="flex border border-lab-border bg-lab-bg shrink-0">
          <button
            type="button"
            disabled={!hasBefore}
            onClick={() => setView("before")}
            className={`px-1.5 py-0.5 font-mono text-[9px] uppercase disabled:opacity-35 disabled:cursor-not-allowed ${
              effectiveView === "before" && hasBefore
                ? "bg-lab-elevated text-lab-alert"
                : "text-lab-subtle hover:text-lab-muted"
            }`}
          >
            ADV
          </button>
          <button
            type="button"
            disabled={!hasAfter}
            onClick={() => setView("after")}
            title={hasAfter ? "Show purified spectrum" : "Run purification to unlock PUR spectrum"}
            className={`px-1.5 py-0.5 font-mono text-[9px] uppercase disabled:opacity-35 disabled:cursor-not-allowed ${
              effectiveView === "after" && hasAfter
                ? "bg-lab-elevated text-lab-cutoff"
                : "text-lab-subtle hover:text-lab-muted"
            }`}
          >
            PUR
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 p-2 flex flex-col items-center justify-between gap-1.5 overflow-hidden">
        <div className="flex-1 min-h-0 w-full flex items-center justify-center">
          {activeUrl ? (
            <div className="relative aspect-square h-full max-h-full max-w-full border border-lab-borderStrong bg-black overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={activeUrl}
                alt={
                  effectiveView === "after"
                    ? "2D-FFT Purified Log Magnitude Spectrum"
                    : "2D-FFT Adversarial (Pre-Purification) Log Magnitude Spectrum"
                }
                className="w-full h-full object-contain block"
              />
              <svg
                viewBox="0 0 100 100"
                className="pointer-events-none absolute inset-0 w-full h-full"
              >
                <line
                  x1="50"
                  y1="0"
                  x2="50"
                  y2="100"
                  stroke="rgba(230,232,236,0.16)"
                  strokeWidth="0.6"
                />
                <line
                  x1="0"
                  y1="50"
                  x2="100"
                  y2="50"
                  stroke="rgba(230,232,236,0.16)"
                  strokeWidth="0.6"
                />
                {activeFftParams && svgRadius !== null ? (
                  isBandstop ? (
                    <>
                      <circle
                        cx="50"
                        cy="50"
                        r={innerStop}
                        fill="none"
                        stroke="#F59E0B"
                        strokeWidth="0.9"
                        strokeDasharray="2 2"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r={outerStop}
                        fill="none"
                        stroke="#F59E0B"
                        strokeWidth="0.9"
                        strokeDasharray="2 2"
                      />
                    </>
                  ) : (
                    <circle
                      cx="50"
                      cy="50"
                      r={svgRadius}
                      fill="none"
                      stroke="#38BDF8"
                      strokeWidth="1"
                      strokeDasharray="2.5 2"
                    />
                  )
                ) : null}
              </svg>
              <div className="pointer-events-none absolute bottom-1 left-1 px-1 py-0.5 bg-lab-bg/85 border border-lab-border font-mono text-[8px] text-lab-subtle tabular-nums">
                {effectiveView === "before"
                  ? "ADV SPECTRUM (PRE-PURIFY)"
                  : activeFftParams
                    ? `APPLIED r=${activeFftParams.cutoff_radius.toFixed(2)} · ${activeFftParams.filter_type.toUpperCase()}`
                    : "NO FFT MASK (SVD-ONLY)"}
              </div>
            </div>
          ) : (
            <div className="aspect-square h-full max-h-full border border-dashed border-lab-border flex items-center justify-center font-mono text-[9px] text-lab-subtle text-center px-3">
              AWAITING 2D-FFT SPECTRUM
            </div>
          )}
        </div>

        <div className="w-full px-2 py-1 border border-lab-border bg-lab-surface font-mono text-[10px] flex items-center justify-between tabular-nums">
          <span className="text-lab-subtle">HF POWER</span>
          {hfBefore !== null && hfAfter !== null ? (
            <span>
              <span className="text-lab-alert">{hfBefore.toFixed(1)}%</span>
              <span className="text-lab-subtle mx-1">→</span>
              <span className="text-lab-cutoff">{hfAfter.toFixed(1)}%</span>
            </span>
          ) : (
            <span className="text-lab-subtle">—</span>
          )}
        </div>
      </div>
    </div>
  );
}
