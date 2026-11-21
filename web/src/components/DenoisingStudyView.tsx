"use client";

import { useCallback, useEffect, useState } from "react";
import { Lock, Unlock } from "lucide-react";
import { StudyColumnCard, type StudyCursor } from "@/components/StudyColumnCard";
import { StudyZoomDeck, type ZoomColumnItem } from "@/components/StudyZoomDeck";
import { sampleLumaAt, useDecodedRgba } from "@/lib/useImageData";
import type { StudyDataBundle, StudyMeasurementRow } from "@/types/workbench";

export function DenoisingStudyView() {
  const [bundle, setBundle] = useState<StudyDataBundle | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [targetId, setTargetId] = useState<string>("harmonic_grid");
  const [focusMode, setFocusMode] = useState<"detail" | "ref">("detail");
  const [selectedSeed, setSelectedSeed] = useState<number>(42);
  const [cursor, setCursor] = useState<StudyCursor>({
    x: 186,
    y: 112,
    active: false,
    locked: false,
  });

  useEffect(() => {
    let active = true;
    fetch("/study/study-data.json")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<StudyDataBundle>;
      })
      .then((data) => {
        if (!active) return;
        setBundle(data);
        setSelectedSeed(data.primary_seed);
      })
      .catch((err: unknown) => {
        if (active) setLoadError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, []);

  const target =
    bundle?.targets.find((t) => t.preset_id === targetId) ?? bundle?.targets[0] ?? null;

  const activeImages =
    target?.images_by_seed?.[String(selectedSeed)] ?? target?.images ?? null;

  const cleanData = useDecodedRgba(activeImages?.clean ?? null);
  const noisyData = useDecodedRgba(activeImages?.noisy ?? null);
  const svdData = useDecodedRgba(activeImages?.svd ?? null);
  const fftData = useDecodedRgba(activeImages?.fft ?? null);

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

  if (loadError) {
    return (
      <div className="flex-1 flex items-center justify-center font-mono text-xs text-lab-alert">
        Failed to load static study data ({loadError}). Run &apos;make detail-study&apos;.
      </div>
    );
  }

  if (!bundle || !target) {
    return (
      <div className="flex-1 flex items-center justify-center font-mono text-xs text-lab-muted">
        Loading pre-generated Denoising Study bundle...
      </div>
    );
  }

  const findRow = (prefix: string): StudyMeasurementRow | null =>
    bundle.rows.find(
      (r) => r.seed === selectedSeed && r.image === target.preset_id && r.method.startsWith(prefix),
    ) ?? null;

  const noisyRow = findRow("Noisy");
  const svdRow = findRow("SVD-only");
  const fftRow = findRow("2D-FFT");

  const zoomCols: ZoomColumnItem[] = [
    {
      key: "clean",
      label: "01 · CLEAN REFERENCE",
      imageData: cleanData,
      row: null,
      cleanContrastLuma: target.clean_feature_contrast_luma,
      accentClass: "text-lab-signal",
    },
    {
      key: "noisy",
      label: `02 · NOISY (σ=${bundle.sigma.toFixed(2)})`,
      imageData: noisyData,
      row: noisyRow,
      cleanContrastLuma: target.clean_feature_contrast_luma,
      accentClass: "text-lab-alert",
    },
    {
      key: "svd",
      label: `03 · SVD-ONLY (k=${bundle.svd_k})`,
      imageData: svdData,
      row: svdRow,
      cleanContrastLuma: target.clean_feature_contrast_luma,
      accentClass: "text-lab-warn",
    },
    {
      key: "fft",
      label: `04 · 2D-FFT GAUSS (r=${bundle.fft_r.toFixed(2)})`,
      imageData: fftData,
      row: fftRow,
      cleanContrastLuma: target.clean_feature_contrast_luma,
      accentClass: "text-lab-cutoff",
    },
  ];

  const cleanY = cursor.active ? sampleLumaAt(cleanData, cursor.x, cursor.y) : null;
  const noisyY = cursor.active ? sampleLumaAt(noisyData, cursor.x, cursor.y) : null;
  const svdY = cursor.active ? sampleLumaAt(svdData, cursor.x, cursor.y) : null;
  const fftY = cursor.active ? sampleLumaAt(fftData, cursor.x, cursor.y) : null;

  return (
    <div className="flex-1 lg:min-h-0 flex flex-col lg:overflow-hidden bg-lab-bg">
      {/* Study Controls & Scientific Framing Strip */}
      <div className="shrink-0 px-3 py-2 border-b border-lab-border bg-lab-surface flex flex-wrap items-center justify-between gap-2.5 select-none">
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Target Calibration Image Switcher */}
          <div className="flex border border-lab-borderStrong bg-lab-bg">
            {bundle.targets.map((t) => (
              <button
                key={t.preset_id}
                type="button"
                onClick={() => setTargetId(t.preset_id)}
                className={`px-2.5 py-1 font-sans text-[11px] font-semibold tracking-[0.04em] transition-colors ${
                  target.preset_id === t.preset_id
                    ? "bg-lab-elevated text-white"
                    : "text-lab-muted hover:text-lab-text"
                }`}
              >
                {t.preset_id === "harmonic_grid" ? "Harmonic Grid" : "Faro Beacon"}
              </button>
            ))}
          </div>

          {/* ROI Focus Selector */}
          <div className="flex border border-lab-border bg-lab-bg">
            <button
              type="button"
              onClick={() => setFocusMode("detail")}
              className={`px-2 py-1 font-mono text-[10px] transition-colors ${
                focusMode === "detail"
                  ? "bg-lab-warn/20 text-lab-warn"
                  : "text-lab-subtle hover:text-lab-muted"
              }`}
            >
              AMBER · DETAIL ROI
            </button>
            <button
              type="button"
              onClick={() => setFocusMode("ref")}
              className={`px-2 py-1 font-mono text-[10px] transition-colors ${
                focusMode === "ref"
                  ? "bg-lab-signal/20 text-lab-signal"
                  : "text-lab-subtle hover:text-lab-muted"
              }`}
            >
              GREEN · REF ROI
            </button>
          </div>

          {/* Multi-seed metric verification switcher */}
          <div className="flex items-center gap-1 border border-lab-border bg-lab-bg px-1.5 py-0.5">
            <span className="font-mono text-[9px] text-lab-subtle mr-1">SEED</span>
            {bundle.seeds.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSelectedSeed(s)}
                className={`px-1.5 py-0.5 font-mono text-[10px] tabular-nums ${
                  selectedSeed === s
                    ? "bg-lab-elevated text-lab-cutoff"
                    : "text-lab-subtle hover:text-lab-muted"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <div className="font-mono text-[10px] text-lab-muted tabular-nums">
          SYNTHETIC CALIBRATION IMAGES · σ={bundle.sigma.toFixed(2)} · SEED={selectedSeed} · STATIC PRECOMPUTED
        </div>
      </div>

      {/* Upper Deck: 4-Column Full-Image Comparison Matrix */}
      <section className="lg:flex-[54] lg:min-h-0 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-lab-border border-b border-lab-border">
        <StudyColumnCard
          indexLabel="01"
          title="CLEAN REFERENCE"
          subtitle="PSNR ∞"
          imageUrl={activeImages?.clean ?? target.images.clean}
          accentHex="#10B981"
          accentClass="text-lab-signal"
          refRoi={target.ref_roi}
          detailRoi={target.detail_roi}
          activeFocus={focusMode}
          cursor={cursor}
          onSelectFocus={setFocusMode}
          onCursorMove={handleMove}
          onCursorLeave={handleLeave}
          onCursorClick={handleClick}
        />
        <StudyColumnCard
          indexLabel="02"
          title="NOISY INPUT"
          subtitle={noisyRow ? `${noisyRow.psnr_db.toFixed(2)} dB` : "—"}
          imageUrl={activeImages?.noisy ?? target.images.noisy}
          accentHex="#EF4444"
          accentClass="text-lab-alert"
          refRoi={target.ref_roi}
          detailRoi={target.detail_roi}
          activeFocus={focusMode}
          cursor={cursor}
          onSelectFocus={setFocusMode}
          onCursorMove={handleMove}
          onCursorLeave={handleLeave}
          onCursorClick={handleClick}
        />
        <StudyColumnCard
          indexLabel="03"
          title={`SVD-ONLY (k=${bundle.svd_k})`}
          subtitle={
            svdRow
              ? `${svdRow.psnr_db.toFixed(2)} dB · ${svdRow.feature_contrast_retained_pct.toFixed(1)}%`
              : "—"
          }
          imageUrl={activeImages?.svd ?? target.images.svd}
          accentHex="#F59E0B"
          accentClass="text-lab-warn"
          refRoi={target.ref_roi}
          detailRoi={target.detail_roi}
          activeFocus={focusMode}
          cursor={cursor}
          onSelectFocus={setFocusMode}
          onCursorMove={handleMove}
          onCursorLeave={handleLeave}
          onCursorClick={handleClick}
        />
        <StudyColumnCard
          indexLabel="04"
          title={`2D-FFT GAUSS (r=${bundle.fft_r.toFixed(2)})`}
          subtitle={
            fftRow
              ? `${fftRow.psnr_db.toFixed(2)} dB · ${fftRow.feature_contrast_retained_pct.toFixed(1)}%`
              : "—"
          }
          imageUrl={activeImages?.fft ?? target.images.fft}
          accentHex="#38BDF8"
          accentClass="text-lab-cutoff"
          refRoi={target.ref_roi}
          detailRoi={target.detail_roi}
          activeFocus={focusMode}
          cursor={cursor}
          onSelectFocus={setFocusMode}
          onCursorMove={handleMove}
          onCursorLeave={handleLeave}
          onCursorClick={handleClick}
        />
      </section>

      {/* Synchronized 4-Column Pixel Telemetry Strip */}
      <div className="h-7 shrink-0 px-3 border-b border-lab-border bg-lab-surface flex items-center justify-between gap-3 font-mono text-[10px] tabular-nums select-none overflow-hidden">
        <div className="flex items-center gap-3 min-w-0">
          <span className="inline-flex items-center gap-1 text-lab-muted shrink-0">
            {cursor.locked ? (
              <Lock className="w-3 h-3 text-lab-cutoff" />
            ) : (
              <Unlock className="w-3 h-3 text-lab-subtle" />
            )}
            <span>SYNC CURSOR</span>
          </span>
          {cursor.active && cleanY !== null ? (
            <>
              <span className="text-lab-text shrink-0">
                POS({String(cursor.x).padStart(3, "0")},{String(cursor.y).padStart(3, "0")})
              </span>
              <span className="text-lab-signal shrink-0">CLN Y={cleanY.toFixed(3)}</span>
              {noisyY !== null && (
                <span className="text-lab-alert shrink-0">NOISY Y={noisyY.toFixed(3)}</span>
              )}
              {svdY !== null && (
                <span className="text-lab-warn shrink-0">SVD Y={svdY.toFixed(3)}</span>
              )}
              {fftY !== null && (
                <span className="text-lab-cutoff shrink-0">FFT Y={fftY.toFixed(3)}</span>
              )}
            </>
          ) : (
            <span className="text-lab-subtle truncate">
              CLICK THE AMBER OR GREEN BOX ON ANY IMAGE TO SWITCH SYNCHRONIZED 4-COLUMN ZOOM · HOVER TO SAMPLE PIXEL LUMA
            </span>
          )}
        </div>
        <span className="text-lab-subtle shrink-0 hidden xl:inline">
          *Pipeline latency (CPU, incl. diagnostics)
        </span>
      </div>

      {/* Lower Deck: Synchronized 4-Column ROI Magnifier + Paired PSNR / Contrast Readouts */}
      <div className="lg:flex-[46] lg:min-h-0 flex flex-col">
        <StudyZoomDeck
          columns={zoomCols}
          roi={focusMode === "detail" ? target.detail_roi : target.ref_roi}
          focusMode={focusMode}
          featureLabel={target.feature_label}
          selectedSeed={selectedSeed}
        />
      </div>
    </div>
  );
}
