"use client";

import { useState } from "react";
import { BookmarkPlus, Cpu, ExternalLink, History, RotateCcw } from "lucide-react";
import type {
  FSMWireState,
  LabState,
  OperatorDeviceProfile,
  SnapshotSummary,
} from "@/types/workbench";

const GITHUB_QUICKSTART_URL = "https://github.com/cgsaxto/faro-cv#quickstart";

interface TopInstrumentBarProps {
  activeView: "study" | "lab";
  onChangeView: (view: "study" | "lab") => void;
  fsm: FSMWireState | null;
  deviceProfile: OperatorDeviceProfile | null;
  snapshots: SnapshotSummary[];
  wsConnected: boolean;
  busy: boolean;
  onCaptureSnapshot: () => void;
  onRestoreSnapshot: (id: string) => void;
  onReset: () => void;
}

const PIPELINE_STAGES: { state: LabState; short: string }[] = [
  { state: "IDLE", short: "IDLE" },
  { state: "MODEL_READY", short: "MODEL" },
  { state: "INPUT_LOADED", short: "INPUT" },
  { state: "ATTACKED", short: "ATTACKED" },
  { state: "PURIFIED", short: "PURIFIED" },
  { state: "EVALUATED", short: "EVALUATED" },
];

const STAGE_ORDER: Record<LabState, number> = {
  IDLE: 0,
  MODEL_READY: 1,
  INPUT_LOADED: 2,
  ATTACKING: 2.5,
  PAUSED: 2.5,
  ATTACKED: 3,
  PURIFYING: 3.5,
  PURIFIED: 4,
  EVALUATED: 5,
  ERROR: -1,
};

export function TopInstrumentBar({
  activeView,
  onChangeView,
  fsm,
  deviceProfile,
  snapshots,
  wsConnected,
  busy,
  onCaptureSnapshot,
  onRestoreSnapshot,
  onReset,
}: TopInstrumentBarProps) {
  const [snapMenuOpen, setSnapMenuOpen] = useState(false);

  const currentState: LabState = fsm?.state ?? "IDLE";
  const currentRank = STAGE_ORDER[currentState] ?? 0;

  return (
    <header className="h-11 shrink-0 border-b border-lab-border bg-lab-surface px-3 flex items-center justify-between gap-3 select-none">
      {/* Brand Mark + Primary View Mode Switch */}
      <div className="flex items-center gap-2.5">
        <div className="w-5 h-5 border border-lab-borderStrong bg-lab-bg flex items-center justify-center">
          <svg viewBox="0 0 16 16" className="w-3.5 h-3.5 text-lab-text" fill="none">
            <path d="M8 2V14M2 8H14" stroke="currentColor" strokeWidth="1" strokeOpacity="0.4" />
            <circle cx="8" cy="8" r="3" stroke="#38BDF8" strokeWidth="1.25" />
            <circle cx="8" cy="8" r="0.9" fill="#E6E8EC" />
          </svg>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="font-sans text-[13px] font-bold tracking-[0.08em] text-lab-text">
            FARO-CV
          </span>
          <span className="font-mono text-[10px] text-lab-subtle tracking-tight hidden sm:inline">
            v0.1
          </span>
        </div>

        {/* Mode Switch: Precomputed Denoising Study (default) vs Live Interactive Lab */}
        <div className="flex border border-lab-borderStrong bg-lab-bg ml-1">
          <button
            type="button"
            onClick={() => onChangeView("study")}
            className={`h-6 px-2 sm:px-2.5 font-mono text-[10px] tracking-wide transition-colors ${
              activeView === "study"
                ? "bg-lab-elevated text-lab-cutoff font-medium"
                : "text-lab-muted hover:text-lab-text"
            }`}
          >
            <span className="hidden sm:inline">DENOISING STUDY</span>
            <span className="sm:hidden">STUDY</span>
          </button>
          <button
            type="button"
            onClick={() => onChangeView("lab")}
            title="Requires local FastAPI backend (make dev-api)"
            className={`h-6 px-2 sm:px-2.5 font-mono text-[10px] tracking-wide transition-colors ${
              activeView === "lab"
                ? "bg-lab-elevated text-white font-medium"
                : "text-lab-muted hover:text-lab-text"
            }`}
          >
            <span className="hidden sm:inline">INTERACTIVE LAB · LOCAL SETUP</span>
            <span className="sm:hidden">LAB · LOCAL</span>
          </button>
        </div>

        <a
          href={GITHUB_QUICKSTART_URL}
          target="_blank"
          rel="noopener noreferrer"
          title="Run the FastAPI backend locally via GitHub Quickstart"
          className="h-6 px-2 border border-lab-border bg-lab-bg hover:border-lab-borderStrong hover:text-white font-mono text-[10px] text-lab-muted inline-flex items-center gap-1 transition-colors shrink-0"
        >
          <span className="hidden md:inline">GITHUB QUICKSTART</span>
          <span className="md:hidden">SETUP</span>
          <ExternalLink className="w-2.5 h-2.5 text-lab-subtle" />
        </a>

        {activeView === "lab" && (
          <span
            className="inline-flex items-center gap-1 px-1.5 py-0.5 border border-lab-border bg-lab-bg font-mono text-[10px] text-lab-muted"
            title={wsConnected ? "WebSocket telemetry stream active" : "WebSocket offline"}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                wsConnected ? "bg-lab-signal" : "bg-lab-warn"
              }`}
            />
            {wsConnected ? "LIVE" : "REST"}
          </span>
        )}
      </div>

      {/* Live FSM Pipeline Strip (shown in Interactive Lab mode) */}
      {activeView === "lab" ? (
        <div className="hidden lg:flex items-center gap-1 px-2 py-1 border border-lab-border bg-lab-bg">
        <span className="font-mono text-[10px] text-lab-subtle mr-1.5">FSM</span>
        {PIPELINE_STAGES.map((stage, idx) => {
          const isCurrent = currentState === stage.state;
          const isPassed = currentRank > idx;
          const isTransientBefore =
            (currentState === "ATTACKING" && stage.state === "ATTACKED") ||
            (currentState === "PURIFYING" && stage.state === "PURIFIED");

          return (
            <div key={stage.state} className="flex items-center gap-1">
              <div
                className={`flex items-center gap-1 px-1.5 py-0.5 font-mono text-[10px] tracking-tight transition-colors ${
                  isCurrent
                    ? "bg-lab-elevated text-lab-text border border-lab-borderStrong"
                    : isTransientBefore
                      ? "bg-lab-warn/10 text-lab-warn border border-lab-warn/40"
                      : isPassed
                        ? "text-lab-muted"
                        : "text-lab-subtle"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isCurrent
                      ? currentState === "EVALUATED"
                        ? "bg-lab-signal"
                        : currentState === "ATTACKED"
                          ? "bg-lab-alert"
                          : "bg-lab-cutoff"
                      : isTransientBefore
                        ? "bg-lab-warn animate-pulse"
                        : isPassed
                          ? "bg-lab-muted/60"
                          : "bg-lab-border"
                  }`}
                />
                <span>{stage.short}</span>
              </div>
              {idx < PIPELINE_STAGES.length - 1 && (
                <span className="font-mono text-[9px] text-lab-borderStrong">→</span>
              )}
            </div>
          );
        })}

        {currentState === "ERROR" && (
          <span className="ml-1.5 px-1.5 py-0.5 font-mono text-[10px] bg-lab-alert/15 text-lab-alert border border-lab-alert/40">
            ERROR
          </span>
        )}
        <span className="ml-2 pl-2 border-l border-lab-border font-mono text-[10px] text-lab-subtle tabular-nums">
          #{String(fsm?.step_count ?? 0).padStart(2, "0")}
        </span>
      </div>
      ) : (
        <div className="hidden xl:flex items-center gap-2 font-mono text-[10px] text-lab-muted">
          <span>SVD (k=28) vs. 2D-FFT GAUSSIAN (r=0.35)</span>
          <span className="text-lab-borderStrong">·</span>
          <span className="text-lab-cutoff">EXACT DFT GRID (DC=1.0)</span>
        </div>
      )}

      {/* Hardware Telemetry & Session Controls (Interactive Lab) or Quick Switch (Study) */}
      {activeView === "lab" ? (
      <div className="flex items-center gap-2 relative">
        {deviceProfile && (
          <div
            className="hidden xl:flex items-center gap-1.5 px-2 py-1 border border-lab-border bg-lab-bg font-mono text-[10px] text-lab-muted tabular-nums"
            title={`PyTorch ${deviceProfile.torch_version ?? "N/A"} · Model ${deviceProfile.model_op.probe_latency_ms}ms · SVD ${deviceProfile.svd_op.probe_latency_ms}ms · FFT ${deviceProfile.fft_op.probe_latency_ms}ms`}
          >
            <Cpu className="w-3 h-3 text-lab-subtle" />
            <span>
              MDL:<strong className="text-lab-text uppercase">{deviceProfile.model_op.device}</strong>
            </span>
            <span className="text-lab-borderStrong">·</span>
            <span>
              SVD:<strong className="text-lab-text uppercase">{deviceProfile.svd_op.device}</strong>
            </span>
            <span className="text-lab-borderStrong">·</span>
            <span>
              FFT:<strong className="text-lab-text uppercase">{deviceProfile.fft_op.device}</strong>
            </span>
          </div>
        )}

        <button
          type="button"
          onClick={onCaptureSnapshot}
          disabled={busy || currentState === "IDLE"}
          className="h-7 px-2 border border-lab-border bg-lab-bg hover:border-lab-borderStrong hover:text-white disabled:opacity-35 disabled:cursor-not-allowed font-mono text-[11px] text-lab-muted flex items-center gap-1.5 transition-colors"
          title="Capture immutable FSM snapshot for A/B time-travel"
        >
          <BookmarkPlus className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">SNAP</span>
        </button>

        <div className="relative">
          <button
            type="button"
            onClick={() => setSnapMenuOpen((v) => !v)}
            disabled={snapshots.length === 0}
            className="h-7 px-2 border border-lab-border bg-lab-bg hover:border-lab-borderStrong hover:text-white disabled:opacity-35 disabled:cursor-not-allowed font-mono text-[11px] text-lab-muted flex items-center gap-1.5 tabular-nums transition-colors"
            title="Restore previous FSM snapshot"
          >
            <History className="w-3.5 h-3.5" />
            <span>{snapshots.length}</span>
          </button>

          {snapMenuOpen && snapshots.length > 0 && (
            <div className="absolute right-0 mt-1 w-64 border border-lab-borderStrong bg-lab-elevated shadow-2xl z-50">
              <div className="px-2.5 py-1.5 border-b border-lab-border font-mono text-[10px] text-lab-subtle uppercase tracking-wider">
                Time-Travel Snapshots
              </div>
              <div className="max-h-56 overflow-y-auto divide-y divide-lab-border">
                {snapshots.map((snap) => (
                  <button
                    key={snap.snapshot_id}
                    type="button"
                    onClick={() => {
                      setSnapMenuOpen(false);
                      onRestoreSnapshot(snap.snapshot_id);
                    }}
                    className="w-full px-2.5 py-2 text-left hover:bg-lab-surface flex items-center justify-between gap-2 transition-colors"
                  >
                    <div className="min-w-0">
                      <div className="font-mono text-[11px] text-lab-text truncate">
                        {snap.label}
                      </div>
                      <div className="font-mono text-[10px] text-lab-subtle">
                        {snap.snapshot_id}
                      </div>
                    </div>
                    <span className="px-1.5 py-0.5 border border-lab-border bg-lab-bg font-mono text-[9px] text-lab-cutoff shrink-0">
                      {snap.state}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={onReset}
          disabled={busy || currentState === "IDLE"}
          className="h-7 px-2 border border-lab-border bg-lab-bg hover:border-lab-alert/60 hover:text-lab-alert disabled:opacity-35 disabled:cursor-not-allowed font-mono text-[11px] text-lab-muted flex items-center gap-1 transition-colors"
          title="Reset FSM session to IDLE"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">RESET</span>
        </button>
      </div>
      ) : (
        <div className="hidden lg:flex items-center gap-2">
          <span className="px-2 py-1 border border-lab-border bg-lab-bg font-mono text-[10px] text-lab-signal">
            STATIC VIEW · ZERO BACKEND
          </span>
        </div>
      )}
    </header>
  );
}
