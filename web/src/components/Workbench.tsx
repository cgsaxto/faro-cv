"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import { ControlRail } from "@/components/ControlRail";
import { DenoisingStudyView } from "@/components/DenoisingStudyView";
import { FftSpectrumPanel } from "@/components/FftSpectrumPanel";
import { MetricsPanel } from "@/components/MetricsPanel";
import { SvdCutoffChart } from "@/components/SvdCutoffChart";
import { SynchronizedViewport } from "@/components/SynchronizedViewport";
import { TopInstrumentBar } from "@/components/TopInstrumentBar";
import { getWorkbenchWsUrl, workbenchApi } from "@/lib/api";
import type {
  AttackConfig,
  DefenseConfig,
  WorkbenchPayload,
  WorkbenchWsMessage,
} from "@/types/workbench";

const DEFAULT_ATTACK: AttackConfig = {
  kind: "pgd",
  norm: "linf",
  epsilon: 12 / 255,
  step_size: 3 / 255,
  steps: 8,
  targeted: false,
  target_class: null,
};

const DEFAULT_DEFENSE: DefenseConfig = {
  kind: "svd",
  svd: {
    rank_k: 28,
    energy_threshold: null,
    channel_mode: "per_channel",
  },
  fft: null,
  pipeline_order: ["svd"],
};

export function Workbench() {
  const [activeView, setActiveView] = useState<"study" | "lab">("study");
  const [payload, setPayload] = useState<WorkbenchPayload | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [liveAttackStep, setLiveAttackStep] = useState<{
    step: number;
    total_steps: number;
    loss: number;
  } | null>(null);

  // Instrument control states
  const [selectedModel, setSelectedModel] = useState("resnet18");
  const [pretrained, setPretrained] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState("beacon_target");
  const [attackConfig, setAttackConfig] = useState<AttackConfig>(DEFAULT_ATTACK);
  const [defenseConfig, setDefenseConfig] = useState<DefenseConfig>(DEFAULT_DEFENSE);

  // Only connect to FastAPI REST/WebSocket when user switches into Interactive Lab mode
  useEffect(() => {
    if (activeView !== "lab") return;
    let active = true;
    setErrorMsg(null);
    workbenchApi
      .getState()
      .then((initial) => {
        if (active) setPayload(initial);
      })
      .catch((err: unknown) => {
        if (active) {
          setErrorMsg(
            `Interactive Lab backend offline (${err instanceof Error ? err.message : "unknown error"}). Start FastAPI with 'make dev-api' or stay in Denoising Study mode.`,
          );
        }
      });

    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(getWorkbenchWsUrl());
      ws.onopen = () => {
        if (active) setWsConnected(true);
      };
      ws.onclose = () => {
        if (active) setWsConnected(false);
      };
      ws.onmessage = (event: MessageEvent<string>) => {
        if (!active) return;
        try {
          const msg = JSON.parse(event.data) as WorkbenchWsMessage;
          if (msg.type === "state_update") {
            setPayload(msg.payload);
            setLiveAttackStep(null);
          } else if (msg.type === "attack_progress") {
            setLiveAttackStep({
              step: msg.step,
              total_steps: msg.total_steps,
              loss: msg.loss,
            });
          }
        } catch {
          // Ignore malformed frames
        }
      };
    } catch {
      setWsConnected(false);
    }

    return () => {
      active = false;
      ws?.close();
    };
  }, [activeView]);

  const runAction = useCallback(async (tag: string, fn: () => Promise<WorkbenchPayload>) => {
    setBusyAction(tag);
    setErrorMsg(null);
    try {
      const next = await fn();
      setPayload(next);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyAction(null);
      setLiveAttackStep(null);
    }
  }, []);

  const handleLoadModel = () =>
    void runAction("model", () => workbenchApi.loadModel(selectedModel, pretrained));

  const handleLoadPreset = (presetOverride?: string) =>
    void runAction("input", () => workbenchApi.loadPreset(presetOverride ?? selectedPreset));

  const handleUploadFile = (file: File) =>
    void runAction("input", () => workbenchApi.uploadImage(file));

  const resolveAttackPayload = useCallback((): AttackConfig => {
    if (attackConfig.adaptive_mode && attackConfig.adaptive_mode !== "none") {
      return {
        ...attackConfig,
        adaptive_defense: defenseConfig,
      };
    }
    return {
      ...attackConfig,
      adaptive_mode: "none",
      adaptive_defense: null,
    };
  }, [attackConfig, defenseConfig]);

  const handleRunAttack = () =>
    void runAction("attack", () => workbenchApi.runAttack(resolveAttackPayload()));

  const handleRunPurify = (overrideConfig?: DefenseConfig) =>
    void runAction("purify", () => workbenchApi.runPurify(overrideConfig ?? defenseConfig, true));

  const handleQuickDemo = () => {
    void runAction("demo", async () => {
      await workbenchApi.loadModel(selectedModel, pretrained);
      await workbenchApi.loadPreset(selectedPreset);
      await workbenchApi.runAttack(resolveAttackPayload());
      return workbenchApi.runPurify(defenseConfig, true);
    });
  };

  const handleCaptureSnapshot = () => {
    const state = payload?.fsm.state ?? "SNAP";
    const rank = defenseConfig.svd?.rank_k;
    const label = rank ? `${state} · SVD k=${rank}` : `${state} @ #${payload?.fsm.step_count ?? 0}`;
    void runAction("snapshot", () => workbenchApi.createSnapshot(label));
  };

  const handleRestoreSnapshot = (snapshotId: string) =>
    void runAction("snapshot", () => workbenchApi.restoreSnapshot(snapshotId));

  const handleReset = () => void runAction("reset", () => workbenchApi.resetSession());

  // Two-way binding for interactive SVD Scree Plot cutoff line dragging
  const handlePreviewRankChange = (rank: number) => {
    setDefenseConfig((prev) => ({
      ...prev,
      kind: prev.kind === "fft" ? "svd" : prev.kind,
      svd: {
        rank_k: rank,
        energy_threshold: null,
        channel_mode: prev.svd?.channel_mode ?? "per_channel",
      },
      pipeline_order: prev.kind === "fft" ? ["svd"] : prev.pipeline_order,
    }));
  };

  const handleCommitRank = (rank: number) => {
    const nextConfig: DefenseConfig = {
      ...defenseConfig,
      kind: defenseConfig.kind === "fft" ? "svd" : defenseConfig.kind,
      svd: {
        rank_k: rank,
        energy_threshold: null,
        channel_mode: defenseConfig.svd?.channel_mode ?? "per_channel",
      },
      pipeline_order: defenseConfig.kind === "fft" ? ["svd"] : defenseConfig.pipeline_order,
    };
    setDefenseConfig(nextConfig);
    const canPurify = (payload?.fsm.allowed_events ?? []).some(
      (ev) => ev.event === "START_PURIFY",
    );
    if (canPurify && !busyAction) {
      handleRunPurify(nextConfig);
    }
  };

  const ctx = payload?.fsm.context ?? null;
  const currentRank =
    defenseConfig.svd?.rank_k ?? ctx?.spectral_summary?.effective_rank ?? 28;
  const canInteractivePurify =
    !busyAction &&
    (payload?.fsm.allowed_events ?? []).some((ev) => ev.event === "START_PURIFY");

  return (
    <div className="min-h-screen lg:h-screen w-full lg:w-screen lg:overflow-hidden flex flex-col bg-lab-bg text-lab-text">
      <TopInstrumentBar
        activeView={activeView}
        onChangeView={setActiveView}
        fsm={payload?.fsm ?? null}
        deviceProfile={payload?.device_profile ?? null}
        snapshots={payload?.snapshots ?? []}
        wsConnected={wsConnected}
        busy={busyAction !== null}
        onCaptureSnapshot={handleCaptureSnapshot}
        onRestoreSnapshot={handleRestoreSnapshot}
        onReset={handleReset}
      />

      {activeView === "study" ? (
        <DenoisingStudyView />
      ) : (
        <>
          {errorMsg && (
            <div className="h-7 shrink-0 px-3 border-b border-lab-alert/40 bg-lab-alert/10 flex items-center justify-between gap-2 font-mono text-[10px] text-lab-alert">
              <div className="flex items-center gap-2 truncate">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">{errorMsg}</span>
              </div>
              <button
                type="button"
                onClick={() => setErrorMsg(null)}
                className="hover:text-white shrink-0"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Main Zero-Scroll Workbench Body */}
          <div className="flex-1 min-h-0 flex overflow-hidden">
        <ControlRail
          fsm={payload?.fsm ?? null}
          models={payload?.available_models ?? []}
          presets={payload?.available_presets ?? []}
          selectedModel={selectedModel}
          pretrained={pretrained}
          selectedPreset={selectedPreset}
          attackConfig={attackConfig}
          defenseConfig={defenseConfig}
          liveAttackStep={liveAttackStep}
          busyAction={busyAction}
          onSelectModel={setSelectedModel}
          onTogglePretrained={setPretrained}
          onSelectPreset={setSelectedPreset}
          onAttackConfigChange={setAttackConfig}
          onDefenseConfigChange={setDefenseConfig}
          onLoadModel={handleLoadModel}
          onLoadPreset={handleLoadPreset}
          onUploadFile={handleUploadFile}
          onRunAttack={handleRunAttack}
          onRunPurify={() => handleRunPurify()}
          onQuickDemo={handleQuickDemo}
        />

        {/* Right Telemetry & Viewport Matrix */}
        <main className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
          {/* Upper Deck: 3-Column Synchronized Pixel Loupe Comparator */}
          <SynchronizedViewport
            images={payload?.images ?? null}
            cleanInference={ctx?.clean_inference ?? null}
            adversarialInference={ctx?.adversarial_inference ?? null}
            purifiedInference={ctx?.purified_inference ?? null}
          />

          {/* Lower Deck: Dual Signal Analyzer + Quantitative Metrics */}
          <section className="flex-[43] min-h-0 grid grid-cols-12 divide-x divide-lab-border overflow-hidden">
            <div className="col-span-6 min-w-0 min-h-0 flex flex-col">
              <SvdCutoffChart
                spectral={ctx?.spectral_summary ?? null}
                currentRank={currentRank}
                interactive={canInteractivePurify}
                onPreviewRankChange={handlePreviewRankChange}
                onCommitRank={handleCommitRank}
              />
            </div>

            <div className="col-span-3 min-w-0 min-h-0 flex flex-col">
              <FftSpectrumPanel
                images={payload?.images ?? null}
                spectral={ctx?.spectral_summary ?? null}
                appliedDefenseConfig={ctx?.defense_config ?? null}
              />
            </div>

            <div className="col-span-3 min-w-0 min-h-0 flex flex-col">
              <MetricsPanel
                metrics={ctx?.evaluation ?? null}
                cleanInference={ctx?.clean_inference ?? null}
                adversarialInference={ctx?.adversarial_inference ?? null}
                purifiedInference={ctx?.purified_inference ?? null}
              />
            </div>
          </section>
        </main>
      </div>
        </>
      )}
    </div>
  );
}
