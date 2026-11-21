"use client";

import { useRef } from "react";
import { Play, ShieldCheck, Sliders, Upload, Zap } from "lucide-react";
import type {
  AdaptiveMode,
  AttackConfig,
  DefenseConfig,
  FFTFilterType,
  FSMWireState,
  ModelOption,
  NormKind,
  PresetOption,
} from "@/types/workbench";

interface ControlRailProps {
  fsm: FSMWireState | null;
  models: ModelOption[];
  presets: PresetOption[];
  selectedModel: string;
  pretrained: boolean;
  selectedPreset: string;
  attackConfig: AttackConfig;
  defenseConfig: DefenseConfig;
  liveAttackStep: { step: number; total_steps: number; loss: number } | null;
  busyAction: string | null;
  onSelectModel: (id: string) => void;
  onTogglePretrained: (v: boolean) => void;
  onSelectPreset: (id: string) => void;
  onAttackConfigChange: (next: AttackConfig) => void;
  onDefenseConfigChange: (next: DefenseConfig) => void;
  onLoadModel: () => void;
  onLoadPreset: (presetId?: string) => void;
  onUploadFile: (file: File) => void;
  onRunAttack: () => void;
  onRunPurify: () => void;
  onQuickDemo: () => void;
}

export function ControlRail({
  fsm,
  models,
  presets,
  selectedModel,
  pretrained,
  selectedPreset,
  attackConfig,
  defenseConfig,
  liveAttackStep,
  busyAction,
  onSelectModel,
  onTogglePretrained,
  onSelectPreset,
  onAttackConfigChange,
  onDefenseConfigChange,
  onLoadModel,
  onLoadPreset,
  onUploadFile,
  onRunAttack,
  onRunPurify,
  onQuickDemo,
}: ControlRailProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const busy = busyAction !== null;

  const allowedSet = new Set(
    (fsm?.allowed_events ?? []).map((ev) => ev.event),
  );
  const canLoadInput = allowedSet.has("LOAD_INPUT");
  const canAttack = allowedSet.has("START_ATTACK");
  const canPurify = allowedSet.has("START_PURIFY");

  const eps255 = Math.round(attackConfig.epsilon * 255);
  const step255 = Math.max(1, Math.round(attackConfig.step_size * 255));

  const svdParams = defenseConfig.svd ?? {
    rank_k: 28,
    energy_threshold: null,
    channel_mode: "per_channel" as const,
  };
  const fftParams = defenseConfig.fft ?? {
    filter_type: "gaussian" as FFTFilterType,
    cutoff_radius: 0.35,
    band_width: 0.1,
    butterworth_order: 2,
    preserve_phase: true,
  };

  const setDefenseMode = (mode: "svd" | "fft" | "fft_svd" | "svd_fft") => {
    if (mode === "svd") {
      onDefenseConfigChange({
        kind: "svd",
        svd: svdParams,
        fft: null,
        pipeline_order: ["svd"],
      });
    } else if (mode === "fft") {
      onDefenseConfigChange({
        kind: "fft",
        svd: null,
        fft: fftParams,
        pipeline_order: ["fft"],
      });
    } else if (mode === "fft_svd") {
      onDefenseConfigChange({
        kind: "pipeline",
        svd: svdParams,
        fft: fftParams,
        pipeline_order: ["fft", "svd"],
      });
    } else {
      onDefenseConfigChange({
        kind: "pipeline",
        svd: svdParams,
        fft: fftParams,
        pipeline_order: ["svd", "fft"],
      });
    }
  };

  const activeDefenseTab =
    defenseConfig.kind === "svd"
      ? "svd"
      : defenseConfig.kind === "fft"
        ? "fft"
        : defenseConfig.pipeline_order[0] === "fft"
          ? "fft_svd"
          : "svd_fft";

  return (
    <aside className="w-[292px] shrink-0 border-r border-lab-border bg-lab-surface flex flex-col overflow-y-auto lab-scroll select-none">
      {/* Quick Run Banner */}
      <div className="p-3 border-b border-lab-border bg-lab-bg/60">
        <button
          type="button"
          onClick={onQuickDemo}
          disabled={busy}
          className="w-full h-8 px-3 border border-lab-borderStrong bg-lab-elevated hover:bg-lab-text hover:text-lab-bg disabled:opacity-40 font-mono text-[11px] font-medium tracking-wide flex items-center justify-center gap-2 transition-colors"
        >
          <Zap className="w-3.5 h-3.5" />
          <span>{busyAction === "demo" ? "RUNNING PIPELINE..." : "ONE-CLICK FULL BENCHMARK"}</span>
        </button>
      </div>

      {/* Rack 01: Backbone & Target */}
      <section className="p-3 border-b border-lab-border space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.08em] text-lab-muted flex items-center gap-1.5">
            <Sliders className="w-3 h-3 text-lab-subtle" />
            01 · Backbone & Target
          </span>
          <span className="font-mono text-[10px] text-lab-subtle">
            {fsm?.context.model_id ?? "UNLOADED"}
          </span>
        </div>

        <div className="space-y-1.5">
          <div className="flex gap-1.5">
            <select
              value={selectedModel}
              onChange={(e) => onSelectModel(e.target.value)}
              disabled={busy}
              className="flex-1 h-7 px-2 border border-lab-border bg-lab-bg font-mono text-[11px] text-lab-text focus:border-lab-cutoff outline-none"
            >
              {(models.length > 0
                ? models
                : [{ model_id: "resnet18", display_name: "ResNet-18", architecture: "CNN", input_size: 224 }]
              ).map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.display_name} ({m.architecture})
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={onLoadModel}
              disabled={busy}
              className="h-7 px-2.5 border border-lab-borderStrong bg-lab-elevated hover:border-lab-cutoff hover:text-white disabled:opacity-40 font-mono text-[10px] tracking-wide transition-colors"
            >
              {busyAction === "model" ? "..." : "LOAD"}
            </button>
          </div>

          <label className="flex items-center justify-between text-[11px] text-lab-muted cursor-pointer">
            <span className="font-mono text-[10px]">ImageNet Pretrained Weights</span>
            <input
              type="checkbox"
              checked={pretrained}
              onChange={(e) => onTogglePretrained(e.target.checked)}
              className="accent-lab-cutoff"
            />
          </label>
        </div>

        <div className="pt-1 border-t border-lab-border/70 space-y-1.5">
          <div className="flex gap-1.5">
            <select
              value={selectedPreset}
              onChange={(e) => {
                onSelectPreset(e.target.value);
                if (canLoadInput) onLoadPreset(e.target.value);
              }}
              disabled={busy || !canLoadInput}
              className="flex-1 h-7 px-2 border border-lab-border bg-lab-bg font-mono text-[11px] text-lab-text disabled:opacity-40 focus:border-lab-cutoff outline-none"
            >
              {(presets.length > 0
                ? presets
                : [{ preset_id: "beacon_target", label: "Faro Beacon Scene", description: "" }]
              ).map((p) => (
                <option key={p.preset_id} value={p.preset_id}>
                  {p.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => onLoadPreset()}
              disabled={busy || !canLoadInput}
              className="h-7 px-2 border border-lab-border bg-lab-bg hover:border-lab-borderStrong disabled:opacity-35 font-mono text-[10px] text-lab-text transition-colors"
            >
              SET
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onUploadFile(file);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={busy || !canLoadInput}
            className="w-full h-7 px-2 border border-dashed border-lab-borderStrong bg-lab-bg hover:border-lab-cutoff hover:text-lab-cutoff disabled:opacity-35 font-mono text-[10px] text-lab-muted flex items-center justify-center gap-1.5 transition-colors"
          >
            <Upload className="w-3 h-3" />
            <span>UPLOAD CUSTOM IMAGE (224×224 CROP)</span>
          </button>
        </div>
      </section>

      {/* Rack 02: Adversarial Synthesis */}
      <section className="p-3 border-b border-lab-border space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.08em] text-lab-muted flex items-center gap-1.5">
            <Play className="w-3 h-3 text-lab-alert" />
            02 · Attack Synthesis
          </span>
          <span className="font-mono text-[10px] text-lab-alert/90 uppercase">
            {attackConfig.kind === "patch_eot" ? "PATCH+EOT" : `${attackConfig.kind} · ${attackConfig.norm}`}
          </span>
        </div>

        {/* Method segmented switch: PGD | FGSM | PATCH */}
        <div className="grid grid-cols-3 border border-lab-border bg-lab-bg">
          {(
            [
              ["pgd", "PGD"],
              ["fgsm", "FGSM"],
              ["patch_eot", "PATCH"],
            ] as const
          ).map(([kind, label]) => (
            <button
              key={kind}
              type="button"
              onClick={() =>
                onAttackConfigChange({
                  ...attackConfig,
                  kind,
                  steps: kind === "fgsm" ? 1 : kind === "patch_eot" ? 16 : Math.max(4, attackConfig.steps),
                  patch_size: kind === "patch_eot" ? (attackConfig.patch_size ?? 48) : attackConfig.patch_size,
                  patch_x: kind === "patch_eot" ? (attackConfig.patch_x ?? 88) : attackConfig.patch_x,
                  patch_y: kind === "patch_eot" ? (attackConfig.patch_y ?? 76) : attackConfig.patch_y,
                  eot_samples: kind === "patch_eot" ? (attackConfig.eot_samples ?? 4) : attackConfig.eot_samples,
                })
              }
              className={`h-6 font-mono text-[10px] uppercase transition-colors ${
                attackConfig.kind === kind
                  ? "bg-lab-elevated text-white font-medium"
                  : "text-lab-subtle hover:text-lab-muted"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* White-box threat model switch: Oblivious | Adaptive (Exact) | BPDA */}
        <div className="space-y-1">
          <div className="flex justify-between font-mono text-[9px] text-lab-subtle">
            <span>THREAT MODEL</span>
            <span className={attackConfig.adaptive_mode && attackConfig.adaptive_mode !== "none" ? "text-lab-warn" : ""}>
              {attackConfig.adaptive_mode === "exact"
                ? "END-TO-END f(g(x))"
                : attackConfig.adaptive_mode === "bpda"
                  ? "STE BPDA"
                  : "OBLIVIOUS f(x)"}
            </span>
          </div>
          <div className="grid grid-cols-3 border border-lab-border bg-lab-bg">
            {(
              [
                ["none", "OBLIVIOUS"],
                ["exact", "ADAPTIVE"],
                ["bpda", "BPDA"],
              ] as [AdaptiveMode, string][]
            ).map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                onClick={() =>
                  onAttackConfigChange({
                    ...attackConfig,
                    adaptive_mode: mode,
                  })
                }
                className={`h-5 font-mono text-[9px] uppercase transition-colors ${
                  (attackConfig.adaptive_mode ?? "none") === mode
                    ? mode === "none"
                      ? "bg-lab-elevated text-lab-text"
                      : "bg-lab-warn/20 text-lab-warn"
                    : "text-lab-subtle hover:text-lab-muted"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {attackConfig.kind === "patch_eot" ? (
          /* Adversarial Patch + EoT Parameters */
          <div className="space-y-1.5">
            <div className="grid grid-cols-2 gap-2.5">
              <div className="space-y-1">
                <div className="flex justify-between font-mono text-[10px] tabular-nums">
                  <span className="text-lab-muted">PATCH</span>
                  <span className="text-lab-text">{attackConfig.patch_size ?? 48}px</span>
                </div>
                <input
                  type="range"
                  min={24}
                  max={96}
                  step={4}
                  value={attackConfig.patch_size ?? 48}
                  onChange={(e) =>
                    onAttackConfigChange({
                      ...attackConfig,
                      patch_size: Number(e.target.value),
                    })
                  }
                  className="lab-slider"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between font-mono text-[10px] tabular-nums">
                  <span className="text-lab-muted">EoT N</span>
                  <span className="text-lab-text">{attackConfig.eot_samples ?? 4}×</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={8}
                  step={1}
                  value={attackConfig.eot_samples ?? 4}
                  onChange={(e) =>
                    onAttackConfigChange({
                      ...attackConfig,
                      eot_samples: Number(e.target.value),
                    })
                  }
                  className="lab-slider"
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div className="space-y-1">
                <div className="flex justify-between font-mono text-[9px] tabular-nums">
                  <span className="text-lab-muted">X</span>
                  <span className="text-lab-text">{attackConfig.patch_x ?? 88}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={224 - (attackConfig.patch_size ?? 48)}
                  step={4}
                  value={attackConfig.patch_x ?? 88}
                  onChange={(e) =>
                    onAttackConfigChange({
                      ...attackConfig,
                      patch_x: Number(e.target.value),
                    })
                  }
                  className="lab-slider"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between font-mono text-[9px] tabular-nums">
                  <span className="text-lab-muted">Y</span>
                  <span className="text-lab-text">{attackConfig.patch_y ?? 76}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={224 - (attackConfig.patch_size ?? 48)}
                  step={4}
                  value={attackConfig.patch_y ?? 76}
                  onChange={(e) =>
                    onAttackConfigChange({
                      ...attackConfig,
                      patch_y: Number(e.target.value),
                    })
                  }
                  className="lab-slider"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between font-mono text-[9px] tabular-nums">
                  <span className="text-lab-muted">ITER</span>
                  <span className="text-lab-text">{attackConfig.steps}</span>
                </div>
                <input
                  type="range"
                  min={4}
                  max={40}
                  step={2}
                  value={attackConfig.steps}
                  onChange={(e) =>
                    onAttackConfigChange({
                      ...attackConfig,
                      steps: Number(e.target.value),
                    })
                  }
                  className="lab-slider"
                />
              </div>
            </div>
          </div>
        ) : (
          /* Perturbation (PGD / FGSM) Parameters */
          <>
            <div className="grid grid-cols-2 border border-lab-border bg-lab-bg">
              {(["linf", "l2"] as NormKind[]).map((norm) => (
                <button
                  key={norm}
                  type="button"
                  onClick={() =>
                    onAttackConfigChange({
                      ...attackConfig,
                      norm,
                      epsilon: norm === "linf" ? 12 / 255 : 0.6,
                      step_size: norm === "linf" ? 3 / 255 : 0.15,
                    })
                  }
                  className={`h-5 font-mono text-[9px] uppercase transition-colors ${
                    attackConfig.norm === norm
                      ? "bg-lab-elevated text-white"
                      : "text-lab-subtle hover:text-lab-muted"
                  }`}
                >
                  {norm === "linf" ? "L∞ NORM" : "L2 NORM"}
                </button>
              ))}
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-mono text-[10px] tabular-nums">
                <span className="text-lab-muted">BUDGET ε</span>
                <span className="text-lab-text">
                  {attackConfig.norm === "linf"
                    ? `${eps255}/255 (${attackConfig.epsilon.toFixed(4)})`
                    : attackConfig.epsilon.toFixed(2)}
                </span>
              </div>
              {attackConfig.norm === "linf" ? (
                <input
                  type="range"
                  min={1}
                  max={32}
                  step={1}
                  value={eps255}
                  onChange={(e) =>
                    onAttackConfigChange({
                      ...attackConfig,
                      epsilon: Number(e.target.value) / 255,
                    })
                  }
                  className="lab-slider"
                />
              ) : (
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={attackConfig.epsilon}
                  onChange={(e) =>
                    onAttackConfigChange({
                      ...attackConfig,
                      epsilon: Number(e.target.value),
                    })
                  }
                  className="lab-slider"
                />
              )}
            </div>

            {attackConfig.kind === "pgd" && (
              <div className="grid grid-cols-2 gap-2.5">
                <div className="space-y-1">
                  <div className="flex justify-between font-mono text-[10px] tabular-nums">
                    <span className="text-lab-muted">STEPS</span>
                    <span className="text-lab-text">{attackConfig.steps}</span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={30}
                    step={1}
                    value={attackConfig.steps}
                    onChange={(e) =>
                      onAttackConfigChange({
                        ...attackConfig,
                        steps: Number(e.target.value),
                      })
                    }
                    className="lab-slider"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between font-mono text-[10px] tabular-nums">
                    <span className="text-lab-muted">STEP α</span>
                    <span className="text-lab-text">
                      {attackConfig.norm === "linf"
                        ? `${step255}/255`
                        : attackConfig.step_size.toFixed(2)}
                    </span>
                  </div>
                  {attackConfig.norm === "linf" ? (
                    <input
                      type="range"
                      min={1}
                      max={8}
                      step={1}
                      value={step255}
                      onChange={(e) =>
                        onAttackConfigChange({
                          ...attackConfig,
                          step_size: Number(e.target.value) / 255,
                        })
                      }
                      className="lab-slider"
                    />
                  ) : (
                    <input
                      type="range"
                      min={0.02}
                      max={0.3}
                      step={0.02}
                      value={attackConfig.step_size}
                      onChange={(e) =>
                        onAttackConfigChange({
                          ...attackConfig,
                          step_size: Number(e.target.value),
                        })
                      }
                      className="lab-slider"
                    />
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {liveAttackStep && (
          <div className="px-2 py-1 border border-lab-warn/40 bg-lab-warn/10 font-mono text-[10px] text-lab-warn flex justify-between tabular-nums">
            <span>
              ITER {liveAttackStep.step}/{liveAttackStep.total_steps}
            </span>
            <span>LOSS {liveAttackStep.loss.toFixed(4)}</span>
          </div>
        )}

        <button
          type="button"
          onClick={onRunAttack}
          disabled={busy || !canAttack}
          className="w-full h-7 border border-lab-alert/50 bg-lab-alert/10 hover:bg-lab-alert/20 text-lab-alert disabled:opacity-30 disabled:cursor-not-allowed font-mono text-[11px] font-medium tracking-wide transition-colors"
        >
          {busyAction === "attack" ? "SYNTHESIZING..." : "SYNTHESIZE ADVERSARIAL"}
        </button>
      </section>

      {/* Rack 03: Matrix & Signal Purification */}
      <section className="p-3 space-y-2.5 flex-1">
        <div className="flex items-center justify-between">
          <span className="font-sans text-[11px] font-semibold uppercase tracking-[0.08em] text-lab-muted flex items-center gap-1.5">
            <ShieldCheck className="w-3 h-3 text-lab-cutoff" />
            03 · Signal Purification
          </span>
          <span className="font-mono text-[10px] text-lab-cutoff uppercase">
            {activeDefenseTab.replace("_", "→")}
          </span>
        </div>

        {/* Defense Mode selector */}
        <div className="grid grid-cols-4 border border-lab-border bg-lab-bg">
          {(
            [
              ["svd", "SVD"],
              ["fft", "FFT"],
              ["fft_svd", "F→S"],
              ["svd_fft", "S→F"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setDefenseMode(id)}
              className={`h-6 font-mono text-[10px] transition-colors ${
                activeDefenseTab === id
                  ? "bg-lab-elevated text-lab-cutoff font-medium"
                  : "text-lab-subtle hover:text-lab-muted"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* SVD Parameter Block */}
        {(defenseConfig.kind === "svd" || defenseConfig.kind === "pipeline") && (
          <div className="p-2 border border-lab-border bg-lab-bg/70 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] text-lab-subtle">SVD TRUNCATION</span>
              <div className="flex border border-lab-border">
                {(["per_channel", "luminance"] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() =>
                      onDefenseConfigChange({
                        ...defenseConfig,
                        svd: { ...svdParams, channel_mode: mode },
                      })
                    }
                    className={`px-1.5 py-0.5 font-mono text-[9px] uppercase ${
                      svdParams.channel_mode === mode
                        ? "bg-lab-elevated text-lab-text"
                        : "text-lab-subtle"
                    }`}
                  >
                    {mode === "per_channel" ? "RGB" : "LUMA Y"}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-mono text-[10px] tabular-nums">
                <span className="text-lab-muted">RANK CUTOFF k</span>
                <span className="text-lab-cutoff">{svdParams.rank_k ?? 28} / 224</span>
              </div>
              <input
                type="range"
                min={2}
                max={160}
                step={1}
                value={svdParams.rank_k ?? 28}
                onChange={(e) =>
                  onDefenseConfigChange({
                    ...defenseConfig,
                    svd: {
                      ...svdParams,
                      rank_k: Number(e.target.value),
                      energy_threshold: null,
                    },
                  })
                }
                className="lab-slider"
              />
            </div>
          </div>
        )}

        {/* 2D-FFT Parameter Block */}
        {(defenseConfig.kind === "fft" || defenseConfig.kind === "pipeline") && (
          <div className="p-2 border border-lab-border bg-lab-bg/70 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-[10px] text-lab-subtle">2D-FFT KERNEL</span>
              <select
                value={fftParams.filter_type}
                onChange={(e) =>
                  onDefenseConfigChange({
                    ...defenseConfig,
                    fft: { ...fftParams, filter_type: e.target.value as FFTFilterType },
                  })
                }
                className="h-5 px-1.5 border border-lab-border bg-lab-surface font-mono text-[10px] text-lab-text outline-none"
              >
                <option value="gaussian">Gaussian</option>
                <option value="butterworth">Butterworth</option>
                <option value="ideal">Ideal (Gibbs)</option>
                <option value="bandstop">Band-stop</option>
              </select>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-mono text-[10px] tabular-nums">
                <span className="text-lab-muted">CUTOFF RADIUS r</span>
                <span className="text-lab-cutoff">{fftParams.cutoff_radius.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.08}
                max={0.95}
                step={0.01}
                value={fftParams.cutoff_radius}
                onChange={(e) =>
                  onDefenseConfigChange({
                    ...defenseConfig,
                    fft: { ...fftParams, cutoff_radius: Number(e.target.value) },
                  })
                }
                className="lab-slider"
              />
            </div>

            <label className="flex items-center justify-between font-mono text-[10px] text-lab-muted cursor-pointer">
              <span>Preserve Spectral Phase ∠F</span>
              <input
                type="checkbox"
                checked={fftParams.preserve_phase}
                onChange={(e) =>
                  onDefenseConfigChange({
                    ...defenseConfig,
                    fft: { ...fftParams, preserve_phase: e.target.checked },
                  })
                }
                className="accent-lab-cutoff"
              />
            </label>
          </div>
        )}

        <button
          type="button"
          onClick={onRunPurify}
          disabled={busy || !canPurify}
          className="w-full h-7 border border-lab-cutoff/60 bg-lab-cutoff/10 hover:bg-lab-cutoff/20 text-lab-cutoff disabled:opacity-30 disabled:cursor-not-allowed font-mono text-[11px] font-medium tracking-wide transition-colors"
        >
          {busyAction === "purify" ? "PURIFYING..." : "APPLY PURIFICATION"}
        </button>
      </section>
    </aside>
  );
}
