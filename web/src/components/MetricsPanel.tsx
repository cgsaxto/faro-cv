"use client";

import type { EvaluationMetrics, InferenceSummary } from "@/types/workbench";

interface MetricsPanelProps {
  metrics: EvaluationMetrics | null;
  cleanInference: InferenceSummary | null;
  adversarialInference: InferenceSummary | null;
  purifiedInference: InferenceSummary | null;
}

export function MetricsPanel({
  metrics,
  cleanInference,
  adversarialInference,
  purifiedInference,
}: MetricsPanelProps) {
  const rows = [
    {
      tag: "CLN",
      color: "bg-lab-signal",
      textClass: "text-lab-signal",
      entry: cleanInference?.top_k[0] ?? null,
    },
    {
      tag: "ADV",
      color: "bg-lab-alert",
      textClass: "text-lab-alert",
      entry: adversarialInference?.top_k[0] ?? null,
    },
    {
      tag: "PUR",
      color: "bg-lab-cutoff",
      textClass: "text-lab-cutoff",
      entry: purifiedInference?.top_k[0] ?? null,
    },
  ];

  const verdict = !metrics
    ? null
    : metrics.attack_succeeded
      ? metrics.conditional_recovered
        ? {
            label: "COND. RECOVERED",
            cls: "border-lab-signal/40 bg-lab-signal/10 text-lab-signal",
          }
        : {
            label: "UNRECOVERED",
            cls: "border-lab-alert/40 bg-lab-alert/10 text-lab-alert",
          }
      : metrics.prediction_consistent
        ? {
            label: "CONSISTENT (N/A)",
            cls: "border-lab-borderStrong bg-lab-bg text-lab-muted",
          }
        : {
            label: "PURIFY DRIFT",
            cls: "border-lab-warn/40 bg-lab-warn/10 text-lab-warn",
          };

  return (
    <div className="flex flex-col min-w-0 min-h-0 bg-lab-bg">
      <div className="h-7 shrink-0 px-2.5 border-b border-lab-border bg-lab-surface flex items-center justify-between gap-2">
        <span className="font-sans text-[11px] font-semibold tracking-[0.07em] text-lab-text">
          EVALUATION TELEMETRY
        </span>
        {verdict ? (
          <span
            className={`px-1.5 py-0.5 font-mono text-[9px] border ${verdict.cls}`}
            title={
              metrics?.attack_succeeded
                ? "Attack flipped top-1 prediction; badge reflects conditional recovery."
                : "Attack did not flip top-1 prediction; conditional recovery is N/A."
            }
          >
            {verdict.label}
          </span>
        ) : (
          <span className="font-mono text-[9px] text-lab-subtle">PENDING</span>
        )}
      </div>

      <div className="flex-1 min-h-0 p-2.5 flex flex-col justify-between gap-2 overflow-hidden">
        {/* 2x2 Quantitative KPI Grid */}
        <div className="grid grid-cols-2 gap-1.5">
          <div className="p-2 border border-lab-border bg-lab-surface">
            <div className="font-mono text-[9px] text-lab-subtle">PSNR (FIDELITY)</div>
            <div className="font-mono text-sm font-semibold text-lab-text tabular-nums mt-0.5">
              {metrics ? `${metrics.purified_psnr_db.toFixed(2)} dB` : "—"}
            </div>
          </div>

          <div className="p-2 border border-lab-border bg-lab-surface">
            <div className="font-mono text-[9px] text-lab-subtle">SSIM (STRUCT)</div>
            <div className="font-mono text-sm font-semibold text-lab-text tabular-nums mt-0.5">
              {metrics ? metrics.purified_ssim.toFixed(4) : "—"}
            </div>
          </div>

          <div className="p-2 border border-lab-border bg-lab-surface">
            <div className="font-mono text-[9px] text-lab-subtle">PERTURB L∞ / L2</div>
            <div className="font-mono text-[11px] font-medium text-lab-text tabular-nums mt-1">
              {metrics
                ? `${metrics.attack_linf_norm.toFixed(4)} / ${metrics.attack_l2_norm.toFixed(2)}`
                : "—"}
            </div>
          </div>

          <div className="p-2 border border-lab-border bg-lab-surface">
            <div className="font-mono text-[9px] text-lab-subtle">CAM ATTN SIM (ADV→PUR)</div>
            <div className="font-mono text-[11px] font-medium tabular-nums mt-1">
              {metrics &&
              metrics.attention_similarity_adv !== null &&
              metrics.attention_similarity_purified !== null ? (
                <span>
                  <span className="text-lab-alert">
                    {metrics.attention_similarity_adv.toFixed(2)}
                  </span>
                  <span className="text-lab-subtle mx-1">→</span>
                  <span className="text-lab-cutoff">
                    {metrics.attention_similarity_purified.toFixed(2)}
                  </span>
                </span>
              ) : (
                <span className="text-lab-subtle">—</span>
              )}
            </div>
          </div>
        </div>

        {/* Clean vs. Adversarial vs. Purified Top-1 Confidence Bars */}
        <div className="border border-lab-border bg-lab-surface p-2 space-y-1.5">
          <div className="font-mono text-[9px] text-lab-subtle tracking-wider">
            TOP-1 CLASSIFICATION SHIFT
          </div>
          {rows.map((row) => (
            <div key={row.tag} className="space-y-0.5">
              <div className="flex items-center justify-between font-mono text-[10px] gap-2 tabular-nums">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className={`font-semibold ${row.textClass}`}>{row.tag}</span>
                  <span className="text-lab-text truncate">
                    {row.entry ? row.entry.label : "—"}
                  </span>
                </div>
                <span className="text-lab-muted shrink-0">
                  {row.entry ? `${(row.entry.probability * 100).toFixed(1)}%` : "0.0%"}
                </span>
              </div>
              <div className="h-1 w-full bg-lab-bg overflow-hidden">
                <div
                  className={`h-full ${row.color} transition-all duration-200`}
                  style={{ width: `${(row.entry?.probability ?? 0) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
