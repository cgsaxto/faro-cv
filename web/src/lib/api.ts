import type {
  AttackConfig,
  DefenseConfig,
  WorkbenchPayload,
} from "@/types/workbench";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_BASE;
}

export function getWorkbenchWsUrl(): string {
  const base = getApiBaseUrl();
  const wsBase = base.replace(/^http/, "ws");
  return `${wsBase}/ws/workbench`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const errBody = (await response.json()) as { detail?: string };
      if (errBody.detail) detail = errBody.detail;
    } catch {
      // Fallback to HTTP status text
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export const workbenchApi = {
  getState: () => requestJson<WorkbenchPayload>("/api/state"),

  loadModel: (modelId: string, pretrained: boolean) =>
    requestJson<WorkbenchPayload>("/api/model", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, pretrained }),
    }),

  loadPreset: (presetId: string) =>
    requestJson<WorkbenchPayload>("/api/input/preset", {
      method: "POST",
      body: JSON.stringify({ preset_id: presetId }),
    }),

  uploadImage: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return requestJson<WorkbenchPayload>("/api/input/upload", {
      method: "POST",
      body: form,
    });
  },

  runAttack: (config: AttackConfig) =>
    requestJson<WorkbenchPayload>("/api/attack", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),

  runPurify: (config: DefenseConfig, autoEvaluate = true) =>
    requestJson<WorkbenchPayload>("/api/purify", {
      method: "POST",
      body: JSON.stringify({ config, auto_evaluate: autoEvaluate }),
    }),

  createSnapshot: (label: string) =>
    requestJson<WorkbenchPayload>("/api/snapshot", {
      method: "POST",
      body: JSON.stringify({ label }),
    }),

  restoreSnapshot: (snapshotId: string) =>
    requestJson<WorkbenchPayload>(`/api/snapshot/${snapshotId}/restore`, {
      method: "POST",
    }),

  resetSession: () =>
    requestJson<WorkbenchPayload>("/api/reset", {
      method: "POST",
    }),
};
