export type ComputeDevice = "mps" | "cuda" | "cpu";
export type AttackKind = "fgsm" | "pgd" | "patch_eot";
export type NormKind = "linf" | "l2";
export type AdaptiveMode = "none" | "exact" | "bpda";
export type DefenseKind = "svd" | "fft" | "pipeline";
export type FFTFilterType = "gaussian" | "butterworth" | "ideal" | "bandstop";

export type LabState =
  | "IDLE"
  | "MODEL_READY"
  | "INPUT_LOADED"
  | "ATTACKING"
  | "PAUSED"
  | "ATTACKED"
  | "PURIFYING"
  | "PURIFIED"
  | "EVALUATED"
  | "ERROR";

export type LabEvent =
  | "LOAD_MODEL"
  | "LOAD_INPUT"
  | "START_ATTACK"
  | "PAUSE_ATTACK"
  | "RESUME_ATTACK"
  | "CANCEL_ATTACK"
  | "COMPLETE_ATTACK"
  | "START_PURIFY"
  | "COMPLETE_PURIFY"
  | "EVALUATE"
  | "FAIL"
  | "RECOVER"
  | "RESET";

export interface OperatorDeviceStatus {
  device: ComputeDevice;
  supported_on_accelerator: boolean;
  probe_latency_ms: number;
  fallback_reason: string | null;
}

export interface OperatorDeviceProfile {
  torch_available: boolean;
  torch_version: string | null;
  preferred_accelerator: ComputeDevice;
  model_op: OperatorDeviceStatus;
  svd_op: OperatorDeviceStatus;
  fft_op: OperatorDeviceStatus;
}

export interface TensorMetadata {
  width: number;
  height: number;
  channels: number;
  dtype: string;
  sha256: string;
  min_value: number;
  max_value: number;
}

export interface PredictionEntry {
  class_index: number;
  label: string;
  probability: number;
}

export interface InferenceSummary {
  top_k: PredictionEntry[];
  entropy: number;
  latency_ms: number;
  device: ComputeDevice;
}

export interface AttackConfig {
  kind: AttackKind;
  norm: NormKind;
  epsilon: number;
  step_size: number;
  steps: number;
  targeted: boolean;
  target_class: number | null;
  patch_size?: number | null;
  patch_x?: number | null;
  patch_y?: number | null;
  eot_samples?: number;
  adaptive_mode?: AdaptiveMode;
  adaptive_defense?: DefenseConfig | null;
}

export interface AttackProgress {
  current_step: number;
  total_steps: number;
  current_loss: number;
  elapsed_ms: number;
}

export interface SVDParams {
  rank_k: number | null;
  energy_threshold: number | null;
  channel_mode: "per_channel" | "luminance";
}

export interface FFTParams {
  filter_type: FFTFilterType;
  cutoff_radius: number;
  band_width: number;
  butterworth_order: number;
  preserve_phase: boolean;
}

export interface DefenseConfig {
  kind: DefenseKind;
  svd: SVDParams | null;
  fft: FFTParams | null;
  pipeline_order: ("svd" | "fft")[];
}

export interface SpectralSummary {
  singular_values: number[];
  cumulative_energy: number[];
  effective_rank: number;
  retained_energy_ratio: number;
  high_freq_energy_before: number;
  high_freq_energy_after: number;
}

export interface EvaluationMetrics {
  attack_l2_norm: number;
  attack_linf_norm: number;
  purified_psnr_db: number;
  purified_ssim: number;
  attack_succeeded: boolean;
  prediction_consistent: boolean;
  conditional_recovered: boolean | null;
  clean_confidence: number;
  adversarial_confidence: number;
  purified_confidence: number;
  purification_latency_ms: number;
  attention_similarity_adv: number | null;
  attention_similarity_purified: number | null;
  ground_truth_class: number | null;
  clean_correct: boolean | null;
  adversarial_correct: boolean | null;
  purified_correct: boolean | null;
}

export interface ExperimentContext {
  session_id: string;
  created_at: string;
  seed: number;
  model_id: string | null;
  device_profile: OperatorDeviceProfile | null;
  input_meta: TensorMetadata | null;
  clean_inference: InferenceSummary | null;
  attack_config: AttackConfig | null;
  attack_progress: AttackProgress | null;
  adversarial_meta: TensorMetadata | null;
  adversarial_inference: InferenceSummary | null;
  defense_config: DefenseConfig | null;
  spectral_summary: SpectralSummary | null;
  purified_meta: TensorMetadata | null;
  purified_inference: InferenceSummary | null;
  evaluation: EvaluationMetrics | null;
  last_error: string | null;
}

export interface EventAvailability {
  event: LabEvent;
  target_state: LabState;
  guard_passed: boolean;
  blocked_reason: string | null;
}

export interface TransitionRecord {
  step_index: number;
  timestamp: string;
  from_state: LabState;
  event: LabEvent;
  to_state: LabState;
  metadata: Record<string, unknown>;
}

export interface FSMWireState {
  state: LabState;
  is_accepting: boolean;
  recovery_state: LabState;
  step_count: number;
  allowed_events: EventAvailability[];
  context: ExperimentContext;
  recent_history: TransitionRecord[];
}

export interface ImageBundle {
  clean_png: string | null;
  adversarial_png: string | null;
  residual_png: string | null;
  purified_png: string | null;
  clean_cam_png: string | null;
  adversarial_cam_png: string | null;
  purified_cam_png: string | null;
  fft_before_png: string | null;
  fft_after_png: string | null;
}

export interface ModelOption {
  model_id: string;
  display_name: string;
  architecture: string;
  input_size: number;
}

export interface PresetOption {
  preset_id: string;
  label: string;
  description: string;
}

export interface SnapshotSummary {
  snapshot_id: string;
  label: string;
  state: LabState;
  created_at: string;
}

export interface WorkbenchPayload {
  fsm: FSMWireState;
  device_profile: OperatorDeviceProfile;
  images: ImageBundle;
  available_models: ModelOption[];
  available_presets: PresetOption[];
  snapshots: SnapshotSummary[];
}

export interface LiveAttackStepMessage {
  type: "attack_progress";
  step: number;
  total_steps: number;
  loss: number;
}

export interface LiveStateMessage {
  type: "state_update";
  payload: WorkbenchPayload;
}

export type WorkbenchWsMessage = LiveAttackStepMessage | LiveStateMessage;

export interface StudyBoxROI {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface StudyImageSet {
  clean: string;
  noisy: string;
  svd: string;
  fft: string;
}

export interface StudyTargetSpec {
  preset_id: string;
  display_name: string;
  feature_label: string;
  clean_feature_contrast_luma: number;
  ref_roi: StudyBoxROI;
  detail_roi: StudyBoxROI;
  feature_fg: StudyBoxROI;
  feature_bg: StudyBoxROI[];
  images: StudyImageSet;
  images_by_seed: Record<string, StudyImageSet>;
}

export interface StudyMeasurementRow {
  seed: number;
  image: string;
  feature_label: string;
  method: string;
  psnr_db: number;
  ssim: number;
  ref_roi_residual_std: number;
  feature_contrast_luma: number;
  feature_contrast_retained_pct: number;
  pipeline_latency_ms: number;
}

export interface StudyDataBundle {
  generated_at: string;
  sigma: number;
  primary_seed: number;
  seeds: number[];
  svd_k: number;
  fft_r: number;
  fft_grid_note: string;
  latency_note: string;
  targets: StudyTargetSpec[];
  rows: StudyMeasurementRow[];
}
