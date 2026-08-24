export interface LoadModelResponse {
  status: string;
  runtime: string;
  input_shape: number[];
  model_version: string;
}

export interface Detection {
  class_id: number;
  label: string;
  confidence: number;
  box: number[];
}

// One user-tunable Decision Rule parameter. The GUI renders controls from this
// schema, so a new rule needs no frontend code.
export interface ParamSpec {
  name: string;
  label: string;
  type: "number" | "enum" | "class" | "class_list" | "bool";
  default: unknown;
  min: number | null;
  max: number | null;
  step: number | null;
  options: string[] | null;
  help: string | null;
}

// Declared by rules that support a per-SKU baseline taught from known-good
// samples. Null for rules with nothing to calibrate.
export interface CalibrationSpec {
  param: string;
  metric: string;
  label: string;
}

export interface DecisionRuleInfo {
  name: string;
  label: string;
  consumes: string[];
  params: ParamSpec[];
  calibration: CalibrationSpec | null;
  // Schema defaults with the model sidecar layered on. Seed controls from these,
  // not from ParamSpec.default, or the sidecar's configuration is discarded.
  defaults: RuleParams;
}

export interface CalibrationState {
  sku_name: string;
  decision_rule: string;
  calibrated: boolean;
  params: RuleParams;
  sample_count?: number;
  spread?: number | null;
  updated_at?: string;
}

export interface CalibrateResult extends CalibrationState {
  skipped: string[];
  measured: number[];
}

export interface DecisionRulesResponse {
  active_feature: string | null;
  // Class Catalog of the loaded model, keyed by class id as a string.
  labels: Record<string, string>;
  default_rule: string | null;
  rules: DecisionRuleInfo[];
}

export type RuleParams = Record<string, unknown>;

export interface InferResponse {
  feature: string;
  // Which Decision Rule produced the Verdict.
  decision_rule: string;
  verdict: "ok" | "not_ok";
  // The rule's primary scalar; null for rules without one. `score_label` names it.
  score: number | null;
  score_label: string;
  metrics: Record<string, unknown>;
  reason: string;
  heatmap_image: string | null;
  segmentation_image: string | null;
  // Detector-backed Features (Presence/Absence, Segmentation).
  annotated_image: string | null;
  overlay_image: string | null;
  detections: Detection[] | null;
}

export interface StatsResponse {
  total: number;
  ok: number;
  not_ok: number;
  pass_rate: number;
}

export interface FeatureInfo {
  name: string;
  label: string;
  threshold_label: string;
}

export interface StatusResponse {
  model_loaded: boolean;
  model_version?: string;
  runtime?: string;
  input_shape?: number[];
  active_feature?: string | null;
  features?: FeatureInfo[];
}

export interface DatasetImage {
  path: string;
  category: string;
  name: string;
  // Thumbnail source override for locally-uploaded images (an object URL).
  // Dataset images omit this and are served from the backend by `path`.
  url?: string;
}

// Rule selection and params travel per-request, exactly like `threshold`, so
// the backend holds no tuning state.
function inferForm(
  skuName: string,
  threshold: number,
  customerName: string,
  decisionRule?: string | null,
  ruleParams?: RuleParams | null
): FormData {
  const form = new FormData();
  form.append("sku_name", skuName);
  form.append("threshold", String(threshold));
  form.append("customer_name", customerName);
  if (decisionRule) form.append("decision_rule", decisionRule);
  if (ruleParams && Object.keys(ruleParams).length > 0) {
    form.append("rule_params", JSON.stringify(ruleParams));
  }
  return form;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async loadModel(file: File, modelVersion: string): Promise<LoadModelResponse> {
    const form = new FormData();
    form.append("model_file", file);
    form.append("model_version", modelVersion);
    const res = await fetch("/api/load-model", { method: "POST", body: form });
    return handleResponse<LoadModelResponse>(res);
  },

  async infer(
    image: File,
    skuName: string,
    threshold: number,
    customerName: string,
    decisionRule?: string | null,
    ruleParams?: RuleParams | null
  ): Promise<InferResponse> {
    const form = inferForm(skuName, threshold, customerName, decisionRule, ruleParams);
    form.append("image", image);
    const res = await fetch("/api/infer", { method: "POST", body: form });
    return handleResponse<InferResponse>(res);
  },

  async inferByPath(
    imagePath: string,
    skuName: string,
    threshold: number,
    customerName: string,
    decisionRule?: string | null,
    ruleParams?: RuleParams | null
  ): Promise<InferResponse> {
    const form = inferForm(skuName, threshold, customerName, decisionRule, ruleParams);
    form.append("image_path", imagePath);
    const res = await fetch("/api/infer", { method: "POST", body: form });
    return handleResponse<InferResponse>(res);
  },

  async getDecisionRules(): Promise<DecisionRulesResponse> {
    const res = await fetch("/api/decision-rules");
    return handleResponse<DecisionRulesResponse>(res);
  },

  async getCalibration(sku: string, decisionRule: string): Promise<CalibrationState> {
    const res = await fetch(
      `/api/skus/${encodeURIComponent(sku)}/calibration?decision_rule=${encodeURIComponent(decisionRule)}`
    );
    return handleResponse<CalibrationState>(res);
  },

  async calibrate(
    skuName: string,
    decisionRule: string,
    threshold: number,
    ruleParams: RuleParams,
    // Dataset samples by path, and browser-uploaded samples as files. A folder
    // browsed in the GUI lives only in the browser, so its files must be sent.
    imagePaths: string[],
    imageFiles: File[]
  ): Promise<CalibrateResult> {
    const form = new FormData();
    form.append("sku_name", skuName);
    form.append("decision_rule", decisionRule);
    form.append("image_paths", JSON.stringify(imagePaths));
    form.append("threshold", String(threshold));
    form.append("rule_params", JSON.stringify(ruleParams));
    for (const file of imageFiles) form.append("images", file);
    const res = await fetch("/api/calibrate", { method: "POST", body: form });
    return handleResponse<CalibrateResult>(res);
  },

  async clearCalibration(sku: string, decisionRule: string): Promise<{ cleared: boolean }> {
    const res = await fetch(
      `/api/skus/${encodeURIComponent(sku)}/calibration?decision_rule=${encodeURIComponent(decisionRule)}`,
      { method: "DELETE" }
    );
    return handleResponse<{ cleared: boolean }>(res);
  },

  async getSkus(): Promise<string[]> {
    const res = await fetch("/api/skus");
    const data = await handleResponse<{ skus: string[] }>(res);
    return data.skus;
  },

  async getSkuImages(sku: string): Promise<DatasetImage[]> {
    const res = await fetch(`/api/skus/${encodeURIComponent(sku)}/images`);
    const data = await handleResponse<{ images: DatasetImage[] }>(res);
    return data.images;
  },

  imageUrl(path: string): string {
    return `/api/images?path=${encodeURIComponent(path)}`;
  },

  async getStats(
    skuName?: string | null,
    feature?: string | null,
    decisionRule?: string | null
  ): Promise<StatsResponse> {
    const params = new URLSearchParams();
    if (skuName) params.set("sku_name", skuName);
    if (feature) params.set("feature", feature);
    if (decisionRule) params.set("decision_rule", decisionRule);
    const qs = params.toString();
    const res = await fetch(qs ? `/api/stats?${qs}` : "/api/stats");
    return handleResponse<StatsResponse>(res);
  },

  async setFeature(feature: string): Promise<LoadModelResponse & { active_feature: string }> {
    const form = new FormData();
    form.append("feature", feature);
    const res = await fetch("/api/feature", { method: "POST", body: form });
    return handleResponse<LoadModelResponse & { active_feature: string }>(res);
  },

  async resetDatabase(): Promise<{ status: string; deleted: number }> {
    const res = await fetch("/api/reset", { method: "POST" });
    return handleResponse<{ status: string; deleted: number }>(res);
  },

  async getStatus(): Promise<StatusResponse> {
    const res = await fetch("/api/status");
    return handleResponse<StatusResponse>(res);
  },

  async generateReport(startDate: string, endDate: string, customerName: string): Promise<Blob> {
    const form = new FormData();
    form.append("start_date", startDate);
    form.append("end_date", endDate);
    form.append("customer_name", customerName);
    const res = await fetch("/api/report", { method: "POST", body: form });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.blob();
  },
};
