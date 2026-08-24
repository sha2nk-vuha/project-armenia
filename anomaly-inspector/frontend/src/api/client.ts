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

export interface InferResponse {
  feature: string;
  verdict: "ok" | "not_ok";
  // Anomaly Detection only; null for Features without a single score.
  anomaly_score: number | null;
  heatmap_image: string | null;
  segmentation_image: string | null;
  // Presence/Absence only.
  annotated_image: string | null;
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

  async infer(image: File, skuName: string, threshold: number, customerName: string): Promise<InferResponse> {
    const form = new FormData();
    form.append("image", image);
    form.append("sku_name", skuName);
    form.append("threshold", String(threshold));
    form.append("customer_name", customerName);
    const res = await fetch("/api/infer", { method: "POST", body: form });
    return handleResponse<InferResponse>(res);
  },

  async inferByPath(imagePath: string, skuName: string, threshold: number, customerName: string): Promise<InferResponse> {
    const form = new FormData();
    form.append("image_path", imagePath);
    form.append("sku_name", skuName);
    form.append("threshold", String(threshold));
    form.append("customer_name", customerName);
    const res = await fetch("/api/infer", { method: "POST", body: form });
    return handleResponse<InferResponse>(res);
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

  async getStats(skuName?: string | null, feature?: string | null): Promise<StatsResponse> {
    const params = new URLSearchParams();
    if (skuName) params.set("sku_name", skuName);
    if (feature) params.set("feature", feature);
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
