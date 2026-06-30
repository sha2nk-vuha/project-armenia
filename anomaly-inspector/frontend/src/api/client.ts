export interface LoadModelResponse {
  status: string;
  runtime: string;
  input_shape: number[];
  model_version: string;
}

export interface InferResponse {
  verdict: "ok" | "not_ok";
  anomaly_score: number;
  heatmap_image: string;
  segmentation_image: string;
}

export interface StatsResponse {
  total: number;
  ok: number;
  not_ok: number;
  pass_rate: number;
}

export interface StatusResponse {
  model_loaded: boolean;
  model_version?: string;
  runtime?: string;
  input_shape?: number[];
}

export interface DatasetImage {
  path: string;
  category: string;
  name: string;
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

  async infer(image: File, skuName: string, threshold: number): Promise<InferResponse> {
    const form = new FormData();
    form.append("image", image);
    form.append("sku_name", skuName);
    form.append("threshold", String(threshold));
    const res = await fetch("/api/infer", { method: "POST", body: form });
    return handleResponse<InferResponse>(res);
  },

  async inferByPath(imagePath: string, skuName: string, threshold: number): Promise<InferResponse> {
    const form = new FormData();
    form.append("image_path", imagePath);
    form.append("sku_name", skuName);
    form.append("threshold", String(threshold));
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

  async getStats(): Promise<StatsResponse> {
    const res = await fetch("/api/stats");
    return handleResponse<StatsResponse>(res);
  },

  async getStatus(): Promise<StatusResponse> {
    const res = await fetch("/api/status");
    return handleResponse<StatusResponse>(res);
  },

  async generateReport(startDate: string, endDate: string): Promise<Blob> {
    const form = new FormData();
    form.append("start_date", startDate);
    form.append("end_date", endDate);
    const res = await fetch("/api/report", { method: "POST", body: form });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.blob();
  },
};
