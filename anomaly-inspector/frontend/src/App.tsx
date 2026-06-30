import { useState, useCallback, useEffect } from "react";
import {
  api,
  type InferResponse,
  type StatsResponse,
  type LoadModelResponse,
  type DatasetImage,
} from "./api/client";
import { SettingsModal } from "./components/SettingsModal";
import { ReportModal } from "./components/ReportModal";
import { InferencePanel } from "./components/InferencePanel";
import { ImageGallery } from "./components/ImageGallery";
import { ResultsDisplay } from "./components/ResultsDisplay";
import { VerdictBadge } from "./components/VerdictBadge";
import { StatsPanel } from "./components/StatsPanel";
import { ResizeHandle } from "./components/ResizeHandle";

const UPLOAD_KEY = "__upload__";

const DEFAULT_SETUP_WIDTH = 288;
const DEFAULT_RESULTS_WIDTH = 480;
const MIN_GALLERY_WIDTH = 200;

const clamp = (v: number, min: number, max: number) =>
  Math.min(Math.max(v, min), Math.max(min, max));

export default function App() {
  const [modelStatus, setModelStatus] = useState<LoadModelResponse | null>(null);

  const [skus, setSkus] = useState<string[]>([]);
  const [selectedSku, setSelectedSku] = useState<string | null>(null);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [imagesLoading, setImagesLoading] = useState(false);

  const [customerName, setCustomerName] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [originalSrc, setOriginalSrc] = useState<string | null>(null);
  const [inferringKey, setInferringKey] = useState<string | null>(null);
  const [inferResult, setInferResult] = useState<InferResponse | null>(null);
  const [verdicts, setVerdicts] = useState<Record<string, "ok" | "not_ok">>({});
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [inferError, setInferError] = useState<string | null>(null);

  // Resizable layout: Setup and Results have explicit widths; Gallery flexes.
  const [setupWidth, setSetupWidth] = useState(DEFAULT_SETUP_WIDTH);
  const [resultsWidth, setResultsWidth] = useState(DEFAULT_RESULTS_WIDTH);

  const modelLoaded = !!modelStatus;

  // Setup panel grows from the left edge, so its width == the pointer X.
  const handleSetupDrag = useCallback(
    (clientX: number) =>
      setSetupWidth(clamp(clientX, 220, window.innerWidth - resultsWidth - MIN_GALLERY_WIDTH)),
    [resultsWidth]
  );
  // Results panel hugs the right edge, so its width == viewport width − pointer X.
  const handleResultsDrag = useCallback(
    (clientX: number) =>
      setResultsWidth(
        clamp(window.innerWidth - clientX, 320, window.innerWidth - setupWidth - MIN_GALLERY_WIDTH)
      ),
    [setupWidth]
  );

  // On load, reflect whatever model the backend already has (it auto-loads the
  // bundled default model on startup) and fetch the browsable SKU list.
  useEffect(() => {
    api
      .getStatus()
      .then((status) => {
        if (status.model_loaded) {
          setModelStatus({
            status: "loaded",
            runtime: status.runtime ?? "",
            input_shape: status.input_shape ?? [],
            model_version: status.model_version ?? "",
          });
        }
      })
      .catch(() => {});
    api.getSkus().then(setSkus).catch(() => {});
    api.getStats().then(setStats).catch(() => {});
  }, []);

  // When the SKU changes, load its images and clear any prior selection/results.
  useEffect(() => {
    if (!selectedSku) {
      setImages([]);
      return;
    }
    setImagesLoading(true);
    setImages([]);
    setSelectedPath(null);
    setOriginalSrc(null);
    setInferResult(null);
    setInferError(null);
    setVerdicts({});
    api
      .getSkuImages(selectedSku)
      .then(setImages)
      .catch(() => setInferError("Failed to load images for this SKU."))
      .finally(() => setImagesLoading(false));
  }, [selectedSku]);

  const handleModelLoaded = useCallback((result: LoadModelResponse) => {
    setModelStatus(result);
  }, []);

  async function runInferByPath(path: string) {
    if (!modelLoaded || !selectedSku || inferringKey) return;
    setSelectedPath(path);
    setOriginalSrc(api.imageUrl(path));
    setInferError(null);
    setInferringKey(path);
    try {
      const result = await api.inferByPath(path, selectedSku, threshold, customerName);
      setInferResult(result);
      setVerdicts((v) => ({ ...v, [path]: result.verdict }));
      setStats(await api.getStats());
    } catch (e) {
      setInferError(e instanceof Error ? e.message : "Inference failed.");
    } finally {
      setInferringKey(null);
    }
  }

  async function runInferUpload(file: File) {
    if (!modelLoaded || !selectedSku || inferringKey) return;
    setSelectedPath(null);
    setOriginalSrc(URL.createObjectURL(file));
    setInferError(null);
    setInferringKey(UPLOAD_KEY);
    try {
      const result = await api.infer(file, selectedSku, threshold, customerName);
      setInferResult(result);
      setStats(await api.getStats());
    } catch (e) {
      setInferError(e instanceof Error ? e.message : "Inference failed.");
    } finally {
      setInferringKey(null);
    }
  }

  return (
    <div className="min-h-screen h-screen bg-gray-50 flex flex-col">
      {/* Navbar */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shadow-sm shrink-0">
        <div className="flex items-center gap-3">
          <img
            src="/logo.svg"
            alt="Vuha"
            height={28}
            className="h-7 w-auto"
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
          />
          <div className="flex flex-col leading-tight">
            <span className="text-base font-semibold text-gray-900 tracking-[.3px]">
              Vuha
            </span>
            <span className="text-[11px] text-gray-400 leading-none">
              Anomaly Inspector
            </span>
          </div>
          {customerName && (
            <span className="text-sm text-gray-500 pl-2 border-l border-gray-200">
              Customer: {customerName}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ReportModal customerName={customerName} />
          <SettingsModal
            onModelLoaded={handleModelLoaded}
            isLoaded={modelLoaded}
            modelVersion={modelStatus?.model_version}
            runtime={modelStatus?.runtime}
          />
        </div>
      </header>

      {/* Main layout — Setup | Gallery | Results */}
      <main className="flex flex-1 overflow-hidden">
        {/* Left panel — Setup */}
        <aside
          style={{ width: setupWidth }}
          className="shrink-0 bg-white flex flex-col overflow-y-auto"
        >
          <div className="px-4 pt-4 pb-2 border-b border-gray-100">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Setup
            </h2>
          </div>
          <InferencePanel
            skus={skus}
            selectedSku={selectedSku}
            threshold={threshold}
            customerName={customerName}
            modelLoaded={modelLoaded}
            onSkuChange={setSelectedSku}
            onThresholdChange={setThreshold}
            onCustomerChange={setCustomerName}
            onUpload={runInferUpload}
          />
          <div className="mt-auto border-t border-gray-200 p-4">
            <StatsPanel stats={stats} />
          </div>
        </aside>

        <ResizeHandle
          onDrag={handleSetupDrag}
          onReset={() => setSetupWidth(DEFAULT_SETUP_WIDTH)}
        />

        {/* Center panel — Gallery */}
        <section className="flex-1 flex flex-col overflow-hidden min-w-0">
          <div className="px-4 pt-4 pb-2 flex items-center justify-between">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              {selectedSku ? `Images — ${selectedSku}` : "Images"}
            </h2>
            {images.length > 0 && (
              <span className="text-[11px] text-gray-400">{images.length} images</span>
            )}
          </div>
          <div className="flex-1 overflow-y-auto px-4 pb-4">
            <ImageGallery
              images={images}
              loading={imagesLoading}
              selectedPath={selectedPath}
              inferringPath={inferringKey}
              verdicts={verdicts}
              disabled={!modelLoaded || !!inferringKey}
              onSelect={runInferByPath}
            />
          </div>
        </section>

        <ResizeHandle
          onDrag={handleResultsDrag}
          onReset={() => setResultsWidth(DEFAULT_RESULTS_WIDTH)}
        />

        {/* Right panel — Results */}
        <section
          style={{ width: resultsWidth }}
          className="shrink-0 flex flex-col overflow-y-auto bg-white p-4 gap-4"
        >
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Result
          </h2>

          <VerdictBadge
            verdict={inferResult?.verdict ?? null}
            anomalyScore={inferResult?.anomaly_score ?? null}
          />

          {inferError && (
            <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {inferError}
            </p>
          )}

          <ResultsDisplay
            originalPreview={originalSrc}
            heatmap={inferResult?.heatmap_image ?? null}
            segmentation={inferResult?.segmentation_image ?? null}
          />
        </section>
      </main>
    </div>
  );
}
