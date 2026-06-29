import { useState, useCallback } from "react";
import { ScanLine } from "lucide-react";
import { api, type InferResponse, type StatsResponse, type LoadModelResponse } from "./api/client";
import { SettingsModal } from "./components/SettingsModal";
import { ReportModal } from "./components/ReportModal";
import { InferencePanel } from "./components/InferencePanel";
import { ResultsDisplay } from "./components/ResultsDisplay";
import { VerdictBadge } from "./components/VerdictBadge";
import { StatsPanel } from "./components/StatsPanel";

export default function App() {
  const [modelStatus, setModelStatus] = useState<LoadModelResponse | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [skuName, setSkuName] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [isInferring, setIsInferring] = useState(false);
  const [inferResult, setInferResult] = useState<InferResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [inferError, setInferError] = useState<string | null>(null);

  const handleImageSelected = useCallback((file: File, preview: string) => {
    setSelectedImage(file);
    setImagePreview(preview);
    setInferResult(null);
    setInferError(null);
  }, []);

  const handleModelLoaded = useCallback((result: LoadModelResponse) => {
    setModelStatus(result);
  }, []);

  async function handleInfer() {
    if (!selectedImage || !skuName.trim() || !modelStatus) return;
    setIsInferring(true);
    setInferError(null);
    try {
      const result = await api.infer(selectedImage, skuName.trim(), threshold);
      setInferResult(result);
      const updatedStats = await api.getStats();
      setStats(updatedStats);
    } catch (e) {
      setInferError(e instanceof Error ? e.message : "Inference failed.");
    } finally {
      setIsInferring(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navbar */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-2">
          <ScanLine className="h-6 w-6 text-blue-600" />
          <h1 className="text-base font-bold text-gray-900 tracking-tight uppercase">
            Anomaly Detection Inspector
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <ReportModal />
          <SettingsModal
            onModelLoaded={handleModelLoaded}
            isLoaded={!!modelStatus}
            modelVersion={modelStatus?.model_version}
            runtime={modelStatus?.runtime}
          />
        </div>
      </header>

      {/* Main layout */}
      <main className="flex flex-1 overflow-hidden">
        {/* Left panel — Setup */}
        <aside className="w-72 min-w-72 border-r border-gray-200 bg-white flex flex-col">
          <div className="px-4 pt-4 pb-2 border-b border-gray-100">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Setup
            </h2>
          </div>
          <InferencePanel
            skuName={skuName}
            modelLoaded={!!modelStatus}
            selectedImage={selectedImage}
            imagePreview={imagePreview}
            threshold={threshold}
            isInferring={isInferring}
            onSkuChange={setSkuName}
            onImageSelected={handleImageSelected}
            onThresholdChange={setThreshold}
            onInfer={handleInfer}
          />
        </aside>

        {/* Right panel — Results */}
        <section className="flex-1 flex flex-col overflow-auto p-6 gap-6">
          <div>
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Results
            </h2>
            <ResultsDisplay
              originalPreview={imagePreview}
              heatmap={inferResult?.heatmap_image ?? null}
              segmentation={inferResult?.segmentation_image ?? null}
            />
          </div>

          <VerdictBadge
            verdict={inferResult?.verdict ?? null}
            anomalyScore={inferResult?.anomaly_score ?? null}
          />

          {inferError && (
            <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {inferError}
            </p>
          )}

          <div className="border-t border-gray-200 pt-4">
            <StatsPanel stats={stats} />
          </div>
        </section>
      </main>
    </div>
  );
}
