import { useState, useCallback, useEffect, useRef } from "react";
import {
  api,
  type InferResponse,
  type StatsResponse,
  type LoadModelResponse,
  type DatasetImage,
  type FeatureInfo,
  type DecisionRuleInfo,
  type RuleParams,
  type CalibrationState,
  type CascadeOptions,
  type CascadeSpec,
  type ModelInfo,
} from "./api/client";
import { SettingsModal } from "./components/SettingsModal";
import { ReportModal } from "./components/ReportModal";
import { InferencePanel } from "./components/InferencePanel";
import { ImageGallery } from "./components/ImageGallery";
import { ResultsDisplay } from "./components/ResultsDisplay";
import { CascadeStages } from "./components/CascadeStages";
import { VerdictBadge } from "./components/VerdictBadge";
import { StatsPanel } from "./components/StatsPanel";
import { ResizeHandle } from "./components/ResizeHandle";

const DEFAULT_SETUP_WIDTH = 288;
const DEFAULT_RESULTS_WIDTH = 480;
const MIN_GALLERY_WIDTH = 200;
const LETTERBOX_STORAGE_KEY = "anomaly-inspector.letterbox-overlay";

function readLetterboxPreference(serverDefault: boolean): boolean {
  try {
    const stored = localStorage.getItem(LETTERBOX_STORAGE_KEY);
    if (stored === "true") return true;
    if (stored === "false") return false;
  } catch {
    // localStorage may be unavailable in some contexts.
  }
  return serverDefault;
}

const clamp = (v: number, min: number, max: number) =>
  Math.min(Math.max(v, min), Math.max(min, max));

// A starter cascade: the anomaly + segmentation AND from the feature request
// when both are available, else a single stage of the first cascadable feature.
function defaultCascadeSpec(options: CascadeOptions): CascadeSpec {
  const pick = (name: string) => options.features.find((f) => f.name === name);
  const seed = (f: NonNullable<ReturnType<typeof pick>>) => {
    const rule = f.rules.find((r) => r.name === f.default_rule) ?? f.rules[0];
    return {
      feature: f.name,
      rule: rule?.name ?? "",
      threshold: 0.5,
      params: {
        ...Object.fromEntries((rule?.params ?? []).map((p) => [p.name, p.default])),
        ...(rule?.defaults ?? {}),
      },
    };
  };
  const anomaly = pick("anomaly_detection");
  const segmentation = pick("segmentation");
  const stages =
    anomaly && segmentation
      ? [seed(anomaly), seed(segmentation)]
      : options.features[0]
        ? [seed(options.features[0])]
        : [];
  return { combinator: "and", short_circuit: true, stages };
}

// Controls are seeded from the backend's effective defaults (schema + model
// sidecar). The browser echoes these back on every inference, so seeding from
// the bare schema would silently discard the sidecar's configuration.
function seedParams(rule: DecisionRuleInfo | null): RuleParams {
  if (!rule) return {};
  return {
    ...Object.fromEntries(rule.params.map((p) => [p.name, p.default])),
    ...(rule.defaults ?? {}),
  };
}

export default function App() {
  const [modelStatus, setModelStatus] = useState<LoadModelResponse | null>(null);
  const [features, setFeatures] = useState<FeatureInfo[]>([]);
  const [activeFeature, setActiveFeature] = useState<string | null>(null);
  const [featureSwitching, setFeatureSwitching] = useState(false);

  const [selectedSku, setSelectedSku] = useState<string>("");
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [imagesLoading, setImagesLoading] = useState(false);
  // Locally-uploaded images, keyed by their synthetic gallery path. Inferring
  // one of these uploads the file to the backend.
  const [uploadedFiles, setUploadedFiles] = useState<Record<string, File>>({});

  // Decision Rule selection + params. They travel per-request like `threshold`,
  // so the browser holds the live values and the backend stays stateless.
  const [decisionRules, setDecisionRules] = useState<DecisionRuleInfo[]>([]);
  const [classLabels, setClassLabels] = useState<Record<string, string>>({});
  const [selectedRule, setSelectedRule] = useState<string | null>(null);
  const [ruleParams, setRuleParams] = useState<RuleParams>({});
  // Per-SKU baseline. Fetched into ruleParams so the browser stays the single
  // source of truth for what an inference runs with.
  const [calibration, setCalibration] = useState<CalibrationState | null>(null);
  const [calibrationGroup, setCalibrationGroup] = useState<string | null>(null);
  const [calibrating, setCalibrating] = useState(false);

  // Cascade mode: options for the builder and the spec being edited. The spec
  // travels per request, so the browser is its source of truth.
  const [cascadeOptions, setCascadeOptions] = useState<CascadeOptions | null>(null);
  const [cascadeSpec, setCascadeSpec] = useState<CascadeSpec | null>(null);
  // Which Features have a model uploaded (per-Feature store; no defaults load).
  const [loadedModels, setLoadedModels] = useState<Record<string, ModelInfo>>({});
  const [uploadingFeature, setUploadingFeature] = useState<string | null>(null);

  const [customerName, setCustomerName] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [letterboxOverlay, setLetterboxOverlay] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [originalSrc, setOriginalSrc] = useState<string | null>(null);
  const [inferringKey, setInferringKey] = useState<string | null>(null);
  const [inferResult, setInferResult] = useState<InferResponse | null>(null);
  const [verdicts, setVerdicts] = useState<Record<string, "ok" | "not_ok">>({});
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [inferError, setInferError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);

  // Batch ("Run all") state: per-image results collected while running the
  // whole folder sequentially. `batchRunning` gates the button; `batchDone`
  // and `batchTotal` drive the progress bar.
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchDone, setBatchDone] = useState(0);
  const [batchTotal, setBatchTotal] = useState(0);
  const [batchRows, setBatchRows] = useState<
    { path: string; name: string; url?: string; verdict?: "ok" | "not_ok"; score?: number | null; error?: string }[]
  >([]);
  const batchCancelRef = useRef(false);

  // Resizable layout: Setup and Results have explicit widths; Gallery flexes.
  const [setupWidth, setSetupWidth] = useState(DEFAULT_SETUP_WIDTH);
  const [resultsWidth, setResultsWidth] = useState(DEFAULT_RESULTS_WIDTH);

  const modelLoaded = !!modelStatus;
  const isCascade = activeFeature === "cascade";
  const activeRule = decisionRules.find((r) => r.name === selectedRule) ?? null;
  // Can an inspection run? Cascade needs a model for every stage's Feature;
  // single-Feature mode needs one for the active Feature.
  const cascadeReady =
    isCascade &&
    !!cascadeSpec &&
    cascadeSpec.stages.length > 0 &&
    cascadeSpec.stages.every((st) => !!loadedModels[st.feature]);
  const canInfer = isCascade ? cascadeReady : !!activeFeature && !!loadedModels[activeFeature];
  // Image groups are the gallery's categories (dataset subfolders), which is how
  // an operator already separates known-good caps from the rest.
  const calibrationGroups = Array.from(new Set(images.map((i) => i.category))).sort();
  const activeFeatureInfo = features.find((f) => f.name === activeFeature) ?? null;
  const thresholdLabel = activeFeatureInfo?.threshold_label ?? "Threshold";
  const resultFeature = inferResult?.feature ?? activeFeature ?? "anomaly_detection";

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

  // On load, reflect which models are already uploaded (nothing auto-loads) and
  // fetch the browsable SKU list.
  useEffect(() => {
    api
      .getStatus()
      .then((status) => {
        setFeatures(status.features ?? []);
        setActiveFeature(status.active_feature ?? null);
        setLoadedModels(status.loaded_models ?? {});
        setLetterboxOverlay(
          readLetterboxPreference(status.overlay_options?.letterbox_default ?? false)
        );
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
  }, []);

  // The rules a Feature can run depend on what its decode produces, so refetch
  // whenever the Feature changes and seed params from the schema defaults.
  const loadDecisionRules = useCallback(async () => {
    try {
      const res = await api.getDecisionRules();
      setDecisionRules(res.rules);
      setClassLabels(res.labels ?? {});
      const preferred =
        res.rules.find((r) => r.name === res.default_rule) ?? res.rules[0] ?? null;
      setSelectedRule(preferred?.name ?? null);
      setRuleParams(seedParams(preferred));
    } catch {
      setDecisionRules([]);
      setSelectedRule(null);
      setRuleParams({});
    }
  }, []);

  useEffect(() => {
    if (modelLoaded) void loadDecisionRules();
  }, [modelLoaded, activeFeature, loadDecisionRules]);

  useEffect(() => {
    if (!isCascade || cascadeOptions) return;
    api
      .getCascadeOptions()
      .then((opts) => {
        setCascadeOptions(opts);
        setCascadeSpec((prev) => prev ?? defaultCascadeSpec(opts));
      })
      .catch(() => {});
  }, [isCascade, cascadeOptions]);

  // A taught baseline belongs to (SKU, rule), so refetch when either changes and
  // fold it over the schema/sidecar defaults.
  const loadCalibration = useCallback(async () => {
    if (!selectedSku || !selectedRule) {
      setCalibration(null);
      return;
    }
    try {
      const state = await api.getCalibration(selectedSku, selectedRule);
      setCalibration(state);
      if (state.calibrated) {
        setRuleParams((prev) => ({ ...prev, ...state.params }));
      }
    } catch {
      setCalibration(null);
    }
  }, [selectedSku, selectedRule]);

  useEffect(() => {
    void loadCalibration();
  }, [loadCalibration]);

  const handleLetterboxOverlayChange = useCallback((value: boolean) => {
    setLetterboxOverlay(value);
    try {
      localStorage.setItem(LETTERBOX_STORAGE_KEY, value ? "true" : "false");
    } catch {
      // Ignore storage failures; the in-session value still applies.
    }
  }, []);

  const handleCalibrate = useCallback(async () => {
    if (!selectedSku || !selectedRule || !calibrationGroup) return;
    const group = images.filter((i) => i.category === calibrationGroup);
    // All images are browser-uploaded (folder picker); send them as files.
    const uploadFiles: File[] = [];
    for (const img of group) {
      const file = uploadedFiles[img.path];
      if (file) uploadFiles.push(file);
    }
    if (uploadFiles.length === 0) return;
    setCalibrating(true);
    setInferError(null);
    try {
      const result = await api.calibrate(
        selectedSku, selectedRule, threshold, ruleParams, uploadFiles, letterboxOverlay
      );
      setCalibration({ ...result, calibrated: true });
      setRuleParams((prev) => ({ ...prev, ...result.params }));
      if (result.skipped.length > 0) {
        setInferError(
          `Calibrated on ${result.sample_count} of ${uploadFiles.length} images; ` +
            `${result.skipped.length} could not be measured.`
        );
      }
    } catch (e) {
      setInferError(e instanceof Error ? e.message : "Calibration failed.");
    } finally {
      setCalibrating(false);
    }
  }, [selectedSku, selectedRule, calibrationGroup, images, uploadedFiles, threshold, ruleParams, letterboxOverlay]);

  const handleClearCalibration = useCallback(async () => {
    if (!selectedSku || !selectedRule) return;
    try {
      await api.clearCalibration(selectedSku, selectedRule);
      const rule = decisionRules.find((r) => r.name === selectedRule);
      const param = rule?.calibration?.param;
      if (param) {
        const fallback = seedParams(rule ?? null)[param] ?? 0;
        setRuleParams((prev) => ({ ...prev, [param]: fallback }));
      }
      setCalibration({
        sku_name: selectedSku, decision_rule: selectedRule,
        calibrated: false, params: {},
      });
    } catch (e) {
      setInferError(e instanceof Error ? e.message : "Failed to clear calibration.");
    }
  }, [selectedSku, selectedRule, decisionRules]);

  // Keep stats scoped to the current SKU, Feature, and Decision Rule: an OK from
  // one rule is not comparable to an OK from another.
  useEffect(() => {
    api.getStats(selectedSku || null, activeFeature, selectedRule).then(setStats).catch(() => {});
  }, [selectedSku, activeFeature, selectedRule]);

  useEffect(() => {
    setCalibrationGroup((prev) =>
      prev && calibrationGroups.includes(prev) ? prev : calibrationGroups[0] ?? null
    );
  }, [calibrationGroups.join("|")]);

  const handleModelLoaded = useCallback((result: LoadModelResponse) => {
    setModelStatus(result);
    // Keep the per-Feature indicators and cascade options in step with uploads.
    api.getStatus().then((st) => setLoadedModels(st.loaded_models ?? {})).catch(() => {});
    if (isCascade) api.getCascadeOptions().then(setCascadeOptions).catch(() => {});
    // Re-fetch here explicitly: when a model was already loaded for the active
    // Feature, `modelLoaded`/`activeFeature` don't change on re-upload, so the
    // effect that normally refetches decision rules won't fire and the new
    // sidecar's expected_classes (and other rule defaults) would stay stale.
    void loadDecisionRules();
  }, [isCascade, loadDecisionRules]);

  // Upload a model (and optional sidecar) for one cascade stage's Feature.
  const handleUploadStageModel = useCallback(
    async (feature: string, model: File, version: string, sidecar: File | null) => {
      setUploadingFeature(feature);
      setInferError(null);
      try {
        await api.loadModel(model, version, feature, sidecar);
        const [st, opts] = await Promise.all([api.getStatus(), api.getCascadeOptions()]);
        setLoadedModels(st.loaded_models ?? {});
        setCascadeOptions(opts);
      } catch (e) {
        setInferError(e instanceof Error ? e.message : "Model upload failed.");
      } finally {
        setUploadingFeature(null);
      }
    },
    []
  );

  const handleFeatureChange = useCallback(
    async (feature: string) => {
      if (feature === activeFeature || featureSwitching) return;
      setFeatureSwitching(true);
      setInferError(null);
      try {
        const res = await api.setFeature(feature);
        setActiveFeature(res.active_feature);
        setModelStatus({
          status: "loaded",
          runtime: res.runtime,
          input_shape: res.input_shape,
          model_version: res.model_version,
        });
        // A different Feature means different result shapes — clear prior output.
        setInferResult(null);
        setSelectedPath(null);
        setOriginalSrc(null);
        setVerdicts({});
      } catch (e) {
        setInferError(e instanceof Error ? e.message : "Failed to switch feature.");
      } finally {
        setFeatureSwitching(false);
      }
    },
    [activeFeature, featureSwitching]
  );

  const handleRuleChange = useCallback(
    (name: string) => {
      const rule = decisionRules.find((r) => r.name === name);
      setSelectedRule(name);
      setRuleParams(seedParams(rule ?? null));
      setInferResult(null);
    },
    [decisionRules]
  );

  const handleRuleParamChange = useCallback((name: string, value: unknown) => {
    setRuleParams((prev) => ({ ...prev, [name]: value }));
  }, []);

  const handleResetDatabase = useCallback(async () => {
    const confirmed = window.confirm(
      "Erase all recorded inspections? This clears the data used for reports and cannot be undone."
    );
    if (!confirmed) return;
    setResetting(true);
    try {
      await api.resetDatabase();
      setStats(await api.getStats(selectedSku, activeFeature, selectedRule));
      setInferResult(null);
      setSelectedPath(null);
      setOriginalSrc(null);
      setVerdicts({});
      setInferError(null);
    } catch (e) {
      setInferError(e instanceof Error ? e.message : "Failed to reset database.");
    } finally {
      setResetting(false);
    }
  }, [selectedSku, activeFeature, selectedRule]);

  async function runInferByPath(path: string) {
    if (!canInfer || !selectedSku || inferringKey || batchRunning) return;
    const file = uploadedFiles[path];
    if (!file) return;
    const image = images.find((i) => i.path === path);
    setSelectedPath(path);
    setOriginalSrc(image?.url ?? null);
    setInferError(null);
    setInferringKey(path);
    try {
      const cascade = isCascade ? cascadeSpec : null;
      const result = await api.infer(
        file,
        selectedSku,
        threshold,
        customerName,
        selectedRule,
        ruleParams,
        cascade,
        letterboxOverlay
      );
      setInferResult(result);
      setVerdicts((v) => ({ ...v, [path]: result.verdict }));
      setStats(await api.getStats(selectedSku || null, activeFeature, selectedRule));
    } catch (e) {
      setInferError(e instanceof Error ? e.message : "Inference failed.");
    } finally {
      setInferringKey(null);
    }
  }

  // Run inference over every image in the uploaded folder, sequentially. A
  // single failure does not abort the run; the row records the error and the
  // run continues. The Stop button flips `batchCancelRef` between images.
  async function runAllInferences() {
    if (!canInfer || !selectedSku || batchRunning) return;
    const entries = Object.entries(uploadedFiles);
    if (entries.length === 0) return;
    setBatchRunning(true);
    batchCancelRef.current = false;
    setBatchDone(0);
    setBatchTotal(entries.length);
    setBatchRows(entries.map(([path, file]) => ({
      path,
      name: file.name,
      url: URL.createObjectURL(file),
    })));
    setInferError(null);
    const cascade = isCascade ? cascadeSpec : null;
    let done = 0;
    for (const [path, file] of entries) {
      if (batchCancelRef.current) break;
      setInferringKey(path);
      setSelectedPath(path);
      try {
        const result = await api.infer(
        file,
        selectedSku,
        threshold,
        customerName,
        selectedRule,
        ruleParams,
        cascade,
        letterboxOverlay
      );
        setBatchRows((rows) =>
          rows.map((r) => (r.path === path ? { ...r, verdict: result.verdict, score: result.score ?? null } : r))
        );
        setVerdicts((v) => ({ ...v, [path]: result.verdict }));
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Inference failed.";
        setBatchRows((rows) =>
          rows.map((r) => (r.path === path ? { ...r, error: msg } : r))
        );
      } finally {
        done += 1;
        setBatchDone(done);
        setInferringKey(null);
      }
    }
    setStats(await api.getStats(selectedSku || null, activeFeature, selectedRule));
    setBatchRunning(false);
    setInferringKey(null);
  }

  function stopBatch() {
    batchCancelRef.current = true;
  }

  // Populate the gallery from a locally-selected folder. The images live in the
  // browser (not the dataset), so each gets a synthetic path and an object URL
  // for its thumbnail; inference uploads the underlying File.
  function handleUploadDirectory(files: FileList) {
    if (!selectedSku) return;
    const picked = Array.from(files).filter((f) =>
      /\.(png|jpe?g|bmp)$/i.test(f.name)
    );
    if (picked.length === 0) {
      setInferError("No images found in the selected folder.");
      return;
    }
    const fileMap: Record<string, File> = {};
    const uploadedImages: DatasetImage[] = picked.map((f) => {
      const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
      const path = `__upload__/${rel}`;
      const parts = rel.split("/");
      const category = parts.length > 1 ? parts[parts.length - 2] : "Uploaded";
      fileMap[path] = f;
      return { path, category, name: f.name, url: URL.createObjectURL(f) };
    });
    setUploadedFiles(fileMap);
    setImages(uploadedImages);
    setSelectedPath(null);
    setOriginalSrc(null);
    setInferResult(null);
    setInferError(null);
    setVerdicts({});
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
          {!isCascade && (
            <SettingsModal
              onModelLoaded={handleModelLoaded}
              feature={activeFeature}
              isLoaded={!!activeFeature && !!loadedModels[activeFeature]}
              modelVersion={activeFeature ? loadedModels[activeFeature]?.model_version : undefined}
              runtime={activeFeature ? loadedModels[activeFeature]?.runtime : undefined}
            />
          )}
        </div>
      </header>

      {/* Main layout — Setup | Gallery | Results */}
      <main className="flex flex-1 overflow-hidden min-h-0">
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
            selectedSku={selectedSku}
            threshold={threshold}
            letterboxOverlay={letterboxOverlay}
            customerName={customerName}
            modelLoaded={modelLoaded}
            features={features}
            activeFeature={activeFeature}
            featureSwitching={featureSwitching}
            thresholdLabel={thresholdLabel}
            decisionRules={decisionRules}
            classLabels={classLabels}
            selectedRule={selectedRule}
            ruleParams={ruleParams}
            calibration={calibration}
            calibrationGroups={calibrationGroups}
            calibrationGroup={calibrationGroup}
            calibrating={calibrating}
            isCascade={isCascade}
            cascadeOptions={cascadeOptions}
            cascadeSpec={cascadeSpec}
            loadedModels={loadedModels}
            uploadingFeature={uploadingFeature}
            onUploadModel={handleUploadStageModel}
            onCascadeChange={setCascadeSpec}
            onRuleChange={handleRuleChange}
            onRuleParamChange={handleRuleParamChange}
            onCalibrationGroupChange={setCalibrationGroup}
            onCalibrate={handleCalibrate}
            onClearCalibration={handleClearCalibration}
            onFeatureChange={handleFeatureChange}
            onSkuChange={setSelectedSku}
            onThresholdChange={setThreshold}
            onLetterboxOverlayChange={handleLetterboxOverlayChange}
            onCustomerChange={setCustomerName}
            onUploadDirectory={handleUploadDirectory}
            onRunAll={runAllInferences}
            batchRunning={batchRunning}
          />
          <div className="mt-auto border-t border-gray-200 p-4">
            <StatsPanel stats={stats} skuName={selectedSku} onReset={handleResetDatabase} resetting={resetting} />
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
              disabled={!canInfer || !!inferringKey}
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
          className="shrink-0 overflow-y-auto bg-white p-4 space-y-4"
        >
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Result
          </h2>

          <VerdictBadge
            verdict={inferResult?.verdict ?? null}
            score={inferResult?.score ?? null}
            scoreLabel={inferResult?.score_label || activeRule?.label || "Score"}
            reason={inferResult?.reason}
          />

          {inferError && (
            <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {inferError}
            </p>
          )}

          <ResultsDisplay
            feature={resultFeature}
            originalPreview={originalSrc}
            heatmap={inferResult?.heatmap_image ?? null}
            segmentation={inferResult?.segmentation_image ?? null}
            annotated={inferResult?.annotated_image ?? inferResult?.overlay_image ?? null}
            detections={inferResult?.detections ?? null}
            metrics={inferResult?.feature === "cascade" ? null : inferResult?.metrics ?? null}
          />

          {inferResult?.feature === "cascade" && (
            <CascadeStages stages={inferResult.stages ?? []} />
          )}

          {/* Batch ("Run all") progress + per-image results table */}
          {(batchRunning || batchRows.length > 0) && (
            <div className="border-t border-gray-100 pt-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Batch Results
                </h3>
                {batchRunning ? (
                  <button
                    type="button"
                    onClick={stopBatch}
                    className="text-xs font-medium text-red-600 hover:text-red-700"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setBatchRows([])}
                    className="text-xs font-medium text-gray-400 hover:text-gray-600"
                  >
                    Clear
                  </button>
                )}
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 rounded-full bg-gray-100 overflow-hidden">
                  <div
                    className="h-full bg-yellow-400 transition-all"
                    style={{
                      width: batchTotal ? `${(batchDone / batchTotal) * 100}%` : "0%",
                    }}
                  />
                </div>
                <span className="text-[11px] text-gray-500 tabular-nums">
                  {batchDone}/{batchTotal}
                </span>
              </div>
              <div className="max-h-64 overflow-y-auto rounded-lg border border-gray-100">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-gray-50 text-gray-400">
                    <tr>
                      <th className="text-left font-medium px-2 py-1.5">Image</th>
                      <th className="text-left font-medium px-2 py-1.5">Verdict</th>
                      <th className="text-left font-medium px-2 py-1.5">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchRows.map((r) => (
                      <tr key={r.path} className="border-t border-gray-50">
                        <td className="px-2 py-1.5 flex items-center gap-2">
                          {r.url && (
                            <img
                              src={r.url}
                              alt={r.name}
                              className="h-7 w-7 rounded object-cover"
                            />
                          )}
                          <span className="truncate max-w-[160px]">{r.name}</span>
                        </td>
                        <td className="px-2 py-1.5">
                          {r.error ? (
                            <span className="text-red-600">error</span>
                          ) : r.verdict ? (
                            <span
                              className={
                                r.verdict === "ok" ? "text-green-600" : "text-red-600"
                              }
                            >
                              {r.verdict === "ok" ? "OK" : "NOK"}
                            </span>
                          ) : (
                            <span className="text-gray-300">…</span>
                          )}
                        </td>
                        <td className="px-2 py-1.5 tabular-nums text-gray-600">
                          {r.error ? (
                            <span className="text-red-500 truncate max-w-[120px]" title={r.error}>
                              {r.error}
                            </span>
                          ) : r.score != null ? (
                            r.score.toFixed(2)
                          ) : (
                            "—"
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
