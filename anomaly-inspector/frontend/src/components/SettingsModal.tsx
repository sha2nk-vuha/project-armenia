import { useState } from "react";
import { Settings } from "lucide-react";
import { api, type LoadModelResponse } from "../api/client";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";

interface Props {
  onModelLoaded: (result: LoadModelResponse) => void;
  // The Feature this upload is for; the model lands in that Feature's store slot.
  feature?: string | null;
  isLoaded: boolean;
  modelVersion?: string;
  runtime?: string;
}

export function SettingsModal({ onModelLoaded, feature, isLoaded, modelVersion, runtime }: Props) {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [version, setVersion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleLoad() {
    if (!file || !version.trim()) {
      setError("Select a model file and enter a version.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await api.loadModel(file, version.trim(), feature ?? undefined);
      onModelLoaded(result);
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load model.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-1.5">
          <Settings className="h-4 w-4" />
          <span className="text-xs">Model</span>
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Model Configuration</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm">
            <span
              className={`h-2.5 w-2.5 rounded-full ${isLoaded ? "bg-green-500" : "bg-gray-300"}`}
            />
            <span className="text-gray-600">
              {isLoaded
                ? `Loaded — ${modelVersion} (${runtime})`
                : "No model loaded"}
            </span>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="model-file">Model File (.onnx)</Label>
            <Input
              id="model-file"
              type="file"
              accept=".onnx"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="cursor-pointer"
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="model-version">Model Version</Label>
            <Input
              id="model-version"
              placeholder="e.g. v1.0-patchcore"
              value={version}
              onChange={(e) => setVersion(e.target.value)}
            />
          </div>

          {error && <p className="text-xs text-red-600">{error}</p>}

          <Button onClick={handleLoad} disabled={loading} className="w-full">
            {loading ? "Loading…" : "Load Model"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
