import { Loader2 } from "lucide-react";
import { api, type DatasetImage } from "../api/client";
import { cn } from "../lib/utils";

interface Props {
  images: DatasetImage[];
  loading: boolean;
  selectedPath: string | null;
  inferringPath: string | null;
  verdicts: Record<string, "ok" | "not_ok">;
  disabled: boolean;
  onSelect: (path: string) => void;
}

export function ImageGallery({
  images,
  loading,
  selectedPath,
  inferringPath,
  verdicts,
  disabled,
  onSelect,
}: Props) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full gap-2 text-gray-400">
        <Loader2 className="h-5 w-5 animate-spin" /> Loading images…
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-gray-400">
        Select a SKU or upload a folder to browse images.
      </div>
    );
  }

  // Group by defect category, preserving the order returned by the API.
  const groups: { category: string; items: DatasetImage[] }[] = [];
  for (const img of images) {
    let g = groups.find((x) => x.category === img.category);
    if (!g) {
      g = { category: img.category, items: [] };
      groups.push(g);
    }
    g.items.push(img);
  }

  return (
    <div className="flex flex-col gap-5">
      {groups.map((g) => (
        <div key={g.category}>
          <div className="flex items-center gap-2 mb-2 sticky top-0 bg-gray-50 py-1 z-10">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500">
              {g.category}
            </h3>
            <span className="text-[11px] text-gray-400">{g.items.length}</span>
          </div>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(84px,1fr))] gap-2">
            {g.items.map((img) => {
              const verdict = verdicts[img.path];
              const isSelected = selectedPath === img.path;
              const isInferring = inferringPath === img.path;
              return (
                <button
                  key={img.path}
                  type="button"
                  disabled={disabled}
                  onClick={() => onSelect(img.path)}
                  title={img.name}
                  className={cn(
                    "relative aspect-square rounded-xl overflow-hidden border bg-white transition-all disabled:cursor-not-allowed disabled:opacity-60",
                    isSelected
                      ? "border-yellow-400 ring-2 ring-yellow-300"
                      : "border-gray-200 hover:border-yellow-300"
                  )}
                >
                  <img
                    src={img.url ?? api.imageUrl(img.path)}
                    alt={img.name}
                    loading="lazy"
                    className="w-full h-full object-cover"
                  />
                  {verdict && (
                    <span
                      className={cn(
                        "absolute top-1 right-1 h-2.5 w-2.5 rounded-full ring-1 ring-white",
                        verdict === "ok" ? "bg-green-500" : "bg-red-500"
                      )}
                    />
                  )}
                  {isInferring && (
                    <span className="absolute inset-0 flex items-center justify-center bg-white/60">
                      <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
