import { useEffect, useRef, useState } from "react";
import type { Detection } from "../api/client";

interface ImagePanelProps {
  src: string | null;
  label: string;
}

function ImagePanel({ src, label }: ImagePanelProps) {
  return (
    <div className="flex flex-col gap-1 flex-1 min-w-0">
      <span className="text-xs font-medium text-gray-500 text-center">{label}</span>
      <div className="aspect-square w-full rounded-2xl bg-gray-100 border border-gray-200 overflow-hidden flex items-center justify-center">
        {src ? (
          <img src={src} alt={label} className="w-full h-full object-contain" />
        ) : (
          <span className="text-xs text-gray-400">—</span>
        )}
      </div>
    </div>
  );
}

interface Props {
  feature: string;
  originalPreview: string | null;
  heatmap: string | null;
  segmentation: string | null;
  annotated: string | null;
  detections: Detection[] | null;
}

// Below this container width the panels stack vertically; at or above it they
// sit side by side. ~600px gives each panel a usable size.
const HORIZONTAL_BREAKPOINT = 600;

export function ResultsDisplay({
  feature,
  originalPreview,
  heatmap,
  segmentation,
  annotated,
  detections,
}: Props) {
  const heatmapSrc = heatmap ? `data:image/jpeg;base64,${heatmap}` : null;
  const segSrc = segmentation ? `data:image/jpeg;base64,${segmentation}` : null;
  const annotatedSrc = annotated ? `data:image/jpeg;base64,${annotated}` : null;

  // Reflow based on the container's own width (works for the resizable pane).
  const ref = useRef<HTMLDivElement>(null);
  const [horizontal, setHorizontal] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setHorizontal(entry.contentRect.width >= HORIZONTAL_BREAKPOINT);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const isPresence = feature === "presence_absence";

  return (
    <div className="flex flex-col gap-3 w-full">
      <div
        ref={ref}
        className={`flex gap-3 w-full ${horizontal ? "flex-row items-start" : "flex-col"}`}
      >
        <ImagePanel src={originalPreview} label="Original" />
        {isPresence ? (
          <ImagePanel src={annotatedSrc} label="Detections" />
        ) : (
          <>
            <ImagePanel src={heatmapSrc} label="Heatmap" />
            <ImagePanel src={segSrc} label="Segmentation" />
          </>
        )}
      </div>

      {isPresence && detections && (
        <div className="rounded-xl border border-gray-200 overflow-hidden">
          <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Detections ({detections.length})
          </div>
          {detections.length === 0 ? (
            <p className="px-3 py-2 text-sm text-gray-400">No objects detected.</p>
          ) : (
            <ul className="divide-y divide-gray-100">
              {detections.map((d, i) => (
                <li key={i} className="flex items-center justify-between px-3 py-1.5 text-sm">
                  <span className="font-medium text-gray-700">{d.label}</span>
                  <span className="font-mono text-xs text-gray-500">
                    {(d.confidence * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
