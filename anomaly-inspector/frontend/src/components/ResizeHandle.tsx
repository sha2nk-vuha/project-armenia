import { useEffect, useRef } from "react";

interface Props {
  /** Called with the pointer's viewport X while dragging. */
  onDrag: (clientX: number) => void;
  /** Optional: called on double-click, e.g. to reset to a default width. */
  onReset?: () => void;
}

/**
 * A thin vertical divider that reports the pointer X while dragged. The parent
 * decides which panel to resize. Listeners live on `window` so the drag keeps
 * tracking even when the cursor moves off the 6px handle.
 */
export function ResizeHandle({ onDrag, onReset }: Props) {
  const dragging = useRef(false);

  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragging.current) return;
      e.preventDefault();
      onDrag(e.clientX);
    };
    const stop = () => {
      if (!dragging.current) return;
      dragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", stop);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", stop);
    };
  }, [onDrag]);

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      onMouseDown={() => {
        dragging.current = true;
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
      }}
      onDoubleClick={onReset}
      className="w-1.5 shrink-0 cursor-col-resize bg-gray-200 hover:bg-blue-400 active:bg-blue-500 transition-colors"
      title="Drag to resize"
    />
  );
}
