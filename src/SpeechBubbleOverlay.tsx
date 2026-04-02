/**
 * SpeechBubbleOverlay — transparent click-capture divs positioned over each
 * GPU-rendered speech bubble.
 *
 * The actual bubble geometry (rounded rect + tail) and text are rendered by
 * the wgpu speech-bubble pipeline directly on the swapchain surface.
 * These invisible divs live at the same screen coordinates so the browser
 * can route pointer events to us, which we forward to Rust via Tauri.
 */

import { invoke } from "@tauri-apps/api/core";

export interface BubbleInfo {
  /** Must match the id passed to `speech_bubble_show`. */
  id: number;
  /** Bubble position in CSS pixels (viewport-relative). */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Extra margin below the bubble body to include the tail in the hit area (CSS px). */
  tailMargin?: number;
}

interface SpeechBubbleOverlayProps {
  bubbles: BubbleInfo[];
}

export function SpeechBubbleOverlay({ bubbles }: SpeechBubbleOverlayProps) {
  if (bubbles.length === 0) return null;

  return (
    <>
      {bubbles.map((b) => (
        <div
          key={b.id}
          style={{
            position: "fixed",
            left: b.x,
            top: b.y,
            width: b.width,
            // Extend hit-area downward to cover the tail triangle.
            height: b.height + (b.tailMargin ?? 32),
            background: "transparent",
            // Above mascot (z 5) but below modals (z 10).
            zIndex: 6,
            cursor: "pointer",
            pointerEvents: "auto",
          }}
          onClick={() => {
            void invoke("speech_bubble_click", { id: b.id });
          }}
        />
      ))}
    </>
  );
}
