/**
 * MascotView — click-detection overlay for a start-screen voxel mascot.
 *
 * The mascot geometry is rendered by the wgpu pipeline (render_mascots in render/mod.rs)
 * directly on top of the start-screen swapchain content.  This component is an invisible
 * fixed-position <div> that sits at the same screen coordinates and captures pointer
 * events so the user can click the mascot.
 *
 * Coordinates passed to mascot_set_screen_rect must be in **physical pixels**
 * (i.e. already multiplied by window.devicePixelRatio).
 */

import { invoke } from "@tauri-apps/api/core";
import { useEffect } from "react";

export interface MascotRect {
  /** Left edge in CSS pixels (viewport-relative). */
  x: number;
  /** Top edge in CSS pixels (viewport-relative). */
  y: number;
  /** Width in CSS pixels. */
  width: number;
  /** Height in CSS pixels. */
  height: number;
}

interface MascotViewProps {
  id: number;
  rect: MascotRect;
  visible: boolean;
  onClick?: (id: number) => void;
}

export function MascotView({ id, rect, visible, onClick }: MascotViewProps) {
  const dpr = window.devicePixelRatio || 1;

  // Keep the Rust side in sync whenever position/visibility changes.
  useEffect(() => {
    void invoke("mascot_set_screen_rect", {
      id,
      x: rect.x * dpr,
      y: rect.y * dpr,
      w: rect.width * dpr,
      h: rect.height * dpr,
    });
  }, [id, rect.x, rect.y, rect.width, rect.height, dpr]);

  useEffect(() => {
    void invoke("mascot_set_visible", { id, visible });
    return () => {
      // Hide when unmounted.
      void invoke("mascot_set_visible", { id, visible: false });
    };
  }, [id, visible]);

  if (!visible) return null;

  return (
    <div
      style={{
        position: "fixed",
        left: rect.x,
        top: rect.y,
        width: rect.width,
        height: rect.height,
        // Transparent background — wgpu draws through this area.
        background: "transparent",
        // Below modals (z-index 10) but above the viewport content.
        zIndex: 5,
        cursor: "pointer",
        // No pointer-event passthrough — this div captures clicks.
        pointerEvents: "auto",
      }}
      onClick={() => onClick?.(id)}
    />
  );
}
