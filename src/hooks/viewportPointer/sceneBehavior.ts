/**
 * Pipeline B (scene tools) policy helpers — not used for Pipeline A (gizmos / gestureRef).
 * See docs/agents/viewport-pointer-pipeline.md
 */

import type { InteractionMode } from "../../types";

/** Subset of scene behavior derived only from interaction mode (no refs / phases). */
export interface ViewportSceneBehavior {
  /**
   * When true, idle pointer move may call `sync_preview_input` with hover coords
   * (still gated in the hook by overGizmo, probing, generator phases, etc.).
   */
  allowsIdleHoverPreviewSync: boolean;
  /**
   * When false, pointer leave clears preview (unless a phased tool is active — checked in hook).
   */
  preservePreviewOnPointerLeave: boolean;
}

const IDLE_HOVER_PREVIEW_MODES: ReadonlySet<InteractionMode> = new Set([
  "add",
  "remove",
  "paint",
  "sculpt",
  "select",
  "selectByColor",
  "selectCoplanar",
  "selectCoplanarEmpty",
  "squishy",
  "bone",
  "generator",
  "stamp",
  "punch",
  "selectExtrude",
]);

/** Modes that keep preview when the pointer leaves the viewport (sidebar/tool chrome). */
const PRESERVE_PREVIEW_ON_LEAVE_MODES: ReadonlySet<InteractionMode> = new Set([
  "select",
  "selectByColor",
  "selectCoplanar",
  "selectCoplanarEmpty",
  "selectExtrude",
  "squishy",
  "bone",
  "generator",
]);

export function getViewportSceneBehavior(mode: InteractionMode): ViewportSceneBehavior {
  return {
    allowsIdleHoverPreviewSync: IDLE_HOVER_PREVIEW_MODES.has(mode),
    preservePreviewOnPointerLeave: PRESERVE_PREVIEW_ON_LEAVE_MODES.has(mode),
  };
}

/** @deprecated Prefer getViewportSceneBehavior(mode).allowsIdleHoverPreviewSync */
export function allowsIdleHoverPreviewSyncForMode(mode: InteractionMode): boolean {
  return getViewportSceneBehavior(mode).allowsIdleHoverPreviewSync;
}
