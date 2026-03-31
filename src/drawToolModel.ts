/**
 * Canonical Tool × Selection Method model for the Draw pane.
 * Maps to Rust `DrawStrokeMode` + stroke style / spray / solid variant on the wire.
 */

/** `line` = anchor-to-cursor line. `brush` = follow ray + connect samples. */
export type StrokeDrawStyle = "line" | "brush";

/** Matches Rust `stroke_modes::DrawStrokeMode` (JSON camelCase). */
export type DrawStrokeModeApi =
  | "line"
  | "spray"
  | "plane"
  | "precise"
  | "circle"
  | "cuboid"
  | "cylinder"
  | "polygonHull"
  | "polygon"
  | "fill";

/** Plane constraint for `plane` stroke and fill (Rust `PlaneAxis`). */
export type PlaneAxisApi = "auto" | "x" | "y" | "z" | "camera";

/** Distinguishes Stroke vs Solid when both use line stroke + no spray (web parity). */
export type StrokeFamilyVariant = "stroke" | "solid";

/** High-level selection method (sidebar Stroke / Surface / Solid / Spray / Fill). */
export type SelectionMethod = "stroke" | "surface" | "solid" | "spray" | "fill";

export type DrawTool = "add" | "remove" | "paint" | "select";

export function strokeModeSkipsDrag(mode: DrawStrokeModeApi): boolean {
  return mode === "fill" || mode === "polygon" || mode === "polygonHull";
}

export function deriveSelectionMethod(s: {
  drawStrokeMode: DrawStrokeModeApi;
  strokeDrawStyle: StrokeDrawStyle;
  sprayDensity: number;
  strokeFamilyVariant: StrokeFamilyVariant;
}): SelectionMethod {
  if (s.drawStrokeMode === "fill") return "fill";
  /** Spray stroke mode (`DrawStrokeMode::Spray`) or brush path with scatter &gt; 0. */
  if (
    s.drawStrokeMode === "spray" ||
    (s.strokeDrawStyle === "brush" && s.sprayDensity > 0)
  ) {
    return "spray";
  }
  if (s.strokeDrawStyle === "brush" && s.sprayDensity === 0) {
    return "surface";
  }
  if (
    s.strokeDrawStyle === "line" &&
    s.sprayDensity === 0 &&
    s.strokeFamilyVariant === "solid"
  ) {
    return "solid";
  }
  if (
    s.strokeDrawStyle === "line" &&
    s.sprayDensity === 0 &&
    s.strokeFamilyVariant === "stroke"
  ) {
    return "stroke";
  }
  return "stroke";
}

/** Sidebar button handlers: set underlying state to match a selection method. */
export function selectionMethodToState(method: SelectionMethod): {
  drawStrokeMode: DrawStrokeModeApi;
  strokeDrawStyle: StrokeDrawStyle;
  sprayDensity: number;
  strokeFamilyVariant: StrokeFamilyVariant;
} {
  switch (method) {
    case "fill":
      return {
        drawStrokeMode: "fill",
        strokeDrawStyle: "line",
        sprayDensity: 0,
        strokeFamilyVariant: "stroke",
      };
    case "spray":
      return {
        drawStrokeMode: "spray",
        strokeDrawStyle: "brush",
        sprayDensity: 0,
        strokeFamilyVariant: "stroke",
      };
    case "surface":
      return {
        drawStrokeMode: "plane",
        strokeDrawStyle: "brush",
        sprayDensity: 0,
        strokeFamilyVariant: "stroke",
      };
    case "solid":
      return {
        drawStrokeMode: "cuboid",
        strokeDrawStyle: "line",
        sprayDensity: 0,
        strokeFamilyVariant: "solid",
      };
    case "stroke":
    default:
      return {
        drawStrokeMode: "line",
        strokeDrawStyle: "line",
        sprayDensity: 0,
        strokeFamilyVariant: "stroke",
      };
  }
}

/** True when sidebar "Stroke" method is active (narrow stroke + line + no spray). */
export function isNarrowStrokeSelectionMethod(s: {
  drawStrokeMode: DrawStrokeModeApi;
  strokeDrawStyle: StrokeDrawStyle;
  sprayDensity: number;
  strokeFamilyVariant: StrokeFamilyVariant;
}): boolean {
  return deriveSelectionMethod(s) === "stroke";
}

export function drawToolFromInteractionMode(mode: string): DrawTool | null {
  if (mode === "add") return "add";
  if (mode === "remove") return "remove";
  if (mode === "paint") return "paint";
  if (
    mode === "select" ||
    mode === "selectByColor" ||
    mode === "selectCoplanar" ||
    mode === "selectCoplanarEmpty"
  ) {
    return "select";
  }
  return null;
}

export function isMenubarTemporarySelectMode(mode: string): boolean {
  return (
    mode === "selectByColor" ||
    mode === "selectCoplanar" ||
    mode === "selectCoplanarEmpty"
  );
}

/**
 * Stroke dispatch info: determines whether a voxel stroke should route to
 * the edit commands (`voxel_edit_at_screen`, etc.) or the selection commands
 * (`selection_stroke_at_screen`, etc.).
 */
export type StrokeDispatch =
  | { kind: "edit"; tool: "add" | "remove" | "paint" }
  | { kind: "selection"; interaction: string };

/**
 * Returns the stroke dispatch for an interaction mode, or `null` if the mode
 * is not a voxel stroke mode (e.g. navigate, sculpt, eyedropper).
 */
export function getStrokeDispatch(mode: string): StrokeDispatch | null {
  if (mode === "add") return { kind: "edit", tool: "add" };
  if (mode === "remove") return { kind: "edit", tool: "remove" };
  if (mode === "paint") return { kind: "edit", tool: "paint" };
  if (mode === "select") return { kind: "selection", interaction: "select" };
  if (mode === "selectByColor")
    return { kind: "selection", interaction: "selectByColor" };
  if (mode === "selectCoplanar")
    return { kind: "selection", interaction: "selectCoplanar" };
  if (mode === "selectCoplanarEmpty")
    return { kind: "selection", interaction: "selectCoplanarEmpty" };
  return null;
}
