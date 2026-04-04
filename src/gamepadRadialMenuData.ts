import type { InteractionMode, SculptStrokeModeApi, ToolsPane } from "./types";
import type { SelectionMethod } from "./drawToolModel";

// ---------------------------------------------------------------------------
// Slice descriptor shared by both radial menus
// ---------------------------------------------------------------------------

export interface RadialSlice {
  id: string;
  label: string;
  icon: string;
}

// ---------------------------------------------------------------------------
// Left-trigger menu: top-level tools
// ---------------------------------------------------------------------------

export const TOOL_SLICES: RadialSlice[] = [
  { id: "add", label: "Add", icon: "\u270f\ufe0f" },
  { id: "remove", label: "Remove", icon: "\ud83d\uddd1" },
  { id: "paint", label: "Paint", icon: "\ud83c\udfa8" },
  { id: "select", label: "Select", icon: "\u2b1c" },
  { id: "sculpt", label: "Sculpt", icon: "\ud83c\udfd4" },
  { id: "eyedropper", label: "Pick", icon: "\ud83d\udc89" },
  { id: "generator", label: "Generate", icon: "\ud83c\udf3f" },
  { id: "squishy", label: "Squishy", icon: "\ud83e\udee7" },
];

/** Map a tool slice id to the InteractionMode + ToolsPane it should activate. */
export function toolSliceToMode(id: string): {
  interactionMode: InteractionMode;
  toolsPane: ToolsPane;
} {
  switch (id) {
    case "add":
      return { interactionMode: "add", toolsPane: "draw" };
    case "remove":
      return { interactionMode: "remove", toolsPane: "draw" };
    case "paint":
      return { interactionMode: "paint", toolsPane: "draw" };
    case "select":
      return { interactionMode: "select", toolsPane: "select" };
    case "sculpt":
      return { interactionMode: "sculpt", toolsPane: "sculpt" };
    case "eyedropper":
      return { interactionMode: "eyedropper", toolsPane: "draw" };
    case "generator":
      return { interactionMode: "generator", toolsPane: "generators" };
    case "squishy":
      return { interactionMode: "squishy", toolsPane: "squishy" };
    default:
      return { interactionMode: "add", toolsPane: "draw" };
  }
}

// ---------------------------------------------------------------------------
// Right-trigger menu: sub-options (context-dependent)
// ---------------------------------------------------------------------------

export type SubOptionChoice =
  | { kind: "selectionMethod"; method: SelectionMethod }
  | { kind: "sculptMode"; mode: SculptStrokeModeApi };

const SELECTION_METHOD_SLICES: RadialSlice[] = [
  { id: "stroke", label: "Stroke", icon: "\u2712\ufe0f" },
  { id: "surface", label: "Surface", icon: "\ud83d\udfe2" },
  { id: "solid", label: "Solid", icon: "\ud83e\uddf1" },
  { id: "spray", label: "Spray", icon: "\ud83d\udca8" },
  { id: "fill", label: "Fill", icon: "\ud83e\udeb3" },
];

const SCULPT_MODE_SLICES: RadialSlice[] = [
  { id: "draw", label: "Draw", icon: "\u270f\ufe0f" },
  { id: "smooth", label: "Smooth", icon: "\ud83e\uddca" },
  { id: "gouge", label: "Gouge", icon: "\u26cf\ufe0f" },
  { id: "wall", label: "Wall", icon: "\ud83e\uddf1" },
  { id: "terrain", label: "Terrain", icon: "\ud83c\udf0d" },
  { id: "extrude", label: "Extrude", icon: "\u2b06\ufe0f" },
];

export function getSubOptionSlices(interactionMode: InteractionMode): RadialSlice[] {
  switch (interactionMode) {
    case "add":
    case "remove":
    case "paint":
    case "select":
    case "selectByColor":
    case "selectCoplanar":
    case "selectCoplanarEmpty":
      return SELECTION_METHOD_SLICES;
    case "sculpt":
      return SCULPT_MODE_SLICES;
    default:
      return [];
  }
}

export function subOptionSliceToChoice(
  id: string,
  interactionMode: InteractionMode,
): SubOptionChoice | null {
  switch (interactionMode) {
    case "add":
    case "remove":
    case "paint":
    case "select":
    case "selectByColor":
    case "selectCoplanar":
    case "selectCoplanarEmpty":
      return { kind: "selectionMethod", method: id as SelectionMethod };
    case "sculpt":
      return { kind: "sculptMode", mode: id as SculptStrokeModeApi };
    default:
      return null;
  }
}
