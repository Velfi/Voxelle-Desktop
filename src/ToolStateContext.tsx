// ── Tool State Context ─────────────────────────────────────────────────
// Provides tool-related state to ToolsSidebar and its children,
// replacing prop drilling for these values from App.tsx.

import { createContext, useContext } from "react";
import type {
  InteractionMode,
  ToolsPane,
  SculptStrokeModeApi,
  GeneratorKindId,
  PaintColorDistrib,
  BrushShape,
} from "./types";
import type {
  DrawStrokeModeApi,
  PlaneAxisApi,
  SelectionMethod,
  StrokeDrawStyle,
  StrokeFamilyVariant,
} from "./drawToolModel";
import type { SelectionCombineModeApi } from "./hooks/useTauriEventListeners";

export interface ToolStateContextValue {
  // Brush settings
  brushRadius: number;
  setBrushRadius: (v: number) => void;
  brushShape: BrushShape;
  setBrushShape: (v: BrushShape) => void;
  brushClipBottomHalf: boolean;
  setBrushClipBottomHalf: (v: boolean) => void;
  mirrorX: boolean;
  setMirrorX: (v: boolean) => void;
  mirrorY: boolean;
  setMirrorY: (v: boolean) => void;
  mirrorZ: boolean;
  setMirrorZ: (v: boolean) => void;

  // Stroke settings
  strokeDrawStyle: StrokeDrawStyle;
  setStrokeDrawStyle: (v: StrokeDrawStyle) => void;
  strokeFamilyVariant: StrokeFamilyVariant;
  setStrokeFamilyVariant: (v: StrokeFamilyVariant) => void;
  drawStrokeMode: DrawStrokeModeApi;
  setDrawStrokeMode: (v: DrawStrokeModeApi) => void;
  planeAxis: PlaneAxisApi;
  setPlaneAxis: (v: PlaneAxisApi) => void;
  sprayDensity: number;
  setSprayDensity: (v: number) => void;

  // Interaction mode
  interactionMode: InteractionMode;
  setInteractionMode: (v: InteractionMode) => void;
  toolsPane: ToolsPane;
  setToolsPane: (v: ToolsPane) => void;

  // Active color / material
  activeColor: number;
  setActiveColor: (v: number) => void;
  activeMaterial: string;
  setActiveMaterial: (v: string) => void;
  selectedColors: number[];
  setSelectedColors: (v: number[]) => void;
  paintColorDistrib: PaintColorDistrib;
  setPaintColorDistrib: (v: PaintColorDistrib) => void;
  matchMaterialSelectColor: boolean;
  setMatchMaterialSelectColor: (v: boolean) => void;

  // Selection options
  fillSelectDiagonals: boolean;
  setFillSelectDiagonals: (v: boolean) => void;
  fillRespectsColor: boolean;
  setFillRespectsColor: (v: boolean) => void;
  selectionCombineMode: SelectionCombineModeApi;
  setSelectionCombineMode: (v: SelectionCombineModeApi) => void;

  // Additional tool-related values from ToolsSidebar props
  selectionMethod: SelectionMethod;
  sculptStrokeMode: SculptStrokeModeApi;
  setSculptStrokeMode: (v: SculptStrokeModeApi) => void;
  generatorKind: GeneratorKindId;
  setGeneratorKind: (v: GeneratorKindId) => void;
  flySpeed: 1 | 2 | 4;
  setFlySpeed: (v: 1 | 2 | 4) => void;
}

export const ToolStateContext = createContext<ToolStateContextValue | null>(null);

/** Consume tool state. Must be used within a ToolStateContext.Provider. */
export function useToolState(): ToolStateContextValue {
  const ctx = useContext(ToolStateContext);
  if (!ctx) {
    throw new Error("useToolState must be used within a ToolStateContext.Provider");
  }
  return ctx;
}
