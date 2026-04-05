// ── Tools sidebar ─────────────────────────────────────────────────────
// Extracted from App.tsx to reduce file size.

import { useRef } from "react";
import { useToolState } from "./ToolStateContext";
import type {
  InteractionMode,
  PaintColorDistrib,
  PaintColorMode,
  FbmParams,
  GradientParams,
  DitherParams,
} from "./types";
import { selectionMethodToState } from "./drawToolModel";
import { MATERIAL_OPTIONS } from "./constants";
import { MATERIAL_BUILTIN_PALETTE_HEX } from "./materialBuiltinPalette";
import { ViewportSettingsSidebar } from "./ViewportSettingsSidebar";

// ── Helper components (moved from App.tsx) ────────────────────────────

function MultiColorPaintSection(props: {
  distrib: PaintColorDistrib;
  setDistrib: (d: PaintColorDistrib) => void;
}) {
  const { distrib, setDistrib } = props;
  const patch = (part: Partial<PaintColorDistrib>) => setDistrib({ ...distrib, ...part });
  const patchFbm = (part: Partial<FbmParams>) => patch({ fbm: { ...distrib.fbm, ...part } });
  const patchGrad = (part: Partial<GradientParams>) =>
    patch({ gradient: { ...distrib.gradient, ...part } });
  const patchDither = (part: Partial<DitherParams>) =>
    patch({ dither: { ...distrib.dither, ...part } });
  return (
    <div className="multi-color-section">
      <div className="sidebar-section-label">
        Color distribution
        {distrib.mode !== "whiteNoise" && distrib.mode !== "randomSingle"
          ? ` · ${distrib.mode === "fbmNoise" ? "FBM" : distrib.mode === "gradient" ? "Gradient" : "Dither"}`
          : ""}
      </div>
      <div className="sidebar-row">
        <label className="sidebar-label-sm">Mode</label>
        <select
          className="sidebar-select-sm"
          value={distrib.mode}
          onChange={(e) => patch({ mode: e.target.value as PaintColorMode })}
        >
          <option value="whiteNoise">White noise</option>
          <option value="randomSingle">Single random per stroke</option>
          <option value="fbmNoise">FBM noise</option>
          <option value="gradient">Gradient</option>
          <option value="dither">Dither</option>
        </select>
      </div>
      {distrib.mode === "fbmNoise" && (
        <>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">Octaves {distrib.fbm.octaves}</label>
            <input
              type="range"
              min={1}
              max={12}
              step={1}
              value={distrib.fbm.octaves}
              onChange={(e) => patchFbm({ octaves: Number(e.target.value) })}
            />
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">Frequency {distrib.fbm.frequency.toFixed(2)}</label>
            <input
              type="range"
              min={0.02}
              max={0.8}
              step={0.01}
              value={distrib.fbm.frequency}
              onChange={(e) => patchFbm({ frequency: Number(e.target.value) })}
            />
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">
              Lacunarity {distrib.fbm.lacunarity.toFixed(2)}
            </label>
            <input
              type="range"
              min={1.5}
              max={3.5}
              step={0.05}
              value={distrib.fbm.lacunarity}
              onChange={(e) => patchFbm({ lacunarity: Number(e.target.value) })}
            />
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">
              Persistence {distrib.fbm.persistence.toFixed(2)}
            </label>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={distrib.fbm.persistence}
              onChange={(e) => patchFbm({ persistence: Number(e.target.value) })}
            />
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">
              <input
                type="checkbox"
                checked={distrib.fbm.quantized}
                onChange={(e) => patchFbm({ quantized: e.target.checked })}
              />{" "}
              Quantized (palette steps)
            </label>
          </div>
        </>
      )}
      {distrib.mode === "gradient" && (
        <>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">Kind</label>
            <select
              className="sidebar-select-sm"
              value={distrib.gradient.kind}
              onChange={(e) => patchGrad({ kind: e.target.value as "linear" | "radial" })}
            >
              <option value="linear">Linear</option>
              <option value="radial">Radial</option>
            </select>
          </div>
          {distrib.gradient.kind === "linear" && (
            <div className="sidebar-row">
              <label className="sidebar-label-sm">Axis</label>
              <select
                className="sidebar-select-sm"
                value={distrib.gradient.linearAxis}
                onChange={(e) => patchGrad({ linearAxis: Number(e.target.value) as 0 | 1 | 2 })}
              >
                <option value={0}>X</option>
                <option value={1}>Y</option>
                <option value={2}>Z</option>
              </select>
            </div>
          )}
          {distrib.gradient.kind === "radial" && (
            <div className="sidebar-row sidebar-row-inline">
              <label className="sidebar-label-sm">Center</label>
              {(["X", "Y", "Z"] as const).map((axis, i) => (
                <label key={axis} className="sidebar-label-sm">
                  {axis}
                  <input
                    type="number"
                    className="sidebar-number-sm"
                    style={{ width: "3.5rem" }}
                    value={distrib.gradient.radialCenter[i]}
                    onChange={(e) => {
                      const c = [...distrib.gradient.radialCenter] as [number, number, number];
                      c[i] = Number(e.target.value);
                      patchGrad({ radialCenter: c });
                    }}
                  />
                </label>
              ))}
            </div>
          )}
          <div className="sidebar-row">
            <label className="sidebar-label-sm">Scale {distrib.gradient.scale.toFixed(3)}</label>
            <input
              type="range"
              min={0.01}
              max={0.5}
              step={0.005}
              value={distrib.gradient.scale}
              onChange={(e) => patchGrad({ scale: Number(e.target.value) })}
            />
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">Phase {distrib.gradient.phase.toFixed(2)}</label>
            <input
              type="range"
              min={-3.15}
              max={3.15}
              step={0.05}
              value={distrib.gradient.phase}
              onChange={(e) => patchGrad({ phase: Number(e.target.value) })}
            />
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">
              <input
                type="checkbox"
                checked={distrib.gradient.quantized}
                onChange={(e) => patchGrad({ quantized: e.target.checked })}
              />{" "}
              Quantized (palette steps)
            </label>
          </div>
        </>
      )}
      {distrib.mode === "dither" && (
        <>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">Bayer size</label>
            <select
              className="sidebar-select-sm"
              value={distrib.dither.orderedSize}
              onChange={(e) =>
                patchDither({
                  orderedSize: Number(e.target.value) as 2 | 4 | 8,
                })
              }
            >
              <option value={2}>2×2</option>
              <option value={4}>4×4</option>
              <option value={8}>8×8</option>
            </select>
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">
              Strength {distrib.dither.orderedStrength.toFixed(2)}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={distrib.dither.orderedStrength}
              onChange={(e) => patchDither({ orderedStrength: Number(e.target.value) })}
            />
          </div>
          <div className="sidebar-row">
            <label className="sidebar-label-sm">Error diffusion</label>
            <select
              className="sidebar-select-sm"
              value={distrib.dither.errorDiffusion}
              onChange={(e) =>
                patchDither({
                  errorDiffusion: e.target.value as "none" | "floydSteinberg",
                })
              }
            >
              <option value="none">None (ordered only)</option>
              <option value="floydSteinberg">Floyd–Steinberg</option>
            </select>
          </div>
        </>
      )}
    </div>
  );
}

/** Palette swatch grid with multi-select support (click, shift+click, drag, shift+drag). */
export function PaletteSwatches(props: {
  activeColor: number;
  selectedColors: number[];
  setActiveColor: (n: number) => void;
  setSelectedColors: (c: number[]) => void;
  disabled: boolean;
  palette: readonly string[];
}) {
  const { activeColor, selectedColors, setActiveColor, setSelectedColors, disabled, palette } =
    props;
  const dragStartIdxRef = useRef<number | null>(null);
  const dragEndIdxRef = useRef<number | null>(null);
  const isDraggingRef = useRef(false);
  const shiftHeldRef = useRef(false);

  function getSwatchIndex(el: Element): number {
    const idx = el.getAttribute("data-idx");
    return idx !== null ? Number(idx) : -1;
  }

  function selectRange(lo: number, hi: number, baseColors?: number[]): number[] {
    const [a, b] = lo <= hi ? [lo, hi] : [hi, lo];
    const rangeRgbs = palette.slice(a, b + 1).map((hex) => Number.parseInt(hex.slice(1), 16));
    if (baseColors) {
      const merged = new Set([...baseColors, ...rangeRgbs]);
      return Array.from(merged);
    }
    return rangeRgbs;
  }

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (disabled) return;
    const target = (e.target as HTMLElement).closest("[data-idx]");
    if (!target) return;
    const idx = getSwatchIndex(target);
    if (idx < 0) return;
    shiftHeldRef.current = e.shiftKey;
    isDraggingRef.current = false;
    dragStartIdxRef.current = idx;
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (disabled || dragStartIdxRef.current === null) return;
    const target = document.elementFromPoint(e.clientX, e.clientY);
    if (!target) return;
    const swatchEl = target.closest("[data-idx]");
    if (!swatchEl) return;
    const idx = getSwatchIndex(swatchEl);
    if (idx < 0) return;
    isDraggingRef.current = true;
    dragEndIdxRef.current = idx;
    const newRange = selectRange(
      dragStartIdxRef.current,
      idx,
      shiftHeldRef.current ? selectedColors : undefined,
    );
    setSelectedColors(newRange);
  }

  function handlePointerUp(e: React.PointerEvent<HTMLDivElement>) {
    if (disabled || dragStartIdxRef.current === null) return;
    const startIdx = dragStartIdxRef.current;
    dragStartIdxRef.current = null;
    const isSingleSwatch =
      !isDraggingRef.current ||
      dragEndIdxRef.current === null ||
      dragEndIdxRef.current === startIdx;
    dragEndIdxRef.current = null;
    isDraggingRef.current = false;
    if (isSingleSwatch) {
      // Single click or drag that stayed on one swatch
      const rgb = Number.parseInt(palette[startIdx]!.slice(1), 16);
      if (e.shiftKey) {
        // Toggle in selected list
        const alreadySelected = selectedColors.includes(rgb);
        if (alreadySelected) {
          const next = selectedColors.filter((c) => c !== rgb);
          setSelectedColors(next);
        } else {
          setSelectedColors([...selectedColors, rgb]);
        }
      } else {
        // Plain click: clear multi-select, set single color
        setSelectedColors([]);
        setActiveColor(rgb);
      }
    }
  }

  const selectedSet = new Set(selectedColors);

  return (
    <div
      className={`sidebar-palette-swatches${disabled ? " is-disabled" : ""}`}
      role="group"
      aria-label="Material color palette (shift+click or drag to multi-select)"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {palette.map((hex, idx) => {
        const rgb = Number.parseInt(hex.slice(1), 16);
        const isActive = selectedColors.length === 0 && (activeColor & 0xffffff) === rgb;
        const isSelected = selectedSet.has(rgb);
        let cls = "sidebar-palette-swatch";
        if (isActive) cls += " is-active";
        if (isSelected) cls += " is-selected";
        return (
          <div
            key={hex}
            data-idx={idx}
            className={cls}
            style={{ backgroundColor: hex }}
            title={`${hex}${isSelected ? " (selected)" : ""}`}
            role="button"
            aria-pressed={isActive || isSelected}
            aria-label={`Color ${hex}`}
          />
        );
      })}
    </div>
  );
}

function SymmetryColorSidebarSections(props: {
  loading: boolean;
  workBusy: boolean;
  activeColor: number;
  setActiveColor: (n: number) => void;
  selectedColors: number[];
  setSelectedColors: (c: number[]) => void;
  paintColorDistrib: PaintColorDistrib;
  setPaintColorDistrib: (d: PaintColorDistrib) => void;
  interactionMode: InteractionMode;
  setInteractionMode: (m: InteractionMode) => void;
  mirrorX: boolean;
  setMirrorX: (v: boolean) => void;
  mirrorY: boolean;
  setMirrorY: (v: boolean) => void;
  mirrorZ: boolean;
  setMirrorZ: (v: boolean) => void;
}) {
  const {
    loading,
    workBusy,
    activeColor,
    setActiveColor,
    selectedColors,
    setSelectedColors,
    paintColorDistrib,
    setPaintColorDistrib,
    interactionMode,
    setInteractionMode,
    mirrorX,
    setMirrorX,
    mirrorY,
    setMirrorY,
    mirrorZ,
    setMirrorZ,
  } = props;
  return (
    <div className="sidebar-symmetry-color-panel">
      <div className="sidebar-section-label">Symmetry</div>
      <div className="sidebar-mode-grid sidebar-mode-grid-3">
        <button
          type="button"
          className={mirrorX ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
          onClick={() => setMirrorX(!mirrorX)}
          title="Mirror across X axis"
        >
          <span className="sidebar-mode-label">X</span>
        </button>
        <button
          type="button"
          className={mirrorY ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
          onClick={() => setMirrorY(!mirrorY)}
          title="Mirror across Y axis"
        >
          <span className="sidebar-mode-label">Y</span>
        </button>
        <button
          type="button"
          className={mirrorZ ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
          onClick={() => setMirrorZ(!mirrorZ)}
          title="Mirror across Z axis"
        >
          <span className="sidebar-mode-label">Z</span>
        </button>
      </div>

      <div className="sidebar-color-stack">
        <div className="sidebar-section-label">Color</div>
        <div className="sidebar-color-row">
          <label className="sidebar-palette-row sidebar-color-swatch">
            <input
              type="color"
              value={`#${activeColor.toString(16).padStart(6, "0")}`}
              onChange={(ev) => {
                const h = ev.target.value.slice(1);
                const n = Number.parseInt(h, 16);
                if (!Number.isNaN(n)) setActiveColor(n);
              }}
              disabled={loading || workBusy}
              aria-label="Brush color"
            />
          </label>
          <button
            type="button"
            className={
              interactionMode === "eyedropper" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"
            }
            disabled={loading || workBusy}
            onClick={() => setInteractionMode("eyedropper")}
          >
            <span className="sidebar-mode-label">Eyedropper</span>
          </button>
        </div>
        <div className="multi-color-hint">
          {selectedColors.length > 0 ? (
            <>
              {selectedColors.length} colors selected
              {selectedColors.length > 1 &&
                ` · ${paintColorDistrib.mode === "whiteNoise" ? "white noise" : paintColorDistrib.mode === "randomSingle" ? "random single" : paintColorDistrib.mode === "fbmNoise" ? "FBM" : paintColorDistrib.mode === "gradient" ? "gradient" : "dither"}`}
              <button
                type="button"
                className="multi-color-clear-btn"
                onClick={() => setSelectedColors([])}
                title="Clear multi-color selection"
              >
                ✕
              </button>
            </>
          ) : (
            <span className="multi-color-hint-placeholder">Shift+drag to multi-select</span>
          )}
        </div>
        <PaletteSwatches
          activeColor={activeColor}
          selectedColors={selectedColors}
          setActiveColor={setActiveColor}
          setSelectedColors={setSelectedColors}
          disabled={loading || workBusy}
          palette={MATERIAL_BUILTIN_PALETTE_HEX}
        />
        {selectedColors.length > 1 && (
          <MultiColorPaintSection distrib={paintColorDistrib} setDistrib={setPaintColorDistrib} />
        )}
      </div>
    </div>
  );
}

// ── Minimal interface for stroke-phase handles used in the sidebar ────

export interface SidebarPhaseHandle {
  readonly active: boolean;
  readonly snapshot: { phase: string } | null;
  cancel(): void;
}

// ── Props ─────────────────────────────────────────────────────────────

export interface ToolsSidebarProps {
  showEditorChrome: boolean;
  loading: boolean;
  workBusy: boolean;

  // Sidebar chrome
  toolsPaneFloating: boolean;
  setToolsPaneFloating: (v: boolean) => void;
  sidebarExpanded: boolean;
  setSidebarExpanded: React.Dispatch<React.SetStateAction<boolean>>;
  toolPanePos: { x: number; y: number };
  onToolPaneDragDown: (e: React.PointerEvent) => void;

  // Selection / stamp
  selectionCount: number;
  stampBookPatternActive: boolean;

  // Generator phase handles
  ropePhase: SidebarPhaseHandle;
  clothPhase: SidebarPhaseHandle;
  rocksPhase: SidebarPhaseHandle;
  grassPhase: SidebarPhaseHandle;
  ashlarPhase: SidebarPhaseHandle;
  floraPhase: SidebarPhaseHandle;
  shapePhase: SidebarPhaseHandle;

  // Rope / cloth state
  ropeFirstScreen: { nx: number; ny: number } | null;
  setRopeFirstScreen: (v: { nx: number; ny: number } | null) => void;
  setClothPins: (pins: [number, number, number][]) => void;
  clothPinsRef: React.MutableRefObject<[number, number, number][]>;

  // Bone
  bonePhase: SidebarPhaseHandle;
  boneMode: "add" | "edit" | "delete";
  setBoneMode: (m: "add" | "edit" | "delete") => void;
  boneJointCount: number;
  boneBoneCount: number;
  boneDefaultRadius: number;
  setBoneDefaultRadius: (n: number) => void;
  ikEnabled: boolean;
  setIkEnabled: (v: boolean) => void;

  // Collapsed palette toggle
  colorPaletteFloating: boolean;
  setColorPaletteFloating: (v: boolean) => void;
}

// ── Component ─────────────────────────────────────────────────────────

export function ToolsSidebar(props: ToolsSidebarProps) {
  const {
    showEditorChrome,
    loading,
    workBusy,
    toolsPaneFloating,
    setToolsPaneFloating,
    sidebarExpanded,
    setSidebarExpanded,
    toolPanePos,
    onToolPaneDragDown,
    selectionCount,
    stampBookPatternActive,
    ropePhase,
    clothPhase,
    rocksPhase,
    grassPhase,
    ashlarPhase,
    floraPhase,
    shapePhase,
    ropeFirstScreen,
    setRopeFirstScreen,
    setClothPins,
    clothPinsRef,
    bonePhase,
    boneMode,
    setBoneMode,
    boneJointCount,
    boneBoneCount,
    boneDefaultRadius,
    setBoneDefaultRadius,
    ikEnabled,
    setIkEnabled,
    colorPaletteFloating,
    setColorPaletteFloating,
  } = props;

  const {
    toolsPane,
    setToolsPane,
    interactionMode,
    setInteractionMode,
    flySpeed,
    setFlySpeed,
    selectionMethod,
    setDrawStrokeMode,
    setStrokeDrawStyle,
    setSprayDensity,
    setStrokeFamilyVariant,
    activeColor,
    setActiveColor,
    selectedColors,
    setSelectedColors,
    paintColorDistrib,
    setPaintColorDistrib,
    mirrorX,
    setMirrorX,
    mirrorY,
    setMirrorY,
    mirrorZ,
    setMirrorZ,
    activeMaterial,
    setActiveMaterial,
    sculptStrokeMode,
    setSculptStrokeMode,
    generatorKind,
    setGeneratorKind,
  } = useToolState();

  if (!showEditorChrome) return null;

  return (
    <>
      {toolsPaneFloating && showEditorChrome ? (
        <div className="app-sidebar-spacer" aria-hidden />
      ) : null}
      {showEditorChrome ? (
        <aside
          className={`${
            sidebarExpanded ? "app-sidebar is-expanded" : "app-sidebar is-collapsed"
          }${toolsPaneFloating ? " is-floating" : ""}`}
          style={toolsPaneFloating ? { left: toolPanePos.x, top: toolPanePos.y } : undefined}
          aria-label="Tools"
        >
          <div
            className={
              toolsPaneFloating ? "sidebar-header sidebar-header-floating" : "sidebar-header"
            }
          >
            {toolsPaneFloating ? (
              <>
                <div
                  className="floating-tools-drag-handle"
                  onPointerDown={onToolPaneDragDown}
                  aria-label="Drag to move tools"
                >
                  <span className="floating-tools-grip" aria-hidden>
                    ⋮⋮
                  </span>
                  {sidebarExpanded ? <span className="floating-tools-title">Tools</span> : null}
                </div>
                <div className="floating-tools-header-actions">
                  <button
                    type="button"
                    className="floating-tools-dock-btn"
                    onClick={() => setToolsPaneFloating(false)}
                    title="Dock tools to the left edge"
                  >
                    Dock
                  </button>
                  <button
                    type="button"
                    className="sidebar-expand-toggle floating-tools-collapse-toggle"
                    onClick={() => setSidebarExpanded((v) => !v)}
                    aria-expanded={sidebarExpanded}
                    title={sidebarExpanded ? "Collapse tools" : "Expand tools"}
                  >
                    <span className="sidebar-expand-toggle-icon" aria-hidden>
                      {sidebarExpanded ? "«" : "»"}
                    </span>
                  </button>
                </div>
              </>
            ) : (
              <div className="sidebar-tools-header-row">
                <button
                  type="button"
                  className="sidebar-expand-toggle"
                  onClick={() => setSidebarExpanded((v) => !v)}
                  aria-expanded={sidebarExpanded}
                  title={sidebarExpanded ? "Collapse tools" : "Expand tools"}
                >
                  <span className="sidebar-expand-toggle-icon" aria-hidden>
                    {sidebarExpanded ? "«" : "»"}
                  </span>
                  {sidebarExpanded ? (
                    <span className="sidebar-expand-toggle-label">Tools</span>
                  ) : null}
                </button>
                <button
                  type="button"
                  className="sidebar-float-btn"
                  onClick={() => setToolsPaneFloating(true)}
                  title="Float tools panel"
                  aria-label="Float tools panel"
                >
                  ⧉
                </button>
              </div>
            )}
          </div>
          <div className="sidebar-scroll">
            {sidebarExpanded ? (
              <>
                <div className="sidebar-toolpane-tabs" role="tablist" aria-label="Tool panes">
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "hand" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "hand"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("hand");
                      setInteractionMode("navigate");
                    }}
                  >
                    ✋
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "walk" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "walk"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("walk");
                      setInteractionMode("walk");
                    }}
                  >
                    Walk
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "fly" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "fly"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("fly");
                      setInteractionMode("fly");
                    }}
                  >
                    Fly
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "draw" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "draw"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("draw");
                      setInteractionMode("add");
                    }}
                  >
                    Draw
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "sculpt" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "sculpt"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("sculpt");
                      setInteractionMode("sculpt");
                    }}
                  >
                    Sculpt
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "select" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "select"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("select");
                      setInteractionMode("select");
                    }}
                  >
                    Select
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "generators" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "generators"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("generators");
                      setInteractionMode("generator");
                    }}
                  >
                    Generators
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "squishy" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "squishy"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("squishy");
                      setInteractionMode("squishy");
                    }}
                  >
                    Squishy
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "bone" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "bone"}
                    disabled={loading || workBusy}
                    onClick={() => {
                      setToolsPane("bone");
                      setInteractionMode("bone");
                    }}
                  >
                    Boney
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={
                      toolsPane === "mood" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                    }
                    aria-selected={toolsPane === "mood"}
                    disabled={loading || workBusy}
                    onClick={() => setToolsPane("mood")}
                  >
                    Mood
                  </button>
                </div>
                <div className="sidebar-expanded-slot" aria-label="Tool pane options">
                  {toolsPane === "hand" ? (
                    <p className="sidebar-pane-hint">Drag in viewport to orbit/pan.</p>
                  ) : null}

                  {toolsPane === "fly" ? (
                    <>
                      <p className="sidebar-pane-hint">
                        Click viewport to capture pointer. WASD move, E/Q up/down, Shift slow. Mouse
                        looks. Esc releases pointer.
                      </p>
                      <div className="sidebar-section-label">Speed</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-3">
                        {([1, 2, 4] as const).map((s) => (
                          <button
                            key={s}
                            type="button"
                            className={
                              flySpeed === s ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"
                            }
                            onClick={() => setFlySpeed(s)}
                          >
                            <span className="sidebar-mode-label">{s}×</span>
                          </button>
                        ))}
                      </div>
                    </>
                  ) : null}

                  {toolsPane === "walk" ? (
                    <p className="sidebar-pane-hint">
                      Click viewport to capture pointer. WASD to walk, Space to jump, Shift slow.
                      Mouse looks. Esc releases pointer.
                    </p>
                  ) : null}

                  {toolsPane === "draw" ? (
                    <>
                      <div className="sidebar-section-label">Tool</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-3">
                        {(["add", "remove", "paint"] as const).map((m) => (
                          <button
                            key={m}
                            type="button"
                            data-mode={m}
                            className={
                              interactionMode === m
                                ? "sidebar-mode-btn is-active"
                                : "sidebar-mode-btn"
                            }
                            disabled={loading || workBusy}
                            onClick={() => setInteractionMode(m)}
                          >
                            <span className="sidebar-mode-label">
                              {m[0].toUpperCase() + m.slice(1)}
                            </span>
                          </button>
                        ))}
                      </div>

                      <div className="sidebar-section-label">Selection method</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-3">
                        <button
                          type="button"
                          className={
                            selectionMethod === "stroke"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("stroke");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Line from pointer down to cursor (web Stroke)"
                        >
                          <span className="sidebar-mode-label">Stroke</span>
                        </button>
                        <button
                          type="button"
                          className={
                            selectionMethod === "surface"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("surface");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Plane / circle / polygon in the face plane (web Surface)"
                        >
                          <span className="sidebar-mode-label">Surface</span>
                        </button>
                        <button
                          type="button"
                          className={
                            selectionMethod === "solid"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("solid");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Solid volume: cube/cylinder/polygon (web Solid)"
                        >
                          <span className="sidebar-mode-label">Solid</span>
                        </button>
                      </div>
                      <div
                        className="sidebar-mode-grid sidebar-mode-grid-2"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <button
                          type="button"
                          className={
                            selectionMethod === "spray"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("spray");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Spray density along brush path"
                        >
                          <span className="sidebar-mode-label">Spray</span>
                        </button>
                        <button
                          type="button"
                          className={
                            selectionMethod === "fill"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("fill");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Fill connected region (add / remove / paint / selection)"
                        >
                          <span className="sidebar-mode-label">Fill</span>
                        </button>
                      </div>

                      <SymmetryColorSidebarSections
                        loading={loading}
                        workBusy={workBusy}
                        activeColor={activeColor}
                        setActiveColor={setActiveColor}
                        interactionMode={interactionMode}
                        setInteractionMode={setInteractionMode}
                        selectedColors={selectedColors}
                        setSelectedColors={setSelectedColors}
                        paintColorDistrib={paintColorDistrib}
                        setPaintColorDistrib={setPaintColorDistrib}
                        mirrorX={mirrorX}
                        setMirrorX={setMirrorX}
                        mirrorY={mirrorY}
                        setMirrorY={setMirrorY}
                        mirrorZ={mirrorZ}
                        setMirrorZ={setMirrorZ}
                      />

                      <div className="sidebar-section-label">Material</div>
                      <select
                        className="sidebar-material-select"
                        value={activeMaterial}
                        onChange={(ev) => setActiveMaterial(ev.target.value)}
                        disabled={loading || workBusy}
                        aria-label="Material"
                      >
                        {MATERIAL_OPTIONS.map((o) => (
                          <option key={o.id} value={o.id}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </>
                  ) : null}

                  {toolsPane === "select" ? (
                    <>
                      <div className="sidebar-section-label">Selection</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-stacked">
                        <button
                          type="button"
                          className={
                            interactionMode === "select"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setInteractionMode("select")}
                        >
                          <span className="sidebar-mode-label">Select</span>
                        </button>
                        <button
                          type="button"
                          className={
                            interactionMode === "stamp"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={
                            loading || workBusy || (selectionCount === 0 && !stampBookPatternActive)
                          }
                          onClick={() => setInteractionMode("stamp")}
                        >
                          <span className="sidebar-mode-label">Stamp</span>
                        </button>
                        <button
                          type="button"
                          className={
                            interactionMode === "punch"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy || selectionCount === 0}
                          onClick={() => setInteractionMode("punch")}
                        >
                          <span className="sidebar-mode-label">Punch</span>
                        </button>
                        <button
                          type="button"
                          className={
                            interactionMode === "selectExtrude"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy || selectionCount === 0}
                          onClick={() => setInteractionMode("selectExtrude")}
                        >
                          <span className="sidebar-mode-label">Extrude</span>
                        </button>
                      </div>

                      <div className="sidebar-section-label">Selection method</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-3">
                        <button
                          type="button"
                          className={
                            selectionMethod === "stroke"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("stroke");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Line from pointer down to cursor (web Stroke)"
                        >
                          <span className="sidebar-mode-label">Stroke</span>
                        </button>
                        <button
                          type="button"
                          className={
                            selectionMethod === "surface"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("surface");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Plane / circle / polygon in the face plane (web Surface)"
                        >
                          <span className="sidebar-mode-label">Surface</span>
                        </button>
                        <button
                          type="button"
                          className={
                            selectionMethod === "solid"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("solid");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Solid volume: cube/cylinder/polygon (web Solid)"
                        >
                          <span className="sidebar-mode-label">Solid</span>
                        </button>
                      </div>
                      <div
                        className="sidebar-mode-grid sidebar-mode-grid-2"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <button
                          type="button"
                          className={
                            selectionMethod === "spray"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("spray");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Spray density along brush path"
                        >
                          <span className="sidebar-mode-label">Spray</span>
                        </button>
                        <button
                          type="button"
                          className={
                            selectionMethod === "fill"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            const s = selectionMethodToState("fill");
                            setDrawStrokeMode(s.drawStrokeMode);
                            setStrokeDrawStyle(s.strokeDrawStyle);
                            setSprayDensity(s.sprayDensity);
                            setStrokeFamilyVariant(s.strokeFamilyVariant);
                          }}
                          title="Fill connected region (selection)"
                        >
                          <span className="sidebar-mode-label">Fill</span>
                        </button>
                      </div>
                    </>
                  ) : null}

                  {toolsPane === "sculpt" ? (
                    <>
                      <div className="sidebar-section-label">Sculpt mode</div>
                      <div className="sidebar-mode-grid">
                        {(
                          [
                            ["draw", "Draw"],
                            ["gouge", "Scrape"],
                            ["smooth", "Smooth"],
                            ["wall", "Wall"],
                            ["extrude", "Extrude"],
                            ["terrain", "Terrain"],
                          ] as const
                        ).map(([id, label]) => (
                          <button
                            key={id}
                            type="button"
                            className={
                              sculptStrokeMode === id
                                ? "sidebar-mode-btn is-active"
                                : "sidebar-mode-btn"
                            }
                            disabled={loading || workBusy || interactionMode !== "sculpt"}
                            onClick={() => setSculptStrokeMode(id)}
                          >
                            <span className="sidebar-mode-label">{label}</span>
                          </button>
                        ))}
                      </div>
                      <SymmetryColorSidebarSections
                        loading={loading}
                        workBusy={workBusy}
                        activeColor={activeColor}
                        setActiveColor={setActiveColor}
                        interactionMode={interactionMode}
                        setInteractionMode={setInteractionMode}
                        selectedColors={selectedColors}
                        setSelectedColors={setSelectedColors}
                        paintColorDistrib={paintColorDistrib}
                        setPaintColorDistrib={setPaintColorDistrib}
                        mirrorX={mirrorX}
                        setMirrorX={setMirrorX}
                        mirrorY={mirrorY}
                        setMirrorY={setMirrorY}
                        mirrorZ={mirrorZ}
                        setMirrorZ={setMirrorZ}
                      />
                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Brush and terrain options are in the tool options panel.
                      </p>
                    </>
                  ) : null}

                  {toolsPane === "generators" ? (
                    <>
                      <div className="sidebar-section-label">Generators</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-2">
                        {(
                          [
                            ["rocks", "Rocks"],
                            ["grass", "Grass"],
                            ["rope", "Rope"],
                            ["cloth", "Cloth"],
                            ["ashlar", "Ashlar"],
                            ["flora", "Flora"],
                            ["roof", "Roof"],
                            ["shape", "Shape"],
                          ] as const
                        ).map(([id, label]) => (
                          <button
                            key={id}
                            type="button"
                            className={
                              generatorKind === id
                                ? "sidebar-mode-btn is-active"
                                : "sidebar-mode-btn"
                            }
                            disabled={loading || workBusy}
                            onClick={() => {
                              setGeneratorKind(id);
                              if (ropePhase.active) ropePhase.cancel();
                              else setRopeFirstScreen(null);
                              if (clothPhase.active) clothPhase.cancel();
                              else {
                                setClothPins([]);
                                clothPinsRef.current = [];
                              }
                              rocksPhase.cancel();
                              grassPhase.cancel();
                              ashlarPhase.cancel();
                              floraPhase.cancel();
                              shapePhase.cancel();
                            }}
                          >
                            <span className="sidebar-mode-label">{label}</span>
                          </button>
                        ))}
                      </div>
                      <SymmetryColorSidebarSections
                        loading={loading}
                        workBusy={workBusy}
                        activeColor={activeColor}
                        setActiveColor={setActiveColor}
                        interactionMode={interactionMode}
                        setInteractionMode={setInteractionMode}
                        selectedColors={selectedColors}
                        setSelectedColors={setSelectedColors}
                        paintColorDistrib={paintColorDistrib}
                        setPaintColorDistrib={setPaintColorDistrib}
                        mirrorX={mirrorX}
                        setMirrorX={setMirrorX}
                        mirrorY={mirrorY}
                        setMirrorY={setMirrorY}
                        mirrorZ={mirrorZ}
                        setMirrorZ={setMirrorZ}
                      />
                      {generatorKind === "rope" && ropeFirstScreen && !ropePhase.active ? (
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint" role="status">
                          Click second point for rope.
                        </p>
                      ) : null}
                      {generatorKind === "rope" && ropePhase.active ? (
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint" role="status">
                          Adjust tension and sag, then Done.
                        </p>
                      ) : null}
                      {generatorKind === "cloth" && !clothPhase.active ? (
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint" role="status">
                          Cloth: click surface to add pins (3+ corners), then Done.
                        </p>
                      ) : null}
                      {generatorKind === "cloth" && clothPhase.active ? (
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint" role="status">
                          Adjust settings, then Done.
                        </p>
                      ) : null}
                      {generatorKind === "roof" ? (
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint" role="status">
                          Roof: click the surface to add pins (3+ corners), then Apply in tool
                          options.
                        </p>
                      ) : null}
                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Size and shape in the tool options panel. Rope: two clicks. Cloth: multi-pin
                        + Apply.
                      </p>
                    </>
                  ) : null}

                  {toolsPane === "squishy" ? (
                    <>
                      <div className="sidebar-section-label">Squishy</div>
                      <button
                        type="button"
                        className={
                          interactionMode === "squishy"
                            ? "sidebar-mode-btn is-active"
                            : "sidebar-mode-btn"
                        }
                        disabled={loading || workBusy}
                        onClick={() => setInteractionMode("squishy")}
                      >
                        <span className="sidebar-mode-label">Metaballs</span>
                      </button>
                      <SymmetryColorSidebarSections
                        loading={loading}
                        workBusy={workBusy}
                        activeColor={activeColor}
                        setActiveColor={setActiveColor}
                        interactionMode={interactionMode}
                        setInteractionMode={setInteractionMode}
                        selectedColors={selectedColors}
                        setSelectedColors={setSelectedColors}
                        paintColorDistrib={paintColorDistrib}
                        setPaintColorDistrib={setPaintColorDistrib}
                        mirrorX={mirrorX}
                        setMirrorX={setMirrorX}
                        mirrorY={mirrorY}
                        setMirrorY={setMirrorY}
                        mirrorZ={mirrorZ}
                        setMirrorZ={setMirrorZ}
                      />
                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Add / pick / delete blobs in the viewport; commit in tool options.
                      </p>
                    </>
                  ) : null}

                  {toolsPane === "bone" ? (
                    <>
                      <div className="sidebar-section-label">
                        Bone{" "}
                        {bonePhase.active
                          ? bonePhase.snapshot?.phase === "build"
                            ? "(Build)"
                            : "(Pose)"
                          : ""}
                      </div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-3">
                        {(bonePhase.snapshot?.phase === "pose"
                          ? (["edit", "delete"] as const)
                          : (["add", "delete"] as const)
                        ).map((m) => (
                          <button
                            key={m}
                            type="button"
                            className={
                              boneMode === m ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"
                            }
                            disabled={loading || workBusy}
                            onClick={() => setBoneMode(m)}
                          >
                            <span className="sidebar-mode-label">
                              {m[0].toUpperCase() + m.slice(1)}
                            </span>
                          </button>
                        ))}
                      </div>
                      <div className="sidebar-row">
                        <label className="sidebar-label-sm">
                          Joint Radius {boneDefaultRadius.toFixed(1)}
                        </label>
                        <input
                          type="range"
                          min={1}
                          max={20}
                          step={0.5}
                          value={boneDefaultRadius}
                          onChange={(e) => setBoneDefaultRadius(Number(e.target.value))}
                        />
                      </div>
                      {bonePhase.snapshot?.phase === "pose" && (
                        <div className="sidebar-row">
                          <label className="sidebar-label-sm">
                            <input
                              type="checkbox"
                              checked={ikEnabled}
                              onChange={(e) => setIkEnabled(e.target.checked)}
                            />{" "}
                            IK Enabled
                          </label>
                        </div>
                      )}
                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Joints: {boneJointCount} &middot; Bones: {boneBoneCount}
                      </p>
                    </>
                  ) : null}

                  {toolsPane === "mood" ? (
                    <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                      Mood sliders are in the tool options panel.
                    </p>
                  ) : null}

                  <ViewportSettingsSidebar loading={loading} workBusy={workBusy} />
                </div>
              </>
            ) : (
              <div className="sidebar-collapsed-tools">
                {/* ── Pane tabs ── */}
                {(
                  [
                    { pane: "hand", label: "Hand", mode: "navigate" },
                    { pane: "walk", label: "Walk", mode: "walk" },
                    { pane: "fly", label: "Fly", mode: "fly" },
                    { pane: "draw", label: "Draw", mode: "add" },
                    { pane: "sculpt", label: "Sculpt", mode: "sculpt" },
                    { pane: "select", label: "Sel", mode: "select" },
                    { pane: "generators", label: "Gen", mode: "generator" },
                    { pane: "squishy", label: "Sqsh", mode: "squishy" },
                    { pane: "bone", label: "Bone", mode: "bone" },
                    { pane: "mood", label: "Mood", mode: null },
                  ] as const
                ).map(({ pane, label, mode }) => (
                  <button
                    key={pane}
                    type="button"
                    className={`sidebar-collapsed-tool-btn${toolsPane === pane ? " is-active" : ""}`}
                    disabled={loading || workBusy}
                    title={label}
                    onClick={() => {
                      setToolsPane(pane as typeof toolsPane);
                      if (mode) setInteractionMode(mode);
                    }}
                  >
                    <span className="sidebar-collapsed-tool-icon">{label}</span>
                  </button>
                ))}

                {/* ── Draw sub-options ── */}
                {toolsPane === "draw" && (
                  <>
                    <div className="sidebar-collapsed-tool-separator" />
                    <div className="sidebar-collapsed-section-label">Tool</div>
                    {(["add", "remove", "paint"] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        data-mode={m}
                        className={`sidebar-collapsed-sub-btn${interactionMode === m ? " is-active" : ""}`}
                        disabled={loading || workBusy}
                        onClick={() => setInteractionMode(m)}
                      >
                        {m[0].toUpperCase() + m.slice(1)}
                      </button>
                    ))}
                    <div className="sidebar-collapsed-section-label">Method</div>
                    {(["stroke", "surface", "solid", "spray", "fill"] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        className={`sidebar-collapsed-sub-btn${selectionMethod === m ? " is-active" : ""}`}
                        disabled={loading || workBusy}
                        onClick={() => {
                          const s = selectionMethodToState(m);
                          setDrawStrokeMode(s.drawStrokeMode);
                          setStrokeDrawStyle(s.strokeDrawStyle);
                          setSprayDensity(s.sprayDensity);
                          setStrokeFamilyVariant(s.strokeFamilyVariant);
                        }}
                      >
                        {m[0].toUpperCase() + m.slice(1)}
                      </button>
                    ))}
                  </>
                )}

                {/* ── Select sub-options ── */}
                {toolsPane === "select" && (
                  <>
                    <div className="sidebar-collapsed-tool-separator" />
                    <div className="sidebar-collapsed-section-label">Selection</div>
                    <button
                      type="button"
                      className={`sidebar-collapsed-sub-btn${interactionMode === "select" ? " is-active" : ""}`}
                      disabled={loading || workBusy}
                      onClick={() => setInteractionMode("select")}
                    >
                      Select
                    </button>
                    <button
                      type="button"
                      className={`sidebar-collapsed-sub-btn${interactionMode === "stamp" ? " is-active" : ""}`}
                      disabled={
                        loading || workBusy || (selectionCount === 0 && !stampBookPatternActive)
                      }
                      onClick={() => setInteractionMode("stamp")}
                    >
                      Stamp
                    </button>
                    <button
                      type="button"
                      className={`sidebar-collapsed-sub-btn${interactionMode === "punch" ? " is-active" : ""}`}
                      disabled={loading || workBusy || selectionCount === 0}
                      onClick={() => setInteractionMode("punch")}
                    >
                      Punch
                    </button>
                    <button
                      type="button"
                      className={`sidebar-collapsed-sub-btn${interactionMode === "selectExtrude" ? " is-active" : ""}`}
                      disabled={loading || workBusy || selectionCount === 0}
                      onClick={() => setInteractionMode("selectExtrude")}
                    >
                      Extrude
                    </button>
                    <div className="sidebar-collapsed-section-label">Method</div>
                    {(["stroke", "surface", "solid", "spray", "fill"] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        className={`sidebar-collapsed-sub-btn${selectionMethod === m ? " is-active" : ""}`}
                        disabled={loading || workBusy}
                        onClick={() => {
                          const s = selectionMethodToState(m);
                          setDrawStrokeMode(s.drawStrokeMode);
                          setStrokeDrawStyle(s.strokeDrawStyle);
                          setSprayDensity(s.sprayDensity);
                          setStrokeFamilyVariant(s.strokeFamilyVariant);
                        }}
                      >
                        {m[0].toUpperCase() + m.slice(1)}
                      </button>
                    ))}
                  </>
                )}

                {/* ── Sculpt sub-options ── */}
                {toolsPane === "sculpt" && (
                  <>
                    <div className="sidebar-collapsed-tool-separator" />
                    <div className="sidebar-collapsed-section-label">Mode</div>
                    {(
                      [
                        ["draw", "Draw"],
                        ["gouge", "Scrape"],
                        ["smooth", "Smooth"],
                        ["wall", "Wall"],
                        ["extrude", "Extrude"],
                        ["terrain", "Terrain"],
                      ] as const
                    ).map(([id, label]) => (
                      <button
                        key={id}
                        type="button"
                        className={`sidebar-collapsed-sub-btn${sculptStrokeMode === id ? " is-active" : ""}`}
                        disabled={loading || workBusy || interactionMode !== "sculpt"}
                        onClick={() => setSculptStrokeMode(id)}
                      >
                        {label}
                      </button>
                    ))}
                  </>
                )}

                {/* ── Generators sub-options ── */}
                {toolsPane === "generators" && (
                  <>
                    <div className="sidebar-collapsed-tool-separator" />
                    <div className="sidebar-collapsed-section-label">Kind</div>
                    {(
                      [
                        ["rocks", "Rocks"],
                        ["grass", "Grass"],
                        ["rope", "Rope"],
                        ["cloth", "Cloth"],
                        ["ashlar", "Ashlar"],
                        ["flora", "Flora"],
                        ["roof", "Roof"],
                      ] as const
                    ).map(([id, label]) => (
                      <button
                        key={id}
                        type="button"
                        className={`sidebar-collapsed-sub-btn${generatorKind === id ? " is-active" : ""}`}
                        disabled={loading || workBusy}
                        onClick={() => {
                          setGeneratorKind(id);
                          if (ropePhase.active) ropePhase.cancel();
                          else setRopeFirstScreen(null);
                          if (clothPhase.active) clothPhase.cancel();
                          else {
                            setClothPins([]);
                            clothPinsRef.current = [];
                          }
                          rocksPhase.cancel();
                          grassPhase.cancel();
                          ashlarPhase.cancel();
                          floraPhase.cancel();
                        }}
                      >
                        {label}
                      </button>
                    ))}
                  </>
                )}

                {/* ── Squishy sub-options ── */}
                {toolsPane === "squishy" && (
                  <>
                    <div className="sidebar-collapsed-tool-separator" />
                    <button
                      type="button"
                      className={`sidebar-collapsed-sub-btn${interactionMode === "squishy" ? " is-active" : ""}`}
                      disabled={loading || workBusy}
                      onClick={() => setInteractionMode("squishy")}
                    >
                      Metaballs
                    </button>
                  </>
                )}

                {/* ── Bone sub-options ── */}
                {toolsPane === "bone" && (
                  <>
                    <div className="sidebar-collapsed-tool-separator" />
                    <button
                      type="button"
                      className={`sidebar-collapsed-sub-btn${interactionMode === "bone" ? " is-active" : ""}`}
                      disabled={loading || workBusy}
                      onClick={() => setInteractionMode("bone")}
                    >
                      Armature
                    </button>
                  </>
                )}

                {/* ── Palette toggle ── */}
                <div className="sidebar-collapsed-tool-separator" />
                <button
                  type="button"
                  className="sidebar-collapsed-tool-btn"
                  onClick={() => setColorPaletteFloating(!colorPaletteFloating)}
                  disabled={loading || workBusy}
                  title="Toggle color palette"
                >
                  <span className="sidebar-collapsed-tool-icon">🎨</span>
                </button>
              </div>
            )}
          </div>
        </aside>
      ) : null}
    </>
  );
}
