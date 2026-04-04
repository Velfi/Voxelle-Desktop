import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow, LogicalPosition } from "@tauri-apps/api/window";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import { MascotView } from "./MascotView";
import { SpeechBubbleOverlay, type BubbleInfo } from "./SpeechBubbleOverlay";
import { CollabJoinProgressModal } from "./CollabJoinProgressModal";
import { JoinSessionModal } from "./JoinSessionModal";
import { PreferencesModal } from "./PreferencesModal";
import { StampBookModal } from "./StampBookModal";
import type { StampBookEntryTuple } from "./stampBookStorage";
import { loadRecentJoinUrls, rememberJoinedUrl } from "./joinRecent";
import {
  applyAppearanceToDocument,
  autosaveSettingsInvokeArgs,
  loadPreferences,
  normalizeCollabAccentColor,
  normalizeCollabDisplayName,
  normalizeCollabHostPort,
  preferencesWithCollabIdentity,
  savePreferences,
  toneMappingToGpuMode,
  type StartShape,
} from "./preferences";
import "./App.css";
import {
  deriveSelectionMethod,
  getStrokeDispatch,
  selectionMethodToState,
  strokeModeSkipsDrag,
  type DrawStrokeModeApi,
  type PlaneAxisApi,
  type StrokeDrawStyle,
  type StrokeFamilyVariant,
} from "./drawToolModel";
import { DrawPaneSelectionToolOptions } from "./toolOptions/DrawPaneSelectionToolOptions";
import { GeneratorToolOptions } from "./toolOptions/GeneratorToolOptions";
import { StatusBar } from "./StatusBar";
import { MATERIAL_BUILTIN_PALETTE_HEX } from "./materialBuiltinPalette";
import { ViewportCameraHud } from "./ViewportCameraHud";
import { useStrokePhase } from "./useStrokePhase";
import { SelectionGizmo, type SelectionGizmoRef } from "./SelectionGizmo";
import { ExtrudeGizmo, type ExtrudeGizmoRef } from "./ExtrudeGizmo";
import RadialPingMenu, { RADIAL_HOLD_MS } from "./RadialPingMenu";
import { PingArrowIndicator } from "./PingArrowIndicator";
import { ViewportSettingsSidebar } from "./ViewportSettingsSidebar";
import { generateIdea } from "./ideaGenerator";
import packageJson from "../package.json";
import type {
  CuboidPlaneGeo,
  DepthPhaseData,
  MoodState,
  PaintColorMode,
  FbmParams,
  GradientParams,
  DitherParams,
  PaintColorDistrib,
  ViewportCursorDebugPayload,
  ViewportCursorDebugScreen,
  RenderingMode,
  RosterEntry,
  LastSessionInfo,
  SceneObjectRow,
  ChatToast,
  InteractionMode,
  ToolsPane,
  SculptStrokeModeApi,
  TerrainSculptOpApi,
  GeneratorKindId,
  ClothGravityDirectionId,
  BrushShape,
  SculptBrushShapeUi,
  WallAreaShapeApi,
  SculptSmoothVariantApi,
  SprayDirectionApi,
} from "./types";
import { defaultMoodState, moodWith } from "./types";
import {
  MAX_GRID_SIZE,
  LS_RENDERING_MODE,
  LS_SIDEBAR_EXPANDED,
  LS_RIGHT_SIDEBAR_EXPANDED,
  LS_TOOLS_FLOATING,
  LS_TOOLS_FLOAT_POS,
  LS_PALETTE_FLOATING,
  LS_PALETTE_FLOAT_POS,
  LS_PALETTE_FLOAT_SIZE,
  LS_VIEWPORT_CURSOR_DEBUG,
  LS_PAINT_COLOR_DISTRIB,
  CHAT_TOAST_CAP,
  PING_HUD_MS,
  MATERIAL_OPTIONS,
  SCULPT_BRUSH_MAX_INDEX,
  loadPaintColorDistrib,
  layoutViewportCssSize,
  viewportCursorOverlayPercent,
  playPingSound,
  playSeagullSpeech,
  basename,
  userFacingUpdaterError,
  lastProjectReopenBlurb,
  sculptBrushShapeToRust,
} from "./constants";

/** App semver from `package.json` (status bar when no file is open). */
const VOXELLE_DESKTOP_VERSION = packageJson.version;

// (Multi-color paint distribution types, presets, and utility functions
// are now in ./types.ts, ./constants.ts, and ./generatorPresets.ts)

/** Avoid duplicate `load_start_screen_logo` in React Strict Mode (dev). */
let startScreenLogoInvokeSent = false;

/** Multi-color paint distribution settings panel (shown when ≥2 palette colors selected). */
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
function PaletteSwatches(props: {
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
    if (!isDraggingRef.current) {
      // Single click
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
    isDraggingRef.current = false;
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
        {selectedColors.length > 0 && (
          <div className="multi-color-hint">
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
          </div>
        )}
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

function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const gizmoRef = useRef<SelectionGizmoRef>(null);
  const extrudeGizmoRef = useRef<ExtrudeGizmoRef>(null);
  const gizmoHoverRef = useRef(false);
  /** Viewport render target in physical pixels (matches projection / picking); from Rust. */
  const viewportPhysRef = useRef({ w: 0, h: 0 });
  /** Swapchain drawable in physical pixels (authoritative native size; may differ from inner×dpr). */
  const surfacePhysRef = useRef({ w: 0, h: 0 });
  /** Last layout viewport CSS size (`layoutViewportCssSize`) — when these change, do not use stale surface for mapping until Rust syncs. */
  const lastLayoutViewportCssRef = useRef({ w: 0, h: 0 });
  const lastRef = useRef({ x: 0, y: 0 });
  /** Last pointer position over `.viewport` in physical pixels (for Z = ping pick). */
  const lastViewportPickNormRef = useRef<{ nx: number; ny: number } | null>(null);
  const pointerStartRef = useRef<{ x: number; y: number } | null>(null);
  const maxPointerMoveRef = useRef(0);
  /** After pick probe: camera orbit/pan/dolly vs voxel click-to-edit (matches web: no hit → camera). */
  const gestureRef = useRef<{
    pointerId: number;
    mode: "camera" | "voxel" | "squishyGizmo" | "selectionGizmo" | "extrudeGizmo";
  } | null>(null);
  const probingRef = useRef(false);
  /** Pointer-up event that arrived while a pick probe was in-flight; replayed after the probe resolves. */
  const pendingPointerUpRef = useRef<React.PointerEvent | null>(null);
  /** Stable ref to onPointerUp so it can be called from inside onPointerDown's async continuation. */
  const onPointerUpRef = useRef<((e: React.PointerEvent) => void) | null>(null);
  const activePointerIdRef = useRef<number | null>(null);
  /** Whether shift was held at the start of the current stroke (used to apply add combine mode). */
  const strokeShiftKeyRef = useRef(false);
  /** Pointer id currently captured by the viewport element (or null when not captured). */
  const capturedPointerIdRef = useRef<number | null>(null);
  const interactionModeRef = useRef<InteractionMode>("navigate");
  const activeColorRef = useRef(0x8899aa);
  const activeMaterialRef = useRef("plastic");
  const brushRadiusRef = useRef(0);
  const brushShapeRef = useRef<BrushShape>("sphere");
  const brushClipBottomHalfRef = useRef(false);
  const strokeDrawStyleRef = useRef<StrokeDrawStyle>("line");
  const drawStrokeModeRef = useRef<DrawStrokeModeApi>("line");
  const planeAxisRef = useRef<PlaneAxisApi>("auto");
  const sprayDensityRef = useRef(0);
  /** Normalized viewport start of stroke (for line stroke); matches Rust `viewport_texels_from_norm`. */
  const strokeViewportStartRef = useRef<{ nx: number; ny: number } | null>(null);
  /** Previous brush sample (normalized viewport). */
  const lastStrokeNormRef = useRef<{ nx: number; ny: number } | null>(null);
  const lastStrokeEditMsRef = useRef(0);
  const lastWallHoverMsRef = useRef(0);
  const dragDidEditRef = useRef(false);
  const loadingRef = useRef(false);
  const interactionBlockedRef = useRef(false);
  const pendingJoinUrlRef = useRef<string | null>(null);
  const collabActiveMenuRef = useRef(false);
  const startHostMenuRef = useRef<() => void>(() => {});
  const leaveSessionMenuRef = useRef<() => void>(() => {});
  const keysDownRef = useRef<Set<string>>(new Set());
  const flyRafRef = useRef<number>(0);
  /** True while fly mouse-look is active (pointer capture + Tauri grab / cursor warp for infinite look). */
  const flyMouseLookActiveRef = useRef(false);
  /** `pointerId` passed to `setPointerCapture` while mouselook is on; cleared on release. */
  const flyCapturedPointerIdRef = useRef<number | null>(null);
  /** Last client coords (CSS px) for fallback when movementX/Y are zero; never store viewport center unless the cursor is there. */
  const flyLastClientRef = useRef<{ x: number; y: number } | null>(null);
  /** Ignore one pointermove after programmatic cursor recenter (avoids treating the warp as a huge delta). */
  const flySkipNextFlyMoveRef = useRef(false);
  /** Physical-pixel look deltas coalesced per animation frame (pointermove IPC was starving RAF and inflating fly dt). */
  const flyPendingLookDxRef = useRef(0);
  const flyPendingLookDyRef = useRef(0);
  const [flySpeed, setFlySpeed] = useState<1 | 2 | 4>(1);
  const flySpeedRef = useRef<1 | 2 | 4>(1);
  flySpeedRef.current = flySpeed;
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("navigate");
  const [mood, setMood] = useState<MoodState>(() => defaultMoodState());
  const [selectionCount, setSelectionCount] = useState(0);
  const selectionCountRef = useRef(0);
  const [viewportCursorDebugEnabled, setViewportCursorDebugEnabled] = useState(() => {
    try {
      return localStorage.getItem(LS_VIEWPORT_CURSOR_DEBUG) === "1";
    } catch {
      return false;
    }
  });
  const [viewportCursorDebugJs, setViewportCursorDebugJs] = useState<{
    nx: number;
    ny: number;
  } | null>(null);
  const [viewportCursorDebugRust, setViewportCursorDebugRust] =
    useState<ViewportCursorDebugPayload | null>(null);
  const [viewportCursorDebugScreen, setViewportCursorDebugScreen] =
    useState<ViewportCursorDebugScreen | null>(null);
  /** Synchronous copy for debug ingest (React state can lag behind rAF). */
  const viewportCursorDebugScreenRef = useRef<ViewportCursorDebugScreen | null>(null);
  const viewportCursorDebugRafRef = useRef<number | null>(null);
  const [hideUI, setHideUI] = useState(false);
  const [matchMaterialSelectColor, setMatchMaterialSelectColor] = useState(false);
  const matchMaterialSelectColorRef = useRef(false);
  const [activeColor, setActiveColor] = useState(0x8899aa);
  /** Multi-color palette selection (empty = single-color mode). */
  const [selectedColors, setSelectedColors] = useState<number[]>([]);
  const selectedColorsRef = useRef<number[]>([]);
  const [paintColorDistrib, setPaintColorDistrib] =
    useState<PaintColorDistrib>(loadPaintColorDistrib);
  const paintColorDistribRef = useRef<PaintColorDistrib>(paintColorDistrib);
  /** Deterministic seed for the current stroke (randomSingle / preview consistency). */
  const currentStrokeSeedRef = useRef<number>(0);
  const [activeMaterial, setActiveMaterial] = useState("plastic");
  const [brushRadius, setBrushRadius] = useState(0);
  const [brushShape, setBrushShape] = useState<BrushShape>("sphere");
  /** Brush: clip to half-space along the face outward normal from the pick (see Rust `brush_clip_half_normal_from_screen`). */
  const [brushClipBottomHalf, setBrushClipBottomHalf] = useState(false);
  /** Mirror / symmetry axes for draw tools (bit 0 = X, bit 1 = Y, bit 2 = Z). */
  const [mirrorX, setMirrorX] = useState(false);
  const [mirrorY, setMirrorY] = useState(false);
  const [mirrorZ, setMirrorZ] = useState(false);
  const mirrorXRef = useRef(false);
  const mirrorYRef = useRef(false);
  const mirrorZRef = useRef(false);
  const [strokeDrawStyle, setStrokeDrawStyle] = useState<StrokeDrawStyle>("line");
  const [strokeFamilyVariant, setStrokeFamilyVariant] = useState<StrokeFamilyVariant>("stroke");
  const strokeFamilyVariantRef = useRef<StrokeFamilyVariant>("stroke");
  const [drawStrokeMode, setDrawStrokeMode] = useState<DrawStrokeModeApi>("line");
  const [planeAxis, setPlaneAxis] = useState<PlaneAxisApi>("auto");
  const [sprayDensity, setSprayDensity] = useState(0);
  /** Selection fill (web `fillSelectDiagonals` / `fillRespectsColor`). */
  const [fillSelectDiagonals, setFillSelectDiagonals] = useState(false);
  const [fillRespectsColor, setFillRespectsColor] = useState(true);
  type SelectionCombineModeApi = "replace" | "add" | "subtract" | "intersect";
  const [selectionCombineMode, setSelectionCombineMode] =
    useState<SelectionCombineModeApi>("replace");
  const fillSelectDiagonalsRef = useRef(false);
  const fillRespectsColorRef = useRef(true);
  const selectionStrokeBegunRef = useRef(false);
  const [toolsPane, setToolsPane] = useState<ToolsPane>("draw");
  const [generatorSphereRadius, setGeneratorSphereRadius] = useState(4);
  const [generatorKind, setGeneratorKind] = useState<GeneratorKindId>("rocks");
  const [squishyMode, setSquishyMode] = useState<"add" | "edit" | "delete">("add");
  const squishyModeRef = useRef<"add" | "edit" | "delete">("add");
  const [squishyHollow, setSquishyHollow] = useState(false);
  const [squishyWallThickness, setSquishyWallThickness] = useState(1);
  const [squishySnapToSurface, setSquishySnapToSurface] = useState(true);
  const [selectionStrokeSnapToSurface, setSelectionStrokeSnapToSurface] = useState(true);
  const [selectionStrokeAxisAlign, setSelectionStrokeAxisAlign] = useState(true);
  const selectionStrokeSnapToSurfaceRef = useRef(true);
  const selectionStrokeAxisAlignRef = useRef(true);
  const [surfacePlaneHollow, setSurfacePlaneHollow] = useState(false);
  const surfacePlaneHollowRef = useRef(false);
  const [sprayConstrainToPlane, setSprayConstrainToPlane] = useState(false);
  const sprayConstrainToPlaneRef = useRef(false);
  const [spraySizeRange, setSpraySizeRange] = useState(false);
  const spraySizeRangeRef = useRef(false);
  /** Scatter: random stamp offset in voxels (web `sprayScatter`; 0 = no scatter). */
  const [sprayScatter, setSprayScatter] = useState(0);
  const sprayScatterRef = useRef(0);
  const [sprayRadiusMin, setSprayRadiusMin] = useState(0);
  const sprayRadiusMinRef = useRef(0);
  const [sprayRadiusMax, setSprayRadiusMax] = useState(4);
  const sprayRadiusMaxRef = useRef(4);
  /** Separate brush shape for spray mode (web `sprayBrushShape`). */
  const [sprayBrushShape, setSprayBrushShape] = useState<BrushShape>("sphere");
  const sprayBrushShapeRef = useRef<BrushShape>("sphere");
  /** Plane reference for constrain-to-plane: auto | camera | x | y | z. */
  type ConstrainToPlaneRef = "auto" | "camera" | "x" | "y" | "z";
  const [sprayConstrainToPlaneRef_, setSprayConstrainToPlaneRef_] =
    useState<ConstrainToPlaneRef>("auto");
  const sprayConstrainToPlaneRefRef = useRef<ConstrainToPlaneRef>("auto");
  const [fillConstrainToPlane, setFillConstrainToPlane] = useState(false);
  const fillConstrainToPlaneRef = useRef(false);
  const [squishyBallCount, setSquishyBallCount] = useState(0);
  const [strokePolygonVerts, setStrokePolygonVerts] = useState<[number, number, number][]>([]);
  /** Kept in sync with `strokePolygonVerts` for `sync_preview_input` / `mergedStrokeAux` (no stale closure). */
  const strokePolygonVertsRef = useRef<[number, number, number][]>([]);
  const strokeClickRef = useRef<{
    circleCenter: [number, number, number] | null;
  }>({
    circleCenter: null,
  });
  /** Solid cuboid depth phase (web parity): plane drag done; adjust depth then Done. */
  const cuboidPhase = useStrokePhase<DepthPhaseData>({
    phases: ["depth"],
    onCancel: () => {
      cuboidDepthRef.current = 1;
      setCuboidDepthUi(1);
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: () => void commitCuboidSolidAtScreen(),
  });
  const [cuboidDepthUi, setCuboidDepthUi] = useState(1);
  const cuboidDepthRef = useRef(1);
  /** Solid cylinder: disk drag done; adjust depth then Done (same flow as cuboid). */
  const cylinderPhase = useStrokePhase<DepthPhaseData>({
    phases: ["depth"],
    onCancel: () => {
      cylinderDepthRef.current = 1;
      setCylinderDepthUi(1);
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: () => void commitCylinderSolidAtScreen(),
  });
  const [cylinderDepthUi, setCylinderDepthUi] = useState(1);
  const cylinderDepthRef = useRef(1);
  /** Solid polygon: vertices placed; adjust depth then Done (same flow as cuboid/cylinder). */
  const polygonPhase = useStrokePhase<{ endNorm: { nx: number; ny: number } }>({
    phases: ["depth"],
    onCancel: () => {
      polygonDepthRef.current = 1;
      setPolygonDepthUi(1);
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: () => void commitPolygonSolid(),
  });
  const [polygonDepthUi, setPolygonDepthUi] = useState(1);
  const polygonDepthRef = useRef(1);
  /** Extrude phased tool: drag creates preview, adjust settings, then commit. */
  const extrudePhase = useStrokePhase<Record<string, never>>({
    phases: ["settings"],
    onCancel: () => {
      extrudeStartNormRef.current = null;
      void invoke("voxel_stroke_end").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: () => {
      extrudeStartNormRef.current = null;
      void invoke("voxel_stroke_end").catch(() => {});
    },
  });
  /** Stored normalized viewport start position for ray-based extrude (persists across re-drags). */
  const extrudeStartNormRef = useRef<{ nx: number; ny: number } | null>(null);
  /** True while the user is re-dragging the extrude endpoint during the settings phase. */
  const extrudeRedragRef = useRef(false);
  /** Inline depth field: draft while focused; +/- still updates value + draft when editing. */
  const [extrusionDepthEditing, setExtrusionDepthEditing] = useState(false);
  const [extrusionDepthDraft, setExtrusionDepthDraft] = useState("");
  const strokePolygonLastScreenRef = useRef<{ nx: number; ny: number } | null>(null);
  const [ropeFirstScreen, setRopeFirstScreen] = useState<{
    nx: number;
    ny: number;
  } | null>(null);
  /** Rope phased tool: click two points → adjust tension/sag → Done/Cancel. */
  const ropePhase = useStrokePhase<{
    nx1: number;
    ny1: number;
    nx2: number;
    ny2: number;
  }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      setRopeFirstScreen(null);
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx1, ny1, nx2, ny2 } = snap.data;
      void invoke("generator_rope_at_screen", {
        args: {
          nx1,
          ny1,
          nx2,
          ny2,
          tension: ropeTensionRef.current,
          gravityDirection: clothGravityDirectionRef.current,
          brushRadius: ropeBrushRadiusIndexRef.current,
          brushShape: sculptBrushShapeToRust(ropeBrushShapeUiRef.current),
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      }).catch(() => {});
      setRopeFirstScreen(null);
    },
  });
  /** Cloth phased tool: click 3+ pins → settings overlay → Done/Cancel. */
  const clothPhase = useStrokePhase<Record<string, never>>({
    phases: ["settings"],
    onCancel: () => {
      setClothPins([]);
      clothPinsRef.current = [];
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: () => {
      const pins = clothPinsRef.current;
      if (pins.length < 3) return;
      void invoke("generator_cloth_from_pins_cmd", {
        args: {
          pins: pins.map((p) => [p[0], p[1], p[2]]),
          tension: clothTensionRef.current,
          gravityDirection: clothGravityDirectionRef.current,
          brushRadius: ropeBrushRadiusIndexRef.current,
          brushShape: sculptBrushShapeToRust(ropeBrushShapeUiRef.current),
          color: activeColorRef.current,
          material: activeMaterialRef.current,
          gravityScale: clothSimGravityPctRef.current / 100,
          stiffnessScale: clothSimStiffnessPctRef.current / 100,
          clothIterations: clothSimIterationsRef.current,
          clothConstraintPasses: clothSimConstraintPassesRef.current,
        },
      }).catch(() => {});
      setClothPins([]);
      clothPinsRef.current = [];
    },
  });
  const [ropeSag, _setRopeSag] = useState(2.5);
  /** 0 = loose, 1 = taut (web ropeTension). */
  const [ropeTension, setRopeTension] = useState(0.5);
  const [ropeBrushRadiusIndex, setRopeBrushRadiusIndex] = useState(2);
  const [ropeBrushShapeUi, setRopeBrushShapeUi] = useState<"sphere" | "cube">("sphere");
  /** Cloth: corner pins (surface picks), then Apply in tool options. */
  const [clothPins, setClothPins] = useState<[number, number, number][]>([]);
  const clothPinsRef = useRef<[number, number, number][]>([]);
  const [clothTension, setClothTension] = useState(0.5);
  const [clothGravityDirection, setClothGravityDirection] =
    useState<ClothGravityDirectionId>("down");
  const [clothSimGravityPct, setClothSimGravityPct] = useState(100);
  const [clothSimStiffnessPct, setClothSimStiffnessPct] = useState(100);
  const [clothSimIterations, setClothSimIterations] = useState(0);
  const [clothSimConstraintPasses, setClothSimConstraintPasses] = useState(2);
  /** Single-click generators: click → settings overlay → Done/Cancel. */
  const rocksPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_rocks_at_screen", {
        args: {
          nx,
          ny,
          seed,
          size: Math.max(1, generatorSphereRadiusRef.current),
          roughness: rockRoughnessRef.current,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
          count: rockCountRef.current,
          clusterRadius: rockClusterRadiusRef.current,
          sinkDirection:
            rockSinkDirectionRef.current === "under"
              ? -1
              : rockSinkDirectionRef.current === "over"
                ? 1
                : 0,
          sinkAmount: rockSinkAmountRef.current,
        },
      }).catch(() => {});
      rockPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });
  const grassPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_grass_at_screen", {
        args: {
          nx,
          ny,
          seed,
          radius: Math.max(1, generatorSphereRadiusRef.current),
          density: grassDensityRef.current,
          maxHeight: grassMaxHeightRef.current,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      }).catch(() => {});
      grassPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });
  const ashlarPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_ashlar_at_screen", {
        args: {
          nx,
          ny,
          seed,
          size: Math.max(1, generatorSphereRadiusRef.current),
          roughness: rockRoughnessRef.current,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
          thickness: ashlarThicknessRef.current,
        },
      }).catch((err: unknown) => {
        console.error("[ashlar] placement failed:", err);
      });
      ashlarPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });
  const floraPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_flora_at_screen", {
        args: {
          nx,
          ny,
          seed,
          height: floraHeight,
          girth: floraGirth,
          wobble: floraWobble,
          taper: floraTaper,
          stemCount: floraStemCount,
          clusterRadius: floraClusterRadius,
          branchCount: floraBranchCount,
          branchDepth: floraBranchDepth,
          branchStart: floraBranchStart,
          branchSpread: floraBranchSpread,
          braidStrands: floraBraidStrands,
          braidTwist: floraBraidTwist,
          canopy: floraCanopy,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      }).catch(() => {});
      floraPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });
  const piscinaPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_piscina_at_screen", {
        args: {
          nx,
          ny,
          seed,
          species: piscinaSpecies,
          length: piscinaLength,
          widthParam: piscinaWidth,
          thickness: piscinaThickness,
          spineBend: piscinaSpineBend,
          spineSCurve: piscinaSpineSCurve,
          finDorsal: piscinaFinDorsal,
          finAnal: piscinaFinAnal,
          finCaudal: piscinaFinCaudal,
          finPectoral: piscinaFinPectoral,
          finPelvic: piscinaFinPelvic,
          finAdipose: piscinaFinAdipose,
          showFinDorsal: piscinaShowFinDorsal,
          showFinAnal: piscinaShowFinAnal,
          showFinCaudal: piscinaShowFinCaudal,
          showFinPectoral: piscinaShowFinPectoral,
          showFinPelvic: piscinaShowFinPelvic,
          showFinAdipose: piscinaShowFinAdipose,
          anchorOffsetU: piscinaAnchorU,
          anchorOffsetV: piscinaAnchorV,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      }).catch(() => {});
      piscinaPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });
  const insectaPhase = useStrokePhase<{ nx: number; ny: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny } = snap.data;
      void invoke("generator_insecta_at_screen", {
        args: {
          nx,
          ny,
          species: insectaSpecies,
          totalLength: insectaTotalLength,
          headRatio: insectaHeadRatio,
          thoraxRatio: insectaThoraxRatio,
          abdomenRatio: insectaAbdomenRatio,
          bodyHalfWidth: insectaBodyHalfWidth,
          bodyHalfHeight: insectaBodyHalfHeight,
          abdomenTaper: insectaAbdomenTaper,
          headShape: insectaHeadShape,
          anchorOffsetU: insectaAnchorU,
          anchorOffsetV: insectaAnchorV,
          bodyYaw: insectaBodyYawDeg * (Math.PI / 180),
          bodyArch: insectaBodyArch,
          antennaLength: insectaAntennaLength,
          antennaSpread: insectaAntennaSpread,
          antennaPitch: insectaAntennaPitch,
          antennaRoot: insectaAntennaRoot,
          mandibleLength: insectaMandibleLength,
          mandibleSpread: insectaMandibleSpread,
          mandibleForward: insectaMandibleForward,
          wingShape: insectaWingShape,
          showWingFore: insectaShowWingFore,
          wingForeLength: insectaWingForeLength,
          wingForeWidth: insectaWingForeWidth,
          wingForeSpread: insectaWingForeSpread,
          wingForePitch: insectaWingForePitch,
          wingForeOffset: insectaWingForeOffset,
          wingForeForwardCant: insectaWingForeForwardCant,
          showWingHind: insectaShowWingHind,
          wingHindLength: insectaWingHindLength,
          wingHindWidth: insectaWingHindWidth,
          wingHindSpread: insectaWingHindSpread,
          wingHindPitch: insectaWingHindPitch,
          wingHindOffset: insectaWingHindOffset,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      }).catch(() => {});
    },
  });
  const faunaPhase = useStrokePhase<{ nx: number; ny: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny } = snap.data;
      void invoke("generator_fauna_at_screen", {
        args: {
          nx,
          ny,
          stance: faunaStance,
          archetype: faunaArchetype,
          anchorOffsetU: faunaAnchorU,
          anchorOffsetV: faunaAnchorV,
          bodyYaw: faunaBodyYawDeg * (Math.PI / 180),
          bodyArch: faunaBodyArch,
          spineSegments: faunaSpineSegments,
          bodyLength: faunaBodyLength,
          bodyHalfWidth: faunaBodyHalfWidth,
          bodyHalfHeight: faunaBodyHalfHeight,
          neckLength: faunaNeckLength,
          neckHalfWidth: faunaNeckHalfWidth,
          neckHalfHeight: faunaNeckHalfHeight,
          headLength: faunaHeadLength,
          headHalfWidth: faunaHeadHalfWidth,
          headHalfHeight: faunaHeadHalfHeight,
          tailLength: faunaTailLength,
          shoulderOffsetForward: faunaShoulderOffsetForward,
          hipOffsetForward: faunaHipOffsetForward,
          frontUpperLength: faunaFrontUpperLength,
          frontLowerLength: faunaFrontLowerLength,
          hindUpperLength: faunaHindUpperLength,
          hindLowerLength: faunaHindLowerLength,
          limbTargets: [
            [20, -2.1, -19],
            [20, 2.1, -19],
            [-3.5, -2.2, -20],
            [-3.5, 2.2, -20],
          ],
          limbPoles: [
            [20, -2.4, 0.6],
            [20, 2.4, 0.6],
            [1.8, -2.8, 1.2],
            [1.8, 2.8, 1.2],
          ],
          spinePoseChest: [0, 0, 0],
          spinePoseNeck: [0, 0, 0],
          spinePoseHead: [0, 0, 0],
          autoFootPlacement: faunaAutoFootPlacement,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      }).catch(() => {});
    },
  });
  /** Squishy (metaball) session phase: Enter commits, Escape cancels. */
  const squishyPhase = useStrokePhase<Record<string, never>>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("squishy_session_clear")
        .then(() => setSquishyBallCount(0))
        .catch(() => {});
    },
    onCommit: () => {
      void invoke("squishy_session_commit", {
        args: {
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      })
        .then(() => invoke("squishy_session_clear"))
        .then(() => setSquishyBallCount(0))
        .catch(() => {});
    },
  });
  const [rockRoughness, setRockRoughness] = useState(0.4);
  const [ashlarThickness, setAshlarThickness] = useState(3);
  const [rockCount, setRockCount] = useState(1);
  const [rockClusterRadius, setRockClusterRadius] = useState(1);
  const [rockSinkDirection, setRockSinkDirection] = useState<"none" | "under" | "over">("none");
  const [rockSinkAmount, setRockSinkAmount] = useState(0);
  const [grassDensity, setGrassDensity] = useState(0.6);
  const [grassMaxHeight, setGrassMaxHeight] = useState(3);
  // Flora params
  const [floraPreset, setFloraPreset] = useState<string>("stalk");
  const [floraHeight, setFloraHeight] = useState(14);
  const [floraGirth, setFloraGirth] = useState(0);
  const [floraWobble, setFloraWobble] = useState(0.12);
  const [floraTaper, setFloraTaper] = useState(0.12);
  const [floraStemCount, setFloraStemCount] = useState(1);
  const [floraClusterRadius, setFloraClusterRadius] = useState(0);
  const [floraBranchCount, setFloraBranchCount] = useState(0);
  const [floraBranchDepth, setFloraBranchDepth] = useState(1);
  const [floraBranchStart, setFloraBranchStart] = useState(0.5);
  const [floraBranchSpread, setFloraBranchSpread] = useState(1.0);
  const [floraBraidStrands, setFloraBraidStrands] = useState(1);
  const [floraBraidTwist, setFloraBraidTwist] = useState(0.35);
  const [floraCanopy, setFloraCanopy] = useState(0.18);
  // Piscina params
  const [piscinaSpecies, setPiscinaSpecies] = useState<string>("trout");
  const [piscinaLength, setPiscinaLength] = useState(16);
  const [piscinaWidth, setPiscinaWidth] = useState(4);
  const [piscinaThickness, setPiscinaThickness] = useState(3);
  const [piscinaSpineBend, setPiscinaSpineBend] = useState(0);
  const [piscinaSpineSCurve, setPiscinaSpineSCurve] = useState(0);
  const [piscinaShowFinDorsal, setPiscinaShowFinDorsal] = useState(true);
  const [piscinaFinDorsal, setPiscinaFinDorsal] = useState(3);
  const [piscinaShowFinAnal, setPiscinaShowFinAnal] = useState(true);
  const [piscinaFinAnal, setPiscinaFinAnal] = useState(3);
  const [piscinaShowFinCaudal, setPiscinaShowFinCaudal] = useState(true);
  const [piscinaFinCaudal, setPiscinaFinCaudal] = useState(3);
  const [piscinaShowFinPectoral, setPiscinaShowFinPectoral] = useState(true);
  const [piscinaFinPectoral, setPiscinaFinPectoral] = useState(3);
  const [piscinaShowFinPelvic, setPiscinaShowFinPelvic] = useState(true);
  const [piscinaFinPelvic, setPiscinaFinPelvic] = useState(3);
  const [piscinaShowFinAdipose, setPiscinaShowFinAdipose] = useState(true);
  const [piscinaFinAdipose, setPiscinaFinAdipose] = useState(3);
  const [piscinaAnchorU, setPiscinaAnchorU] = useState(0);
  const [piscinaAnchorV, setPiscinaAnchorV] = useState(0);
  // Insecta params
  const [insectaSpecies, setInsectaSpecies] = useState<string>("bee");
  const [insectaTotalLength, setInsectaTotalLength] = useState(24);
  const [insectaHeadRatio, setInsectaHeadRatio] = useState(1.0);
  const [insectaThoraxRatio, setInsectaThoraxRatio] = useState(1.2);
  const [insectaAbdomenRatio, setInsectaAbdomenRatio] = useState(2.0);
  const [insectaBodyHalfWidth, setInsectaBodyHalfWidth] = useState(3);
  const [insectaBodyHalfHeight, setInsectaBodyHalfHeight] = useState(3);
  const [insectaAbdomenTaper, setInsectaAbdomenTaper] = useState(0.6);
  const [insectaHeadShape, setInsectaHeadShape] = useState(60);
  const [insectaBodyYawDeg, setInsectaBodyYawDeg] = useState(0);
  const [insectaBodyArch, setInsectaBodyArch] = useState(0);
  const [insectaAnchorU, setInsectaAnchorU] = useState(0);
  const [insectaAnchorV, setInsectaAnchorV] = useState(0);
  const [insectaAntennaLength, setInsectaAntennaLength] = useState(6);
  const [insectaAntennaSpread, setInsectaAntennaSpread] = useState(20);
  const [insectaAntennaPitch, setInsectaAntennaPitch] = useState(30);
  const [insectaAntennaRoot, setInsectaAntennaRoot] = useState(0);
  const [insectaMandibleLength, setInsectaMandibleLength] = useState(0);
  const [insectaMandibleSpread, setInsectaMandibleSpread] = useState(0);
  const [insectaMandibleForward, setInsectaMandibleForward] = useState(0);
  const [insectaWingShape, setInsectaWingShape] = useState(85);
  const [insectaShowWingFore, setInsectaShowWingFore] = useState(true);
  const [insectaWingForeLength, setInsectaWingForeLength] = useState(12);
  const [insectaWingForeWidth, setInsectaWingForeWidth] = useState(3);
  const [insectaWingForeSpread, setInsectaWingForeSpread] = useState(15);
  const [insectaWingForePitch, setInsectaWingForePitch] = useState(0);
  const [insectaWingForeOffset, setInsectaWingForeOffset] = useState(0);
  const [insectaWingForeForwardCant, setInsectaWingForeForwardCant] = useState(0);
  const [insectaShowWingHind, setInsectaShowWingHind] = useState(false);
  const [insectaWingHindLength, setInsectaWingHindLength] = useState(8);
  const [insectaWingHindWidth, setInsectaWingHindWidth] = useState(2);
  const [insectaWingHindSpread, setInsectaWingHindSpread] = useState(15);
  const [insectaWingHindPitch, setInsectaWingHindPitch] = useState(0);
  const [insectaWingHindOffset, setInsectaWingHindOffset] = useState(0);
  // Fauna params
  const [faunaStance, setFaunaStance] = useState<string>("quadruped");
  const [faunaArchetype, setFaunaArchetype] = useState<string>("ungulate");
  const [faunaBodyYawDeg, setFaunaBodyYawDeg] = useState(0);
  const [faunaBodyArch, setFaunaBodyArch] = useState(0.02);
  const [faunaSpineSegments, setFaunaSpineSegments] = useState(7);
  const [faunaBodyLength, setFaunaBodyLength] = useState(17);
  const [faunaBodyHalfWidth, setFaunaBodyHalfWidth] = useState(2);
  const [faunaBodyHalfHeight, setFaunaBodyHalfHeight] = useState(3);
  const [faunaNeckLength, setFaunaNeckLength] = useState(8);
  const [faunaNeckHalfWidth, setFaunaNeckHalfWidth] = useState(2);
  const [faunaNeckHalfHeight, setFaunaNeckHalfHeight] = useState(3);
  const [faunaHeadLength, setFaunaHeadLength] = useState(6);
  const [faunaHeadHalfWidth, setFaunaHeadHalfWidth] = useState(2);
  const [faunaHeadHalfHeight, setFaunaHeadHalfHeight] = useState(3);
  const [faunaTailLength, setFaunaTailLength] = useState(1);
  const [faunaShoulderOffsetForward, setFaunaShoulderOffsetForward] = useState(3);
  const [faunaHipOffsetForward, setFaunaHipOffsetForward] = useState(-3);
  const [faunaFrontUpperLength, setFaunaFrontUpperLength] = useState(7);
  const [faunaFrontLowerLength, setFaunaFrontLowerLength] = useState(7);
  const [faunaHindUpperLength, setFaunaHindUpperLength] = useState(8);
  const [faunaHindLowerLength, setFaunaHindLowerLength] = useState(8);
  const [faunaAnchorU, setFaunaAnchorU] = useState(0);
  const [faunaAnchorV, setFaunaAnchorV] = useState(0);
  const [faunaAutoFootPlacement, setFaunaAutoFootPlacement] = useState(false);
  // Roof params
  const [roofStyle, setRoofStyle] = useState<string>("gable");
  const [roofHeight, setRoofHeight] = useState(6);
  const [roofHollow, setRoofHollow] = useState(false);
  const [roofPins, setRoofPins] = useState<[number, number, number][]>([]);
  const roofPinsRef = useRef<[number, number, number][]>([]);
  const [roofAreaShape, setRoofAreaShape] = useState<"polygon" | "square" | "circle">("polygon");
  const roofAreaShapeRef = useRef<"polygon" | "square" | "circle">("polygon");
  // First click anchor for square/circle roof modes
  const [roofFirstClick, setRoofFirstClick] = useState<[number, number, number] | null>(null);
  const roofFirstClickRef = useRef<[number, number, number] | null>(null);
  const [sculptStrokeMode, setSculptStrokeMode] = useState<SculptStrokeModeApi>("draw");
  const [terrainSculptOp, setTerrainSculptOp] = useState<TerrainSculptOpApi>("raise");
  const [terrainBaseY, setTerrainBaseY] = useState(0);
  const [terrainStrength] = useState(4);
  const [terrainSmoothRadius, setTerrainSmoothRadius] = useState(2);
  const [terrainFlattenUseBaseY, setTerrainFlattenUseBaseY] = useState(false);
  const [terrainSubVoxel, setTerrainSubVoxel] = useState(false);
  const [terrainHoverY, setTerrainHoverY] = useState<number | null>(null);
  const [sculptSmoothPasses, setSculptSmoothPasses] = useState(1);
  /** Web `sculptBrushRadius` index (display = index + 1 voxel span). */
  const [sculptBrushRadius, setSculptBrushRadius] = useState(2);
  const [sculptBrushStrength, setSculptBrushStrength] = useState(100);
  const [sculptBrushFalloff, setSculptBrushFalloff] = useState(0);
  const [sculptBrushShapeUi, setSculptBrushShapeUi] = useState<SculptBrushShapeUi>("circle");
  // Extrude-specific params
  const [extrudeDirectionRef, setExtrudeDirectionRef] = useState<
    "camera" | "auto" | "x" | "y" | "z"
  >("camera");
  const extrudeDirectionRefRef = useRef<"camera" | "auto" | "x" | "y" | "z">("camera");
  extrudeDirectionRefRef.current = extrudeDirectionRef;
  const [extrudeProfile, setExtrudeProfile] = useState<"cube" | "cylinder">("cube");
  const [extrudeEndCap, setExtrudeEndCap] = useState<"flat" | "rounded" | "pointed">("flat");
  const [extrudeTaper, setExtrudeTaper] = useState(false);
  const [extrudeTaperStart, setExtrudeTaperStart] = useState(3);
  const [extrudeTaperEnd, setExtrudeTaperEnd] = useState(0);
  const [wallAreaShape, setWallAreaShape] = useState<WallAreaShapeApi>("brush");
  const [sprayDirection, setSprayDirection] = useState<SprayDirectionApi>("auto");
  const [wallWidthIndex, setWallWidthIndex] = useState(0);
  const [wallHeightVox, setWallHeightVox] = useState(2);
  const [wallLockStartHeight, setWallLockStartHeight] = useState(false);
  const [wallAxisAlign, setWallAxisAlign] = useState(false);
  const [sculptSmoothVariant, setSculptSmoothVariant] =
    useState<SculptSmoothVariantApi>("majority");
  /** Web `smoothNeighborRadius` 0–6 (inclusive). */
  const [smoothNeighborRadius, setSmoothNeighborRadius] = useState(0);
  const [smoothAggressiveness, setSmoothAggressiveness] = useState(100);
  const [smoothLaplacianIterations, setSmoothLaplacianIterations] = useState(4);
  const [smoothLaplacianRelaxPct, setSmoothLaplacianRelaxPct] = useState(50);
  /** Wall + polygon area: corner voxels (object-local), then Done commits a wall stroke. */
  const [wallSculptPolygonVerts, setWallSculptPolygonVerts] = useState<[number, number, number][]>(
    [],
  );
  const [pathLabel, setPathLabel] = useState("");
  /** Mascots loaded for the start screen. Set to true once mascot_load commands have fired. */
  const [mascotsLoaded, setMascotsLoaded] = useState(false);
  const MASCOT_W = 180;
  const MASCOT_H = 180;
  const MASCOT_PAD = 24;
  const [mascotRect, setMascotRect] = useState(() => ({
    x: MASCOT_PAD,
    y: window.innerHeight - MASCOT_H - MASCOT_PAD,
    width: MASCOT_W,
    height: MASCOT_H,
  }));
  /** Active GPU-rendered speech bubbles (click-capture overlays). */
  const [speechBubbles, setSpeechBubbles] = useState<BubbleInfo[]>([]);
  const nextBubbleId = useRef(0);

  /** Cold-start title mesh from `Logo.voxelle`; enables bottom menu layout and viewport orbit. */
  const [startScreenLogoLoaded, setStartScreenLogoLoaded] = useState(false);
  const startScreenLogoLoadedRef = useRef(false);
  const [logoLightControlsVisible, setLogoLightControlsVisible] = useState(false);
  const [logoLightAzimuth, setLogoLightAzimuth] = useState(0);
  const [logoLightElevation, setLogoLightElevation] = useState(30);
  const [logoCamAzimuth, setLogoCamAzimuth] = useState(62);
  const [logoCamElevation, setLogoCamElevation] = useState(12);
  const [logoCamDist, setLogoCamDist] = useState(2.2);
  const [loadError, setLoadError] = useState<string | null>(null);
  /** Session ended (leave, lost connection, or kicked); cleared on dismiss or new load/join. */
  const [collabBanner, setCollabBanner] = useState<{
    text: string;
    tone: "info" | "alert";
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  /** Short label from the load pipeline (e.g. mesh phase); empty when idle. */
  const [loadPhase, setLoadPhase] = useState("");
  /** Save / heavy mesh / undo-redo (Rust `voxelle-work-progress`). */
  const [workBusy, setWorkBusy] = useState(false);
  const [workProgress, setWorkProgress] = useState(0);
  const [workPhase, setWorkPhase] = useState("");
  const workPhaseRef = useRef("");
  workPhaseRef.current = workPhase;
  /** True from flood-fill invoke start until it settles; mesh phases say "Applying edit…" not "Fill", so HUD can't rely on phase text alone. */
  const [fillOperationPending, setFillOperationPending] = useState(false);
  const fillOperationPendingRef = useRef(false);
  /** Pending large-fill confirmation: stores the callback pair to resolve when user picks. */
  const [pendingFillConfirm, setPendingFillConfirm] = useState<{
    resolve: (confirmed: boolean) => void;
  } | null>(null);
  const [fpsDisplayed, setFpsDisplayed] = useState(0);
  const [showFpsCounter, setShowFpsCounter] = useState(() => loadPreferences().showFpsCounter);
  const [pingMs, setPingMs] = useState<number | null>(null);
  const [showPingLatency, setShowPingLatency] = useState(() => loadPreferences().showPingLatency);
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [newGridSize, setNewGridSize] = useState(() => loadPreferences().newProjectDefaultSize);
  const [newGridShape, setNewGridShape] = useState<StartShape>(
    () => loadPreferences().newProjectDefaultShape,
  );
  const [rotateDialogOpen, setRotateDialogOpen] = useState(false);
  const [rotateDialogAxis, setRotateDialogAxis] = useState<0 | 1 | 2>(1);
  const [rotateDialogDegrees, setRotateDialogDegrees] = useState(90);
  const [scaleDialogOpen, setScaleDialogOpen] = useState(false);
  const [scaleDialogFactor, setScaleDialogFactor] = useState(2);
  const [sidebarExpanded, setSidebarExpanded] = useState(() => {
    if (typeof localStorage === "undefined") return true;
    const stored = localStorage.getItem(LS_SIDEBAR_EXPANDED);
    return stored === null ? true : stored === "1";
  });
  const [rightSidebarExpanded, setRightSidebarExpanded] = useState(() => {
    if (typeof localStorage === "undefined") return true;
    const stored = localStorage.getItem(LS_RIGHT_SIDEBAR_EXPANDED);
    return stored === null ? true : stored === "1";
  });
  const [toolsPaneFloating, setToolsPaneFloating] = useState(() => {
    if (typeof localStorage === "undefined") return false;
    return localStorage.getItem(LS_TOOLS_FLOATING) === "1";
  });
  const [toolPanePos, setToolPanePos] = useState(() => {
    if (typeof localStorage === "undefined") return { x: 16, y: 56 };
    try {
      const s = localStorage.getItem(LS_TOOLS_FLOAT_POS);
      if (s) {
        const j = JSON.parse(s) as { x?: unknown; y?: unknown };
        if (typeof j.x === "number" && typeof j.y === "number") {
          return { x: j.x, y: j.y };
        }
      }
    } catch {
      /* ignore */
    }
    return { x: 16, y: 56 };
  });
  const [colorPaletteFloating, setColorPaletteFloating] = useState(() => {
    if (typeof localStorage === "undefined") return false;
    return localStorage.getItem(LS_PALETTE_FLOATING) === "1";
  });
  const [colorPalettePos, setColorPalettePos] = useState(() => {
    if (typeof localStorage === "undefined") return { x: 220, y: 56 };
    try {
      const s = localStorage.getItem(LS_PALETTE_FLOAT_POS);
      if (s) {
        const j = JSON.parse(s) as { x?: unknown; y?: unknown };
        if (typeof j.x === "number" && typeof j.y === "number") {
          return { x: j.x, y: j.y };
        }
      }
    } catch {
      /* ignore */
    }
    return { x: 220, y: 56 };
  });
  const toolPaneDragRef = useRef<{
    pid: number;
    startX: number;
    startY: number;
    origX: number;
    origY: number;
  } | null>(null);
  const toolPanePosRef = useRef(toolPanePos);
  toolPanePosRef.current = toolPanePos;

  const [colorPaletteSize, setColorPaletteSize] = useState(() => {
    if (typeof localStorage === "undefined") return { w: 200, h: 260 };
    try {
      const s = localStorage.getItem(LS_PALETTE_FLOAT_SIZE);
      if (s) {
        const j = JSON.parse(s) as { w?: unknown; h?: unknown };
        if (typeof j.w === "number" && typeof j.h === "number") {
          return { w: j.w, h: j.h };
        }
      }
    } catch {
      /* ignore */
    }
    return { w: 200, h: 260 };
  });
  const colorPalettePosRef = useRef(colorPalettePos);
  colorPalettePosRef.current = colorPalettePos;

  const [stampBookOpen, setStampBookOpen] = useState(false);
  /** True when a stamp was loaded from the stamp book (not from the edit selection). */
  const [stampBookPatternActive, setStampBookPatternActive] = useState(false);
  const [stampRotX, setStampRotX] = useState(0);
  const [stampRotY, setStampRotY] = useState(0);
  const [stampRotZ, setStampRotZ] = useState(0);
  const stampRotXRef = useRef(0);
  const stampRotYRef = useRef(0);
  const stampRotZRef = useRef(0);
  stampRotXRef.current = stampRotX;
  stampRotYRef.current = stampRotY;
  stampRotZRef.current = stampRotZ;
  /** Stamp placement origin X: 0 = min edge, 1 = center, 2 = max edge. */
  const [stampOriginX, setStampOriginX] = useState(0);
  /** Stamp placement origin Z: 0 = min edge, 1 = center, 2 = max edge. */
  const [stampOriginZ, setStampOriginZ] = useState(0);
  const stampOriginXRef = useRef(0);
  const stampOriginZRef = useRef(0);
  stampOriginXRef.current = stampOriginX;
  stampOriginZRef.current = stampOriginZ;

  const [joinModalOpen, setJoinModalOpen] = useState(false);
  const [leaveConfirmOpen, setLeaveConfirmOpen] = useState(false);
  const [collabJoinPending, setCollabJoinPending] = useState(false);
  const [hostWsUrl, setHostWsUrl] = useState<string | null>(null);
  const [joinUrl, setJoinUrl] = useState(() => {
    const r = loadRecentJoinUrls();
    return r[0] ?? "ws://127.0.0.1:27300";
  });
  const [displayName, setDisplayName] = useState(() => loadPreferences().collabDisplayName);
  const [accentColor, setAccentColor] = useState(() => loadPreferences().collabAccentColor);
  const [roster, setRoster] = useState<RosterEntry[]>([]);
  const [chatLines, setChatLines] = useState<string[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatPanelOpen, setChatPanelOpen] = useState(false);
  const [chatToasts, setChatToasts] = useState<ChatToast[]>([]);
  const chatToastIdRef = useRef(0);
  const chatPanelOpenRef = useRef(false);
  const pingHudRef = useRef<{
    name: string;
    wx: number;
    wy: number;
    wz: number;
    until: number;
    emoji?: string;
  } | null>(null);
  const [pingHudTick, setPingHudTick] = useState(0);
  // Radial emoji-ping menu state
  const [radialMenu, setRadialMenu] = useState<{ x: number; y: number; visible: boolean }>({
    x: 0,
    y: 0,
    visible: false,
  });
  const radialHoldTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Stash the normalized pick coords on Z-down so the keyup/radial-select path can use them. */
  const pendingPingRef = useRef<{ nx: number; ny: number } | null>(null);
  /** Last known cursor screen position (updated on every pointermove). */
  const lastCursorScreenRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const collabActiveRef = useRef(false);
  const localPeerIdRef = useRef(0);
  const [hostPort, setHostPort] = useState(() => loadPreferences().collabHostPort);
  /** From preferences: UPnP when hosting (default off). */
  const [prefsEnableUpnp, setPrefsEnableUpnp] = useState(() => loadPreferences().enableUpnp);
  /** Set when UPnP reports a public WebSocket URL (host only). */
  const [hostWanUrl, setHostWanUrl] = useState<string | null>(null);
  const [natPending, setNatPending] = useState(false);
  const [natError, setNatError] = useState<string | null>(null);
  const [lastSessionInfo, setLastSessionInfo] = useState<LastSessionInfo | null>(null);
  const [lastSessionReady, setLastSessionReady] = useState(false);
  /** True from startup until we've resolved whether to auto-reopen; hides start screen during that window. */
  const [pendingAutoReopen, setPendingAutoReopen] = useState(
    () => loadPreferences().reopenLastProject,
  );
  const [collabActive, setCollabActive] = useState(false);
  /** Set when hosting or after welcome; 0 when solo. */
  const [localPeerId, setLocalPeerId] = useState(0);
  const [hostingCopied, setHostingCopied] = useState(false);
  const [sceneObjects, setSceneObjects] = useState<SceneObjectRow[]>([]);
  const [activeObjectId, setActiveObjectId] = useState(0);
  const [sceneObjectsErr, setSceneObjectsErr] = useState<string | null>(null);

  const refreshSceneObjects = useCallback(() => {
    void invoke<{ objects: SceneObjectRow[]; activeObjectId: number }>("get_scene_objects")
      .then((p) => {
        setSceneObjects(p.objects);
        setActiveObjectId(p.activeObjectId);
        setSceneObjectsErr(null);
      })
      .catch((e: unknown) => {
        setSceneObjects([]);
        setSceneObjectsErr(String(e));
      });
  }, []);

  const hexToRgb = (hex: string): number => {
    const h = hex.replace("#", "");
    const n = parseInt(
      h.length === 3
        ? h
            .split("")
            .map((c) => c + c)
            .join("")
        : h,
      16,
    );
    return n & 0xffffff;
  };

  const sendResize = useCallback(() => {
    const el = viewportRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const { w: layoutW, h: layoutH } = layoutViewportCssSize();

    const rect = el.getBoundingClientRect();
    const rw = rect.width;
    const rh = rect.height;
    if (rw <= 0 || rh <= 0) return;

    // Prefer last native swapchain size so configure matches drawable; bootstrap with layout×dpr.
    const innerChanged =
      lastLayoutViewportCssRef.current.w !== layoutW ||
      lastLayoutViewportCssRef.current.h !== layoutH;
    if (innerChanged) {
      lastLayoutViewportCssRef.current = { w: layoutW, h: layoutH };
    }
    const surf = surfacePhysRef.current;
    // Height-first bootstrap matches typical swapchain rounding and pairs with viewport math below.
    const bootstrapH = Math.max(1, Math.round(layoutH * dpr));
    const bootstrapW = Math.max(1, Math.round(bootstrapH * (layoutW / layoutH)));
    // After a window resize, native size is unknown until the next frame — use bootstrap for configure + origin.
    const useNativeSurface = surf.w > 0 && surf.h > 0 && !innerChanged;
    const surfaceWidth = useNativeSurface ? surf.w : bootstrapW;
    const surfaceHeight = useNativeSurface ? surf.h : bootstrapH;

    // Derive viewport texture size from the same surface×layout fractions as viewportX/Y. Using
    // round(rh*dpr) here while origin uses (rect.top/ih)*surfaceHeight caused vertical drift when
    // surfaceHeight ≠ ih*dpr (native swapchain vs CSS estimate).
    const viewportHeight = Math.max(1, Math.round((rh / layoutH) * surfaceHeight));
    const viewportWidth = Math.max(1, Math.round(viewportHeight * (rw / rh)));
    // Proportional placement in the same pixel space as the swapchain (not raw rect×dpr alone).
    const viewportX = Math.max(0, Math.round((rect.left / layoutW) * surfaceWidth));
    const viewportY = Math.max(0, Math.round((rect.top / layoutH) * surfaceHeight));
    viewportPhysRef.current = { w: viewportWidth, h: viewportHeight };
    void invoke("viewer_resize", {
      surfaceWidth,
      surfaceHeight,
      viewportX,
      viewportY,
      viewportWidth,
      viewportHeight,
    })
      .then(() =>
        invoke<{
          width: number;
          height: number;
          surfaceWidth: number;
          surfaceHeight: number;
        }>("get_viewport_pixel_size"),
      )
      .then((sz) => {
        viewportPhysRef.current = { w: sz.width, h: sz.height };
        surfacePhysRef.current = { w: sz.surfaceWidth, h: sz.surfaceHeight };
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const p = loadPreferences();
    void invoke("set_emission_lighting", {
      enabled: p.enableEmissionLighting,
    }).catch(() => {});
    void invoke("set_gizmo_on_top", { enabled: p.gizmoOnTop }).catch(() => {});
  }, []);

  useEffect(() => {
    chatPanelOpenRef.current = chatPanelOpen;
    collabActiveRef.current = collabActive;
    localPeerIdRef.current = localPeerId;
  }, [chatPanelOpen, collabActive, localPeerId]);

  useEffect(() => {
    startScreenLogoLoadedRef.current = startScreenLogoLoaded;
  }, [startScreenLogoLoaded]);

  useEffect(() => {
    if (chatPanelOpen) setChatToasts([]);
  }, [chatPanelOpen]);

  useEffect(() => {
    const w = window as unknown as {
      toggleVoxelleViewportCursorDebug?: () => void;
    };
    w.toggleVoxelleViewportCursorDebug = () => {
      try {
        const on = localStorage.getItem(LS_VIEWPORT_CURSOR_DEBUG) !== "1";
        localStorage.setItem(LS_VIEWPORT_CURSOR_DEBUG, on ? "1" : "0");
        setViewportCursorDebugEnabled(on);
        if (!on) {
          setViewportCursorDebugJs(null);
          setViewportCursorDebugRust(null);
          viewportCursorDebugScreenRef.current = null;
          setViewportCursorDebugScreen(null);
        }
        void invoke("debug_menu_sync_viewport_cursor_overlay", {
          enabled: on,
        }).catch(() => {});
      } catch {
        /* ignore */
      }
    };
    return () => {
      delete w.toggleVoxelleViewportCursorDebug;
      if (viewportCursorDebugRafRef.current != null) {
        cancelAnimationFrame(viewportCursorDebugRafRef.current);
      }
    };
  }, []);

  useEffect(() => {
    try {
      const enabled = localStorage.getItem(LS_VIEWPORT_CURSOR_DEBUG) === "1";
      void invoke("debug_menu_sync_viewport_cursor_overlay", {
        enabled,
      }).catch(() => {});
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    sendResize();
    const ro = new ResizeObserver(() => sendResize());
    const el = viewportRef.current;
    if (el) ro.observe(el);

    const clearCollabSessionUi = () => {
      pendingJoinUrlRef.current = null;
      setCollabJoinPending(false);
      setCollabActive(false);
      setHostWsUrl(null);
      setHostWanUrl(null);
      setNatPending(false);
      setNatError(null);
      setRoster([]);
      setLocalPeerId(0);
      setPingMs(null);
      setChatLines([]);
      setChatInput("");
      setChatToasts([]);
      setHostingCopied(false);
    };

    /** `listen()` is async; React Strict Mode runs cleanup before those promises resolve, which used to call stale `unlisten` and break Tauri's listener table. */
    let active = true;
    const unlistenReady = Promise.all([
      listen<string>("voxelle-load-start", (e) => {
        setLoadError(null);
        setCollabBanner(null);
        setStartScreenLogoLoaded(false);
        setPathLabel(e.payload);
        setLoading(true);
        setLoadProgress(0);
        setLoadPhase("");
        setSpeechBubbles([]);
      }),
      listen<{ fraction: number; phase: string }>("voxelle-load-progress", (e) => {
        const p = e.payload;
        setLoadProgress(p.fraction);
        setLoadPhase(p.phase);
        if (p.fraction >= 1) {
          setLoading(false);
          setLoadPhase("");
        }
      }),
      listen<{ fraction: number; phase: string }>("voxelle-work-progress", (e) => {
        const p = e.payload;
        setWorkProgress(p.fraction);
        setWorkPhase(p.phase);
        if (p.fraction >= 1) {
          setWorkBusy(false);
          setWorkPhase("");
          fillOperationPendingRef.current = false;
          setFillOperationPending(false);
        } else {
          setWorkBusy(true);
        }
      }),
      listen<unknown>("logo-loaded", () => {
        setStartScreenLogoLoaded(true);
      }),
      listen<boolean>("voxelle-debug-logo-light-controls", (e) => {
        setLogoLightControlsVisible(e.payload);
      }),
      listen<unknown>("voxelle-loaded", (e) => {
        setLoadError(null);
        const p = e.payload;
        if (typeof p === "string") {
          setPathLabel(p);
          setStartScreenLogoLoaded(false);
        } else if (p && typeof p === "object" && "path" in p) {
          const o = p as {
            path: string;
            mood?: Partial<MoodState>;
          };
          setPathLabel(o.path);
          setStartScreenLogoLoaded(false);
          if (o.mood) {
            setMood(moodWith(defaultMoodState(), o.mood));
          } else {
            setMood(defaultMoodState());
          }
        }
        setLoading(false);
        setLoadProgress(1);
        setLoadPhase("");
        refreshSceneObjects();
      }),
      listen<string>("voxelle-load-error", (e) => {
        setLoadError(e.payload);
        setLoading(false);
        setLoadPhase("");
        setCollabJoinPending((p) => (p ? false : p));
      }),
      listen<number>("viewport-fps", (e) => {
        setFpsDisplayed(e.payload);
      }),
      listen<number>("collab-latency-ms", (e) => {
        setPingMs(e.payload);
      }),
      listen<{
        width: number;
        height: number;
        surfaceWidth: number;
        surfaceHeight: number;
      }>("viewport-pixel-size", (e) => {
        const p = e.payload;
        viewportPhysRef.current = { w: p.width, h: p.height };
        surfacePhysRef.current = { w: p.surfaceWidth, h: p.surfaceHeight };
      }),
      listen("voxelle-open-new-project", () => {
        setNewProjectOpen(true);
      }),
      listen("voxelle-project-closed", () => {
        setPathLabel("");
        setLoading(false);
        setLoadProgress(0);
        setLoadPhase("");
        setLoadError(null);
        setWorkBusy(false);
        setSpeechBubbles([]);
        setMood(defaultMoodState());
        void invoke("load_start_screen_logo").catch(() => {});
      }),
      listen("voxelle-collab-start-session", () => {
        if (collabActiveMenuRef.current) return;
        startHostMenuRef.current();
      }),
      listen("voxelle-collab-join-session", () => {
        setJoinModalOpen(true);
      }),
      listen("voxelle-collab-leave-session", () => {
        if (!collabActiveMenuRef.current) return;
        leaveSessionMenuRef.current();
      }),
      listen("voxelle-show-chat-panel", () => {
        setChatPanelOpen(true);
      }),
      listen("voxelle-open-preferences", () => {
        setPreferencesOpen(true);
      }),
      listen("voxelle-menu-stamp-book", () => {
        setStampBookOpen(true);
      }),
      listen<string>("collab-ping", (e) => {
        try {
          const j = JSON.parse(e.payload) as {
            displayName?: string;
            display_name?: string;
            x?: number;
            y?: number;
            z?: number;
            emoji?: string;
          };
          const name = j.displayName ?? j.display_name ?? "?";
          const vx = j.x ?? 0;
          const vy = j.y ?? 0;
          const vz = j.z ?? 0;
          pingHudRef.current = {
            name,
            wx: vx + 0.5,
            wy: vy + 0.5,
            wz: vz + 0.5,
            until: Date.now() + PING_HUD_MS,
            emoji: j.emoji || undefined,
          };
          setPingHudTick((n) => n + 1);
          playPingSound();
        } catch {
          /* ignore */
        }
      }),
      listen<string>("collab-chat", (e) => {
        let line: string;
        let fromPeerId: number | undefined;
        try {
          const j = JSON.parse(e.payload) as {
            displayName?: string;
            display_name?: string;
            text?: string;
            peer_id?: number;
            peerId?: number;
          };
          const who = j.displayName ?? j.display_name ?? "?";
          line = `${who}: ${j.text ?? ""}`;
          fromPeerId = j.peerId ?? j.peer_id;
          setChatLines((prev) => [...prev.slice(-80), line]);
        } catch {
          line = e.payload;
          setChatLines((prev) => [...prev.slice(-80), e.payload]);
        }
        const showToast =
          collabActiveRef.current &&
          !chatPanelOpenRef.current &&
          (fromPeerId === undefined || fromPeerId !== localPeerIdRef.current);
        if (showToast) {
          setChatToasts((prev) => {
            const id = ++chatToastIdRef.current;
            const next = [...prev, { id, text: line }];
            return next.length > CHAT_TOAST_CAP ? next.slice(-CHAT_TOAST_CAP) : next;
          });
        }
      }),
      listen("collab-joined", () => {
        setCollabBanner(null);
        setCollabActive(true);
        setCollabJoinPending(false);
        const u = pendingJoinUrlRef.current;
        if (u) {
          rememberJoinedUrl(u);
          pendingJoinUrlRef.current = null;
        }
        setJoinModalOpen(false);
        // Announce our avatar choice so other peers see the right model immediately.
        const avatarName = loadPreferences().collabAvatarName;
        void invoke("set_local_avatar", { avatarName }).catch(() => {});
      }),
      listen<number>("collab-local-peer", (e) => {
        setLocalPeerId(typeof e.payload === "number" ? e.payload : 0);
      }),
      listen<string>("collab-roster", (e) => {
        try {
          const arr = JSON.parse(e.payload) as RosterEntry[];
          setRoster(arr);
        } catch {
          /* ignore */
        }
      }),
      listen<string>("collab-peer-left", (e) => {
        if (localPeerIdRef.current !== 1) return;
        try {
          const j = JSON.parse(e.payload) as {
            displayName?: string;
            reason?: string;
          };
          const name =
            typeof j.displayName === "string" && j.displayName.length > 0 ? j.displayName : "Guest";
          const text = j.reason === "left" ? `${name} left the session.` : `${name} disconnected.`;
          setCollabBanner({ text, tone: "info" });
        } catch {
          /* ignore */
        }
      }),
      listen<string>("collab-error", (e) => {
        pendingJoinUrlRef.current = null;
        setCollabJoinPending(false);
        setLoadError(e.payload);
      }),
      listen<unknown>("collab-nat-result", (e) => {
        try {
          const raw = e.payload;
          const j =
            typeof raw === "string"
              ? (JSON.parse(raw) as {
                  wanUrl?: string | null;
                  error?: string | null;
                })
              : (raw as { wanUrl?: string | null; error?: string | null });
          setNatPending(false);
          setNatError(typeof j.error === "string" && j.error.length > 0 ? j.error : null);
          setHostWanUrl(typeof j.wanUrl === "string" && j.wanUrl.length > 0 ? j.wanUrl : null);
        } catch {
          setNatPending(false);
        }
      }),
      listen<string>("collab-ended", (e) => {
        const text = typeof e.payload === "string" ? e.payload : String(e.payload ?? "");
        if (text.trim().length > 0) {
          setCollabBanner({ text, tone: "info" });
        }
        clearCollabSessionUi();
      }),
      listen<string>("collab-kicked", (e) => {
        const msg = typeof e.payload === "string" ? e.payload : String(e.payload ?? "");
        setCollabBanner({
          text: `Removed from session: ${msg}`,
          tone: "alert",
        });
        clearCollabSessionUi();
      }),
      listen("voxelle-check-updates", async () => {
        try {
          const update = await check();
          if (!update) {
            window.alert("You're up to date.");
            return;
          }
          const ok = await invoke<boolean>("confirm_app_update_dialog", {
            message: `Download and install Voxelle Desktop ${update.version}?`,
            title: "Update available",
          });
          if (!ok) return;
          await update.downloadAndInstall();
          await relaunch();
        } catch (e) {
          window.alert(userFacingUpdaterError(e));
        }
      }),
      listen<string>("voxelle-rendering-mode-changed", (e) => {
        const m = e.payload;
        if (m === "greedy" || m === "marchingCubes" || m === "dualContour" || m === "ray") {
          localStorage.setItem(LS_RENDERING_MODE, m);
        }
      }),
      listen("voxelle-reload-start-screen-overlays", () => {
        void invoke("load_start_screen_logo").catch(() => {});
        void invoke("mascot_load_embedded", { id: 0, name: "seagull" }).catch(() => {});
      }),
      listen<string>("voxelle-menu-selection-mode", (e) => {
        const m = e.payload;
        if (m === "selectByColor" || m === "selectCoplanar" || m === "selectCoplanarEmpty") {
          setInteractionMode(m);
        }
      }),
      listen<boolean>("voxelle-menu-match-material", (e) => {
        setMatchMaterialSelectColor(e.payload);
      }),
      listen<boolean>("voxelle-debug-viewport-cursor-overlay", (e) => {
        const enabled = e.payload;
        try {
          localStorage.setItem(LS_VIEWPORT_CURSOR_DEBUG, enabled ? "1" : "0");
        } catch {
          /* ignore */
        }
        setViewportCursorDebugEnabled(enabled);
        if (!enabled) {
          setViewportCursorDebugJs(null);
          setViewportCursorDebugRust(null);
          viewportCursorDebugScreenRef.current = null;
          setViewportCursorDebugScreen(null);
        }
      }),
      listen<{
        frame_count: number;
        viewport_width: number;
        viewport_height: number;
        total_ms: number;
        avg_ms: number;
        stddev_ms: number;
        min_ms: number;
        p50_ms: number;
        p95_ms: number;
        p99_ms: number;
        max_ms: number;
        mpix_per_sec: number;
        frame_times_ms: number[];
      }>("voxelle-debug-raytrace-benchmark", (e) => {
        const r = e.payload;
        const f = (n: number) => n.toFixed(2);
        console.group(
          `Ray trace benchmark — ${r.viewport_width}×${r.viewport_height} — ${r.frame_count} frames — ${f(r.mpix_per_sec)} Mpix/s`,
        );
        console.log(
          `avg ${f(r.avg_ms)} ms  σ ${f(r.stddev_ms)} ms  min ${f(r.min_ms)} ms  p50 ${f(r.p50_ms)} ms  p95 ${f(r.p95_ms)} ms  p99 ${f(r.p99_ms)} ms  max ${f(r.max_ms)} ms`,
        );
        console.log(`total ${f(r.total_ms)} ms over ${r.frame_count} frames`);
        console.log(
          "frame times (ms):",
          r.frame_times_ms.map((t) => +t.toFixed(2)),
        );
        console.groupEnd();
      }),
      listen<boolean>("voxelle-hide-ui", (e) => {
        setHideUI(e.payload);
      }),
      listen<number>("voxelle-selection-updated", (e) => {
        setSelectionCount(typeof e.payload === "number" ? e.payload : 0);
      }),
      listen<string>("voxelle-selection-combine-mode", (e) => {
        const p = e.payload;
        if (p === "replace" || p === "add" || p === "subtract" || p === "intersect") {
          setSelectionCombineMode(p);
        }
      }),
      listen<string>("voxelle-menu-not-implemented", (e) => {
        const msg = typeof e.payload === "string" ? e.payload : String(e.payload ?? "");
        console.warn(msg);
      }),
      listen("voxelle-menu-rotate-selection", () => {
        setRotateDialogOpen(true);
      }),
      listen("voxelle-menu-scale-selection", () => {
        setScaleDialogOpen(true);
      }),
    ]).then((unlisteners) => {
      if (!active) {
        unlisteners.forEach((u) => u());
        return undefined;
      }
      return unlisteners;
    });

    return () => {
      ro.disconnect();
      active = false;
      void unlistenReady.then((uns) => {
        if (uns) uns.forEach((u) => u());
      });
    };
  }, [sendResize, refreshSceneObjects]);

  /** Sidebars change flex width; sync native viewer after layout so `.viewport` matches `viewer_resize`. */
  useLayoutEffect(() => {
    sendResize();
    const id = requestAnimationFrame(() => {
      sendResize();
    });
    return () => cancelAnimationFrame(id);
  }, [
    sendResize,
    sidebarExpanded,
    rightSidebarExpanded,
    toolsPaneFloating,
    colorPaletteFloating,
    colorPalettePos,
    pathLabel,
    collabActive,
    loading,
    collabJoinPending,
    workBusy,
  ]);

  const onToolPaneDragMove = useCallback((e: PointerEvent) => {
    const d = toolPaneDragRef.current;
    if (!d || e.pointerId !== d.pid) return;
    const dx = e.clientX - d.startX;
    const dy = e.clientY - d.startY;
    setToolPanePos(() => {
      let nx = d.origX + dx;
      let ny = d.origY + dy;
      const pad = 8;
      const maxX = Math.max(pad, window.innerWidth - 160);
      const maxY = Math.max(pad, window.innerHeight - 80);
      nx = Math.min(Math.max(pad, nx), maxX);
      ny = Math.min(Math.max(pad, ny), maxY);
      return { x: nx, y: ny };
    });
  }, []);

  const onToolPaneDragEnd = useCallback(
    (e: PointerEvent) => {
      const d = toolPaneDragRef.current;
      if (!d || e.pointerId !== d.pid) return;
      toolPaneDragRef.current = null;
      window.removeEventListener("pointermove", onToolPaneDragMove);
      window.removeEventListener("pointerup", onToolPaneDragEnd);
      window.removeEventListener("pointercancel", onToolPaneDragEnd);
    },
    [onToolPaneDragMove],
  );

  const onToolPaneDragDown = useCallback(
    (e: React.PointerEvent) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const p = toolPanePosRef.current;
      toolPaneDragRef.current = {
        pid: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        origX: p.x,
        origY: p.y,
      };
      window.addEventListener("pointermove", onToolPaneDragMove);
      window.addEventListener("pointerup", onToolPaneDragEnd);
      window.addEventListener("pointercancel", onToolPaneDragEnd);
    },
    [onToolPaneDragMove, onToolPaneDragEnd],
  );

  // Expire ping HUD ref after its duration (label now rendered on GPU)
  useEffect(() => {
    if (pingHudTick === 0 && !pingHudRef.current) return;
    let cancelled = false;
    let raf = 0;
    const tick = () => {
      if (cancelled) return;
      const p = pingHudRef.current;
      if (p && Date.now() > p.until) {
        pingHudRef.current = null;
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [pingHudTick]);

  useEffect(() => {
    const prev = interactionModeRef.current;
    interactionModeRef.current = interactionMode;
    // Clear squishy session when leaving squishy mode
    if (prev === "squishy" && interactionMode !== "squishy") {
      if (squishyPhase.active) {
        squishyPhase.cancel();
      } else {
        void invoke("squishy_session_clear")
          .then(() => setSquishyBallCount(0))
          .catch(() => {});
      }
    }
  }, [interactionMode]);

  const prevInteractionModeForEyedropperRef = useRef<InteractionMode>(interactionMode);
  const eyedropperReturnModeRef = useRef<InteractionMode | null>(null);
  useLayoutEffect(() => {
    const prev = prevInteractionModeForEyedropperRef.current;
    if (interactionMode === "eyedropper" && prev !== "eyedropper") {
      eyedropperReturnModeRef.current = prev;
    }
    prevInteractionModeForEyedropperRef.current = interactionMode;
  }, [interactionMode]);

  useEffect(() => {
    void invoke<SelectionCombineModeApi>("get_selection_combine_mode")
      .then((m) => setSelectionCombineMode(m))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (startScreenLogoInvokeSent) return;
    startScreenLogoInvokeSent = true;
    void invoke("load_start_screen_logo").catch(() => {});
  }, []);

  // ── Mascot position (lower-left corner, tracks window size) ──────────────
  useEffect(() => {
    const update = () =>
      setMascotRect({
        x: MASCOT_PAD,
        y: window.innerHeight - MASCOT_H - MASCOT_PAD,
        width: MASCOT_W,
        height: MASCOT_H,
      });
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  // ── Reposition speech bubbles when the mascot moves (window resize) ───────
  useEffect(() => {
    if (speechBubbles.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const BUBBLE_W = 280;
    const bubbleX = mascotRect.x + mascotRect.width + 12;
    const bubbleY = mascotRect.y - 4;
    const tailX = mascotRect.x + mascotRect.width * 0.35;
    const tailY = mascotRect.y + mascotRect.height * 0.22;
    for (const b of speechBubbles) {
      void invoke("speech_bubble_reposition", {
        id: b.id,
        rx: bubbleX * dpr,
        ry: bubbleY * dpr,
        rw: BUBBLE_W * dpr,
        rh: b.height * dpr,
        tx: tailX * dpr,
        ty: tailY * dpr,
      });
    }
    setSpeechBubbles((prev) =>
      prev.map((b) => ({ ...b, x: bubbleX, y: bubbleY, width: BUBBLE_W })),
    );
  }, [mascotRect]);

  // ── Mascot loading ────────────────────────────────────────────────────────
  // Await listener registration before invoking so the "mascot-loaded" event
  // cannot fire before the handler is wired up.
  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    void (async () => {
      unlisten = await listen<number>("mascot-loaded", (ev) => {
        if (ev.payload === 0) setMascotsLoaded(true);
      });
      if (cancelled) {
        unlisten();
        return;
      }
      void invoke("mascot_load_embedded", { id: 0, name: "seagull" }).catch((e) =>
        console.error("[voxelle] mascot load error", e),
      );
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  // ── Speech bubble dismissed event ─────────────────────────────────────────
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    void listen<number>("speech-bubble-dismissed", (ev) => {
      setSpeechBubbles((prev) => prev.filter((b) => b.id !== ev.payload));
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, []);

  /** Show or advance the mascot's greeting speech bubble. */
  const handleMascotClick = useCallback(
    async (mascotId: number) => {
      if (mascotId !== 0) return;

      // If a bubble is already open for this mascot, forward the click to it.
      if (speechBubbles.length > 0) {
        void invoke("speech_bubble_click", { id: speechBubbles[0].id });
        return;
      }

      const dpr = window.devicePixelRatio || 1;
      const BUBBLE_W = 280;
      const BUBBLE_H = 96;
      // Position bubble to the right of and level with the mascot top edge.
      const bubbleX = mascotRect.x + mascotRect.width + 12;
      const bubbleY = mascotRect.y - 4;
      // Tail tip: upper-left area of the mascot (its head region).
      const tailX = mascotRect.x + mascotRect.width * 0.35;
      const tailY = mascotRect.y + mascotRect.height * 0.22;

      const id = nextBubbleId.current++;
      const computedRhPx = await invoke<number>("speech_bubble_show", {
        id,
        pages: [generateIdea()],
        rx: bubbleX * dpr,
        ry: bubbleY * dpr,
        rw: BUBBLE_W * dpr,
        rh: BUBBLE_H * dpr,
        tx: tailX * dpr,
        ty: tailY * dpr,
      });
      const actualH = computedRhPx / dpr;
      setSpeechBubbles((prev) => [
        ...prev,
        { id, x: bubbleX, y: bubbleY, width: BUBBLE_W, height: actualH },
      ]);
      playSeagullSpeech();
    },
    [mascotRect, speechBubbles],
  );

  useEffect(() => {
    selectionCountRef.current = selectionCount;
  }, [selectionCount]);
  useEffect(() => {
    activeColorRef.current = activeColor;
    if (selectionCountRef.current > 0) {
      void invoke("paint_selection", {
        args: {
          color: activeColor,
          strokeSeed: Math.floor(Math.random() * 0xffffffff),
          material: activeMaterialRef.current,
        },
      }).catch((e) => console.error("[voxelle] paint_selection error", e));
    }
  }, [activeColor]);
  useEffect(() => {
    selectedColorsRef.current = selectedColors;
    if (selectionCountRef.current > 0 && selectedColors.length >= 1) {
      void invoke("paint_selection", {
        args: {
          color: activeColorRef.current,
          palette: selectedColors,
          paintColorDistrib: paintColorDistribRef.current,
          strokeSeed: Math.floor(Math.random() * 0xffffffff),
          material: activeMaterialRef.current,
        },
      }).catch((e) => console.error("[voxelle] paint_selection error", e));
    }
  }, [selectedColors]);
  useEffect(() => {
    paintColorDistribRef.current = paintColorDistrib;
    try {
      localStorage.setItem(LS_PAINT_COLOR_DISTRIB, JSON.stringify(paintColorDistrib));
    } catch {}
  }, [paintColorDistrib]);
  useEffect(() => {
    activeMaterialRef.current = activeMaterial;
    if (selectionCountRef.current > 0) {
      const palette = selectedColorsRef.current;
      const multiColor = palette.length > 1;
      void invoke("paint_selection", {
        args: {
          color: activeColorRef.current,
          ...(multiColor ? { palette, paintColorDistrib: paintColorDistribRef.current } : {}),
          strokeSeed: Math.floor(Math.random() * 0xffffffff),
          material: activeMaterial,
        },
      }).catch((e) => console.error("[voxelle] paint_selection error", e));
    }
  }, [activeMaterial]);
  useEffect(() => {
    brushRadiusRef.current = brushRadius;
  }, [brushRadius]);
  useEffect(() => {
    brushShapeRef.current = brushShape;
  }, [brushShape]);
  useEffect(() => {
    mirrorXRef.current = mirrorX;
  }, [mirrorX]);
  useEffect(() => {
    mirrorYRef.current = mirrorY;
  }, [mirrorY]);
  useEffect(() => {
    mirrorZRef.current = mirrorZ;
  }, [mirrorZ]);
  useEffect(() => {
    brushClipBottomHalfRef.current = brushClipBottomHalf;
  }, [brushClipBottomHalf]);
  const generatorSphereRadiusRef = useRef(4);
  const generatorKindRef = useRef<GeneratorKindId>("rocks");
  const sculptStrokeModeRef = useRef<SculptStrokeModeApi>("draw");
  const terrainSculptOpRef = useRef<TerrainSculptOpApi>("raise");
  const terrainBaseYRef = useRef(0);
  const terrainStrengthRef = useRef(4);
  const terrainSmoothRadiusRef = useRef(2);
  const terrainFlattenUseBaseYRef = useRef(false);
  const terrainSubVoxelRef = useRef(false);
  const lastTerrainHoverMsRef = useRef(0);
  const sculptSmoothPassesRef = useRef(1);
  const sculptBrushRadiusRef = useRef(2);
  const sculptBrushStrengthRef = useRef(100);
  const sculptBrushFalloffRef = useRef(0);
  const sculptBrushShapeUiRef = useRef<SculptBrushShapeUi>("circle");
  const extrudeProfileRef = useRef<"cube" | "cylinder">("cube");
  const extrudeEndCapRef = useRef<"flat" | "rounded" | "pointed">("flat");
  const extrudeTaperRef = useRef(false);
  const extrudeTaperStartRef = useRef(3);
  const extrudeTaperEndRef = useRef(0);
  const wallAreaShapeRef = useRef<WallAreaShapeApi>("brush");
  const sprayDirectionRef = useRef<SprayDirectionApi>("auto");
  const wallWidthIndexRef = useRef(0);
  const wallHeightVoxRef = useRef(2);
  const wallLockStartHeightRef = useRef(false);
  const wallAxisAlignRef = useRef(false);
  const sculptSmoothVariantRef = useRef<SculptSmoothVariantApi>("majority");
  const smoothNeighborRadiusRef = useRef(0);
  const smoothAggressivenessRef = useRef(100);
  const smoothLaplacianIterationsRef = useRef(4);
  const smoothLaplacianRelaxPctRef = useRef(50);
  const wallSculptPolygonVertsRef = useRef<[number, number, number][]>([]);
  useEffect(() => {
    generatorSphereRadiusRef.current = generatorSphereRadius;
  }, [generatorSphereRadius]);
  useEffect(() => {
    generatorKindRef.current = generatorKind;
  }, [generatorKind]);
  useEffect(() => {
    sculptStrokeModeRef.current = sculptStrokeMode;
  }, [sculptStrokeMode]);
  useEffect(() => {
    terrainSculptOpRef.current = terrainSculptOp;
  }, [terrainSculptOp]);
  useEffect(() => {
    terrainBaseYRef.current = terrainBaseY;
  }, [terrainBaseY]);
  useEffect(() => {
    terrainStrengthRef.current = terrainStrength;
  }, [terrainStrength]);
  useEffect(() => {
    terrainSmoothRadiusRef.current = terrainSmoothRadius;
  }, [terrainSmoothRadius]);
  useEffect(() => {
    terrainFlattenUseBaseYRef.current = terrainFlattenUseBaseY;
  }, [terrainFlattenUseBaseY]);
  useEffect(() => {
    terrainSubVoxelRef.current = terrainSubVoxel;
  }, [terrainSubVoxel]);
  useEffect(() => {
    sculptSmoothPassesRef.current = sculptSmoothPasses;
  }, [sculptSmoothPasses]);
  useEffect(() => {
    sculptBrushRadiusRef.current = sculptBrushRadius;
  }, [sculptBrushRadius]);
  useEffect(() => {
    sculptBrushStrengthRef.current = sculptBrushStrength;
  }, [sculptBrushStrength]);
  useEffect(() => {
    sculptBrushFalloffRef.current = sculptBrushFalloff;
  }, [sculptBrushFalloff]);
  useEffect(() => {
    sculptBrushShapeUiRef.current = sculptBrushShapeUi;
  }, [sculptBrushShapeUi]);
  useEffect(() => {
    extrudeProfileRef.current = extrudeProfile;
  }, [extrudeProfile]);
  useEffect(() => {
    extrudeEndCapRef.current = extrudeEndCap;
  }, [extrudeEndCap]);
  useEffect(() => {
    extrudeTaperRef.current = extrudeTaper;
  }, [extrudeTaper]);
  useEffect(() => {
    extrudeTaperStartRef.current = extrudeTaperStart;
  }, [extrudeTaperStart]);
  useEffect(() => {
    extrudeTaperEndRef.current = extrudeTaperEnd;
  }, [extrudeTaperEnd]);
  useEffect(() => {
    wallAreaShapeRef.current = wallAreaShape;
  }, [wallAreaShape]);
  useEffect(() => {
    sprayDirectionRef.current = sprayDirection;
  }, [sprayDirection]);
  useEffect(() => {
    wallWidthIndexRef.current = wallWidthIndex;
  }, [wallWidthIndex]);
  useEffect(() => {
    wallHeightVoxRef.current = wallHeightVox;
  }, [wallHeightVox]);
  useEffect(() => {
    wallLockStartHeightRef.current = wallLockStartHeight;
  }, [wallLockStartHeight]);
  useEffect(() => {
    wallAxisAlignRef.current = wallAxisAlign;
  }, [wallAxisAlign]);
  useEffect(() => {
    sculptSmoothVariantRef.current = sculptSmoothVariant;
  }, [sculptSmoothVariant]);
  useEffect(() => {
    smoothNeighborRadiusRef.current = smoothNeighborRadius;
  }, [smoothNeighborRadius]);
  useEffect(() => {
    smoothAggressivenessRef.current = smoothAggressiveness;
  }, [smoothAggressiveness]);
  useEffect(() => {
    smoothLaplacianIterationsRef.current = smoothLaplacianIterations;
  }, [smoothLaplacianIterations]);
  useEffect(() => {
    smoothLaplacianRelaxPctRef.current = smoothLaplacianRelaxPct;
  }, [smoothLaplacianRelaxPct]);
  useEffect(() => {
    wallSculptPolygonVertsRef.current = wallSculptPolygonVerts;
  }, [wallSculptPolygonVerts]);
  useEffect(() => {
    if (wallAreaShape !== "polygon" || sculptStrokeMode !== "wall") {
      setWallSculptPolygonVerts([]);
    }
  }, [wallAreaShape, sculptStrokeMode]);
  useEffect(() => {
    selectionStrokeSnapToSurfaceRef.current = selectionStrokeSnapToSurface;
  }, [selectionStrokeSnapToSurface]);
  useEffect(() => {
    selectionStrokeAxisAlignRef.current = selectionStrokeAxisAlign;
  }, [selectionStrokeAxisAlign]);
  useEffect(() => {
    strokeDrawStyleRef.current = strokeDrawStyle;
  }, [strokeDrawStyle]);
  useEffect(() => {
    strokeFamilyVariantRef.current = strokeFamilyVariant;
  }, [strokeFamilyVariant]);
  useEffect(() => {
    drawStrokeModeRef.current = drawStrokeMode;
    if (drawStrokeMode !== "cuboid" && cuboidPhase.active) {
      cuboidPhase.cancel();
    }
    if (drawStrokeMode !== "cylinder" && cylinderPhase.active) {
      cylinderPhase.cancel();
    }
  }, [drawStrokeMode]);
  useEffect(() => {
    planeAxisRef.current = planeAxis;
  }, [planeAxis]);
  useEffect(() => {
    strokeClickRef.current = {
      circleCenter: null,
    };
    setStrokePolygonVerts([]);
    strokePolygonVertsRef.current = [];
    strokePolygonLastScreenRef.current = null;
  }, [drawStrokeMode]);
  useEffect(() => {
    strokePolygonVertsRef.current = strokePolygonVerts;
  }, [strokePolygonVerts]);
  useEffect(() => {
    clothPinsRef.current = clothPins;
  }, [clothPins]);
  useEffect(() => {
    roofAreaShapeRef.current = roofAreaShape;
  }, [roofAreaShape]);
  const ropeFirstScreenRef = useRef<{ nx: number; ny: number } | null>(null);
  const ropeSagRef = useRef(ropeSag);
  const ropeTensionRef = useRef(ropeTension);
  const ropeBrushRadiusIndexRef = useRef(ropeBrushRadiusIndex);
  const ropeBrushShapeUiRef = useRef<"sphere" | "cube">(ropeBrushShapeUi);
  const clothTensionRef = useRef(clothTension);
  const clothGravityDirectionRef = useRef(clothGravityDirection);
  const clothSimGravityPctRef = useRef(clothSimGravityPct);
  const clothSimStiffnessPctRef = useRef(clothSimStiffnessPct);
  const clothSimIterationsRef = useRef(clothSimIterations);
  const clothSimConstraintPassesRef = useRef(clothSimConstraintPasses);
  const rockRoughnessRef = useRef(rockRoughness);
  const ashlarThicknessRef = useRef(ashlarThickness);
  const ashlarPreviewSeedRef = useRef((Math.random() * 1e9) | 0);
  const rockCountRef = useRef(rockCount);
  const rockClusterRadiusRef = useRef(rockClusterRadius);
  const rockSinkDirectionRef = useRef(rockSinkDirection);
  const rockSinkAmountRef = useRef(rockSinkAmount);
  const grassDensityRef = useRef(grassDensity);
  const grassMaxHeightRef = useRef(grassMaxHeight);
  const rockPreviewSeedRef = useRef((Math.random() * 1e9) | 0);
  const grassPreviewSeedRef = useRef((Math.random() * 1e9) | 0);
  const floraPreviewSeedRef = useRef((Math.random() * 1e9) | 0);
  const piscinaPreviewSeedRef = useRef((Math.random() * 1e9) | 0);
  const roofStyleRef = useRef(roofStyle);
  const roofHeightRef = useRef(roofHeight);
  const roofHollowRef = useRef(roofHollow);
  useEffect(() => {
    ropeFirstScreenRef.current = ropeFirstScreen;
  }, [ropeFirstScreen]);
  useEffect(() => {
    ropeSagRef.current = ropeSag;
  }, [ropeSag]);
  useEffect(() => {
    ropeTensionRef.current = ropeTension;
  }, [ropeTension]);
  useEffect(() => {
    ropeBrushRadiusIndexRef.current = ropeBrushRadiusIndex;
  }, [ropeBrushRadiusIndex]);
  useEffect(() => {
    ropeBrushShapeUiRef.current = ropeBrushShapeUi;
  }, [ropeBrushShapeUi]);
  useEffect(() => {
    clothTensionRef.current = clothTension;
  }, [clothTension]);
  useEffect(() => {
    clothGravityDirectionRef.current = clothGravityDirection;
  }, [clothGravityDirection]);
  useEffect(() => {
    clothSimGravityPctRef.current = clothSimGravityPct;
  }, [clothSimGravityPct]);
  useEffect(() => {
    clothSimStiffnessPctRef.current = clothSimStiffnessPct;
  }, [clothSimStiffnessPct]);
  useEffect(() => {
    clothSimIterationsRef.current = clothSimIterations;
  }, [clothSimIterations]);
  useEffect(() => {
    clothSimConstraintPassesRef.current = clothSimConstraintPasses;
  }, [clothSimConstraintPasses]);
  useEffect(() => {
    rockRoughnessRef.current = rockRoughness;
  }, [rockRoughness]);
  useEffect(() => {
    ashlarThicknessRef.current = ashlarThickness;
  }, [ashlarThickness]);
  useEffect(() => {
    rockCountRef.current = rockCount;
  }, [rockCount]);
  useEffect(() => {
    rockClusterRadiusRef.current = rockClusterRadius;
  }, [rockClusterRadius]);
  useEffect(() => {
    rockSinkDirectionRef.current = rockSinkDirection;
  }, [rockSinkDirection]);
  useEffect(() => {
    rockSinkAmountRef.current = rockSinkAmount;
  }, [rockSinkAmount]);
  useEffect(() => {
    grassDensityRef.current = grassDensity;
  }, [grassDensity]);
  useEffect(() => {
    grassMaxHeightRef.current = grassMaxHeight;
  }, [grassMaxHeight]);
  useEffect(() => {
    roofStyleRef.current = roofStyle;
  }, [roofStyle]);
  useEffect(() => {
    roofHeightRef.current = roofHeight;
  }, [roofHeight]);
  useEffect(() => {
    roofHollowRef.current = roofHollow;
  }, [roofHollow]);
  useEffect(() => {
    squishyModeRef.current = squishyMode;
  }, [squishyMode]);
  useEffect(() => {
    void invoke("squishy_session_set_flags", {
      args: {
        hollow: squishyHollow,
        addSnapToSurface: squishySnapToSurface,
        wallThickness: Math.max(1, squishyWallThickness | 0),
      },
    }).catch(() => {});
  }, [squishyHollow, squishySnapToSurface, squishyWallThickness]);

  function mergedStrokeAux(base: Record<string, unknown> = {}): Record<string, unknown> {
    const sm = drawStrokeModeRef.current;
    const constrainToPlane =
      sm === "fill"
        ? fillConstrainToPlaneRef.current
        : sm === "spray"
          ? sprayConstrainToPlaneRef.current
          : false;
    const poly = strokePolygonVertsRef.current;
    const polygonVertices =
      (sm === "polygon" || sm === "polygonHull") && poly.length > 0
        ? poly.map((v) => [v[0], v[1], v[2]] as [number, number, number])
        : undefined;
    const out: Record<string, unknown> = {
      ...base,
      planeHollow: surfacePlaneHollowRef.current,
      constrainToPlane,
      spraySizeRange: spraySizeRangeRef.current,
      sprayScatter: sprayScatterRef.current,
      sprayRadiusMin: sprayRadiusMinRef.current,
      sprayRadiusMax: sprayRadiusMaxRef.current,
      sprayBrushShape: sm === "spray" ? sprayBrushShapeRef.current : undefined,
      constrainToPlaneRef: constrainToPlane ? sprayConstrainToPlaneRefRef.current : undefined,
      strokeFamilyVariant: strokeFamilyVariantRef.current,
      strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
      strokeAxisAlign: selectionStrokeAxisAlignRef.current,
      brushClipBottomHalf: brushClipBottomHalfRef.current,
      ...(polygonVertices != null && polygonVertices.length > 0 ? { polygonVertices } : {}),
    };
    if (sm === "cuboid" && cuboidPhase.ref.current) {
      out.cuboidDepth = cuboidDepthRef.current;
      out.cuboidHollowWallThickness = 1;
      const geo = cuboidPhase.ref.current.data.frozenGeo;
      if (geo) {
        out.cuboidFrozenA = geo.a;
        out.cuboidFrozenB = geo.b;
        out.cuboidFrozenPlaneAx = geo.planeAx;
        out.cuboidFrozenHit = geo.hit;
        out.cuboidFrozenPrev = geo.prev;
      }
    }
    if (sm === "cylinder" && cylinderPhase.ref.current) {
      out.cylinderDepth = cylinderDepthRef.current;
      out.cylinderTaperPct = 0;
      out.cuboidHollowWallThickness = 1;
      const geo = cylinderPhase.ref.current.data.frozenGeo;
      if (geo) {
        out.cuboidFrozenA = geo.a;
        out.cuboidFrozenB = geo.b;
        out.cuboidFrozenPlaneAx = geo.planeAx;
        out.cuboidFrozenHit = geo.hit;
        out.cuboidFrozenPrev = geo.prev;
      }
    }
    if ((sm === "polygon" || sm === "polygonHull") && polygonPhase.ref.current) {
      out.polygonDepth = polygonDepthRef.current;
    }
    return out;
  }

  useEffect(() => {
    cuboidDepthRef.current = cuboidDepthUi;
  }, [cuboidDepthUi]);

  useEffect(() => {
    cylinderDepthRef.current = cylinderDepthUi;
  }, [cylinderDepthUi]);

  function runDepthPhasePreview(
    endNorm: { nx: number; ny: number },
    lineStart: { nx: number; ny: number } | null,
    strokeMode: string,
    extraAux: Record<string, unknown> = {},
  ) {
    const im = interactionModeRef.current;
    const dispatch = getStrokeDispatch(im);
    if (!dispatch) return;
    if (dispatch.kind === "selection") {
      void runStrokeAtScreen(
        endNorm.nx,
        endNorm.ny,
        extraAux,
        lineStart ? { lineStart } : undefined,
      );
      return;
    }
    void invoke("voxel_stroke_preview_at_screen", {
      args: {
        nx: endNorm.nx,
        ny: endNorm.ny,
        tool: dispatch.tool,
        color: activeColorRef.current,
        material: activeMaterialRef.current,
        brushRadius: brushRadiusRef.current,
        brushShape: brushShapeRef.current,
        sprayDensity: sprayDensityRef.current,
        strokeMode,
        planeAxis: planeAxisRef.current,
        strokeAux: mergedStrokeAux(extraAux),
        matchMaterial: matchMaterialSelectColorRef.current,
        mirrorAxes:
          (mirrorXRef.current ? 1 : 0) |
          (mirrorYRef.current ? 2 : 0) |
          (mirrorZRef.current ? 4 : 0),
        ...(lineStart ? { strokeLineStartNx: lineStart.nx, strokeLineStartNy: lineStart.ny } : {}),
      },
    }).catch(() => {});
  }

  // Cuboid depth preview: re-invoke when phase data or depth changes.
  useEffect(() => {
    const snap = cuboidPhase.snapshot;
    if (!snap || loading || workBusy) return;
    const { lineStart, endNorm } = snap.data;
    runDepthPhasePreview(endNorm, lineStart, "cuboid");
  }, [cuboidPhase.snapshot, cuboidDepthUi, loading, workBusy, interactionMode]);

  // Cylinder depth preview: re-invoke when phase data or depth changes.
  useEffect(() => {
    const snap = cylinderPhase.snapshot;
    if (!snap || loading || workBusy) return;
    const { lineStart, endNorm } = snap.data;
    runDepthPhasePreview(endNorm, lineStart, "cylinder");
  }, [cylinderPhase.snapshot, cylinderDepthUi, loading, workBusy, interactionMode]);

  // Polygon solid depth preview: re-invoke when phase data or depth changes.
  useEffect(() => {
    const snap = polygonPhase.snapshot;
    if (!snap || loading || workBusy) return;
    const { endNorm } = snap.data;
    runDepthPhasePreview(endNorm, null, drawStrokeModeRef.current, {
      polygonDepth: polygonDepthRef.current,
    });
  }, [polygonPhase.snapshot, polygonDepthUi, loading, workBusy, interactionMode]);

  // Cancel extrude phase if interaction mode changes
  useEffect(() => {
    if (
      extrudePhase.active &&
      interactionMode !== "sculpt" &&
      interactionMode !== "selectExtrude"
    ) {
      extrudePhase.cancel();
    }
  }, [extrudePhase.active, interactionMode]);

  // Live-update extrude preview when settings change during the phase.
  useEffect(() => {
    if (!extrudePhase.active) return;
    // Selection extrude manages its own preview via selection_extrude_preview; recompute
    // is only relevant for the sculpt-mode ray extrude which stores a ray spine.
    if (interactionMode === "selectExtrude") return;
    void invoke("extrude_recompute_preview", {
      args: {
        extrudeProfile: extrudeProfileRef.current,
        extrudeEndCap: extrudeEndCapRef.current,
        extrudeTaper: extrudeTaperRef.current,
        extrudeTaperStart: extrudeTaperRef.current ? extrudeTaperStartRef.current : 0,
        extrudeTaperEnd: extrudeTaperRef.current ? extrudeTaperEndRef.current : 0,
      },
    }).catch(() => {});
  }, [
    extrudePhase.active,
    interactionMode,
    extrudeProfile,
    extrudeEndCap,
    extrudeTaper,
    extrudeTaperStart,
    extrudeTaperEnd,
  ]);

  // Cancel cuboid/cylinder depth phase if interaction mode becomes incompatible.
  useEffect(() => {
    if (!cuboidPhase.active && !cylinderPhase.active) return;
    const im = interactionMode;
    if (
      im !== "add" &&
      im !== "remove" &&
      im !== "paint" &&
      im !== "select" &&
      im !== "selectByColor" &&
      im !== "selectCoplanar" &&
      im !== "selectCoplanarEmpty"
    ) {
      if (cuboidPhase.active) cuboidPhase.cancel();
      if (cylinderPhase.active) cylinderPhase.cancel();
    }
  }, [interactionMode, cuboidPhase.active, cylinderPhase.active]);

  function commitDepthPhaseAtScreen(shape: "cuboid" | "cylinder") {
    const phase = shape === "cuboid" ? cuboidPhase : cylinderPhase;
    const snap = phase.ref.current;
    if (!snap) return;
    const { lineStart, endNorm } = snap.data;
    const depth = shape === "cuboid" ? cuboidDepthRef.current : cylinderDepthRef.current;
    const depthKey = shape === "cuboid" ? "cuboidDepth" : "cylinderDepth";
    runStrokeAtScreen(endNorm.nx, endNorm.ny, { [depthKey]: depth }, { lineStart });
    phase.cancel();
  }

  function commitCuboidSolidAtScreen() {
    commitDepthPhaseAtScreen("cuboid");
  }

  function commitCylinderSolidAtScreen() {
    commitDepthPhaseAtScreen("cylinder");
  }

  function commitPolygonSolid() {
    const snap = polygonPhase.ref.current;
    if (!snap) return;
    const { endNorm } = snap.data;
    const depth = polygonDepthRef.current;
    runStrokeAtScreen(endNorm.nx, endNorm.ny, {
      polygonVertices: strokePolygonVertsRef.current.map(
        (v) => [v[0], v[1], v[2]] as [number, number, number],
      ),
      polygonDepth: depth,
    });
    polygonPhase.cancel();
    setStrokePolygonVerts([]);
    strokePolygonVertsRef.current = [];
  }

  /** Payload for `sync_preview_input` — must match Rust `SyncPreviewInput` (camelCase). */
  function buildSyncPreviewPayload(_nx: number, _ny: number, modeStr: string) {
    // During rope settings phase, lock the hover position to the stored
    // second endpoint so the rope preview stays fixed while adjusting params.
    const rSnap = ropePhase.ref.current;
    // During any single-click generator settings phase, lock position to the clicked point.
    const genSnap =
      rocksPhase.ref.current ??
      grassPhase.ref.current ??
      ashlarPhase.ref.current ??
      floraPhase.ref.current ??
      piscinaPhase.ref.current ??
      insectaPhase.ref.current ??
      faunaPhase.ref.current;
    const nx = rSnap ? rSnap.data.nx2 : genSnap ? genSnap.data.nx : _nx;
    const ny = rSnap ? rSnap.data.ny2 : genSnap ? genSnap.data.ny : _ny;
    const im = interactionModeRef.current;
    const isSculpt = im === "sculpt";
    const isGenerator = im === "generator";
    const gk = generatorKindRef.current;
    const brushRadius = isGenerator
      ? gk === "rocks" || gk === "grass"
        ? Math.max(1, generatorSphereRadiusRef.current)
        : ropeBrushRadiusIndexRef.current
      : im === "squishy"
        ? Math.max(2, generatorSphereRadiusRef.current)
        : isSculpt
          ? sculptBrushRadiusRef.current
          : brushRadiusRef.current;
    const brushShape = isSculpt
      ? sculptBrushShapeToRust(sculptBrushShapeUiRef.current)
      : isGenerator
        ? sculptBrushShapeToRust(ropeBrushShapeUiRef.current)
        : brushShapeRef.current;
    const mirrorAxes =
      (mirrorXRef.current ? 1 : 0) | (mirrorYRef.current ? 2 : 0) | (mirrorZRef.current ? 4 : 0);
    return {
      nx,
      ny,
      mode: modeStr,
      brushRadius,
      brushShape,
      sprayDensity: isSculpt ? 0 : sprayDensityRef.current,
      strokeMode: isSculpt ? "precise" : drawStrokeModeRef.current,
      planeAxis: isSculpt ? "auto" : planeAxisRef.current,
      strokeAux: mergedStrokeAux({}),
      color: activeColorRef.current,
      mirrorAxes,
      ...(() => {
        const pal = selectedColorsRef.current;
        return pal.length > 1
          ? { palette: pal, paintColorDistrib: paintColorDistribRef.current }
          : {};
      })(),
      material: activeMaterialRef.current,
      matchMaterial: matchMaterialSelectColorRef.current,
      useBrushPreview: im !== "squishy" && !isGenerator,
      ...(isGenerator
        ? {
            generatorKind: gk,
            generatorRopeFirstNx: ropeFirstScreenRef.current?.nx,
            generatorRopeFirstNy: ropeFirstScreenRef.current?.ny,
            generatorRopeTension: ropeTensionRef.current,
            generatorRopeGravityDirection: clothGravityDirectionRef.current,
            generatorClothPins: clothPinsRef.current.map((p) => [p[0], p[1], p[2]]),
            generatorClothTension: clothTensionRef.current,
            generatorClothGravityDirection: clothGravityDirectionRef.current,
            generatorClothGravityScale: clothSimGravityPctRef.current / 100,
            generatorClothStiffnessScale: clothSimStiffnessPctRef.current / 100,
            generatorClothIterations: clothSimIterationsRef.current,
            generatorClothConstraintPasses: clothSimConstraintPassesRef.current,
            generatorRockSize: Math.max(1, generatorSphereRadiusRef.current),
            generatorRockRoughness: rockRoughnessRef.current,
            generatorRockSeed: rocksPhase.ref.current?.data.seed ?? rockPreviewSeedRef.current,
            generatorAshlarSize: Math.max(1, generatorSphereRadiusRef.current),
            generatorAshlarRoughness: rockRoughnessRef.current,
            generatorAshlarSeed: ashlarPhase.ref.current?.data.seed ?? ashlarPreviewSeedRef.current,
            generatorAshlarThickness: ashlarThicknessRef.current,
            generatorRockCount: rockCountRef.current,
            generatorRockClusterRadius: rockClusterRadiusRef.current,
            generatorRockSinkDirection:
              rockSinkDirectionRef.current === "under"
                ? -1
                : rockSinkDirectionRef.current === "over"
                  ? 1
                  : 0,
            generatorRockSinkAmount: rockSinkAmountRef.current,
            generatorGrassRadius: Math.max(1, generatorSphereRadiusRef.current),
            generatorGrassDensity: grassDensityRef.current,
            generatorGrassMaxHeight: grassMaxHeightRef.current,
            generatorGrassSeed: grassPhase.ref.current?.data.seed ?? grassPreviewSeedRef.current,
            generatorRoofPins: roofPinsRef.current.map((p) => [p[0], p[1], p[2]]),
            generatorRoofStyle: roofStyleRef.current,
            generatorRoofHeight: roofHeightRef.current,
            generatorRoofThickness: 1,
            generatorRoofBreakRatio: 0.5,
            generatorRoofWallHeight: 3,
            generatorRoofParapetHeight: 2,
            generatorRoofSaltSkew: 0,
            generatorRoofHollow: roofHollowRef.current,
            // Flora
            generatorFloraSeed: floraPhase.ref.current?.data.seed ?? floraPreviewSeedRef.current,
            generatorFloraHeight: floraHeight,
            generatorFloraGirth: floraGirth,
            generatorFloraWobble: floraWobble,
            generatorFloraTaper: floraTaper,
            generatorFloraStemCount: floraStemCount,
            generatorFloraClusterRadius: floraClusterRadius,
            generatorFloraBranchCount: floraBranchCount,
            generatorFloraBranchDepth: floraBranchDepth,
            generatorFloraBranchStart: floraBranchStart,
            generatorFloraBranchSpread: floraBranchSpread,
            generatorFloraBraidStrands: floraBraidStrands,
            generatorFloraBraidTwist: floraBraidTwist,
            generatorFloraCanopy: floraCanopy,
            // Insecta
            generatorInsectaSpecies: insectaSpecies,
            generatorInsectaTotalLength: insectaTotalLength,
            generatorInsectaHeadRatio: insectaHeadRatio,
            generatorInsectaThoraxRatio: insectaThoraxRatio,
            generatorInsectaAbdomenRatio: insectaAbdomenRatio,
            generatorInsectaBodyHalfWidth: insectaBodyHalfWidth,
            generatorInsectaBodyHalfHeight: insectaBodyHalfHeight,
            generatorInsectaAbdomenTaper: insectaAbdomenTaper,
            generatorInsectaHeadShape: insectaHeadShape,
            generatorInsectaAnchorOffsetU: insectaAnchorU,
            generatorInsectaAnchorOffsetV: insectaAnchorV,
            generatorInsectaBodyYaw: insectaBodyYawDeg * (Math.PI / 180),
            generatorInsectaBodyArch: insectaBodyArch,
            generatorInsectaAntennaLength: insectaAntennaLength,
            generatorInsectaAntennaSpread: insectaAntennaSpread,
            generatorInsectaAntennaPitch: insectaAntennaPitch,
            generatorInsectaAntennaRoot: insectaAntennaRoot,
            generatorInsectaMandibleLength: insectaMandibleLength,
            generatorInsectaMandibleSpread: insectaMandibleSpread,
            generatorInsectaMandibleForward: insectaMandibleForward,
            generatorInsectaWingShape: insectaWingShape,
            generatorInsectaShowWingFore: insectaShowWingFore,
            generatorInsectaWingForeLength: insectaWingForeLength,
            generatorInsectaWingForeWidth: insectaWingForeWidth,
            generatorInsectaWingForeSpread: insectaWingForeSpread,
            generatorInsectaWingForePitch: insectaWingForePitch,
            generatorInsectaWingForeOffset: insectaWingForeOffset,
            generatorInsectaWingForeForwardCant: insectaWingForeForwardCant,
            generatorInsectaShowWingHind: insectaShowWingHind,
            generatorInsectaWingHindLength: insectaWingHindLength,
            generatorInsectaWingHindWidth: insectaWingHindWidth,
            generatorInsectaWingHindSpread: insectaWingHindSpread,
            generatorInsectaWingHindPitch: insectaWingHindPitch,
            generatorInsectaWingHindOffset: insectaWingHindOffset,
            // Fauna
            generatorFaunaStance: faunaStance,
            generatorFaunaArchetype: faunaArchetype,
            generatorFaunaAnchorOffsetU: faunaAnchorU,
            generatorFaunaAnchorOffsetV: faunaAnchorV,
            generatorFaunaBodyYaw: faunaBodyYawDeg * (Math.PI / 180),
            generatorFaunaBodyArch: faunaBodyArch,
            generatorFaunaSpineSegments: faunaSpineSegments,
            generatorFaunaBodyLength: faunaBodyLength,
            generatorFaunaBodyHalfWidth: faunaBodyHalfWidth,
            generatorFaunaBodyHalfHeight: faunaBodyHalfHeight,
            generatorFaunaNeckLength: faunaNeckLength,
            generatorFaunaNeckHalfWidth: faunaNeckHalfWidth,
            generatorFaunaNeckHalfHeight: faunaNeckHalfHeight,
            generatorFaunaHeadLength: faunaHeadLength,
            generatorFaunaHeadHalfWidth: faunaHeadHalfWidth,
            generatorFaunaHeadHalfHeight: faunaHeadHalfHeight,
            generatorFaunaTailLength: faunaTailLength,
            generatorFaunaShoulderOffsetForward: faunaShoulderOffsetForward,
            generatorFaunaHipOffsetForward: faunaHipOffsetForward,
            generatorFaunaFrontUpperLength: faunaFrontUpperLength,
            generatorFaunaFrontLowerLength: faunaFrontLowerLength,
            generatorFaunaHindUpperLength: faunaHindUpperLength,
            generatorFaunaHindLowerLength: faunaHindLowerLength,
            generatorFaunaAutoFootPlacement: faunaAutoFootPlacement,
            // Piscina
            generatorPiscinaSeed:
              piscinaPhase.ref.current?.data.seed ?? piscinaPreviewSeedRef.current,
            generatorPiscinaSpecies: piscinaSpecies,
            generatorPiscinaLength: piscinaLength,
            generatorPiscinaWidth: piscinaWidth,
            generatorPiscinaThickness: piscinaThickness,
            generatorPiscinaSpineBend: piscinaSpineBend,
            generatorPiscinaSpineSCurve: piscinaSpineSCurve,
            generatorPiscinaFinDorsal: piscinaFinDorsal,
            generatorPiscinaFinAnal: piscinaFinAnal,
            generatorPiscinaFinCaudal: piscinaFinCaudal,
            generatorPiscinaFinPectoral: piscinaFinPectoral,
            generatorPiscinaFinPelvic: piscinaFinPelvic,
            generatorPiscinaFinAdipose: piscinaFinAdipose,
            generatorPiscinaShowFinDorsal: piscinaShowFinDorsal,
            generatorPiscinaShowFinAnal: piscinaShowFinAnal,
            generatorPiscinaShowFinCaudal: piscinaShowFinCaudal,
            generatorPiscinaShowFinPectoral: piscinaShowFinPectoral,
            generatorPiscinaShowFinPelvic: piscinaShowFinPelvic,
            generatorPiscinaShowFinAdipose: piscinaShowFinAdipose,
            generatorPiscinaAnchorOffsetU: piscinaAnchorU,
            generatorPiscinaAnchorOffsetV: piscinaAnchorV,
          }
        : {}),
      stampOriginX: stampOriginXRef.current,
      stampOriginZ: stampOriginZRef.current,
    };
  }

  /**
   * Unified stroke dispatch: calls `voxel_edit_at_screen` for edits or
   * `selection_stroke_at_screen` for selection, building shared args once.
   */
  function runStrokeAtScreen(
    nx: number,
    ny: number,
    strokeAux: Record<string, unknown>,
    opts?: {
      lineStart?: { nx: number; ny: number } | null;
      brushPrev?: { nx: number; ny: number } | null;
    },
  ): Promise<void> {
    const dispatch = getStrokeDispatch(interactionModeRef.current);
    if (!dispatch) return Promise.resolve();
    const lineStart = opts?.lineStart;
    const brushPrev = opts?.brushPrev;
    const isFill = drawStrokeModeRef.current === "fill";
    if (isFill) beginFillOperation();
    const mirrorAxes =
      (mirrorXRef.current ? 1 : 0) | (mirrorYRef.current ? 2 : 0) | (mirrorZRef.current ? 4 : 0);
    const sharedArgs: Record<string, unknown> = {
      nx,
      ny,
      brushRadius: brushRadiusRef.current,
      brushShape: brushShapeRef.current,
      sprayDensity: sprayDensityRef.current,
      strokeMode: drawStrokeModeRef.current,
      planeAxis: planeAxisRef.current,
      strokeAux: mergedStrokeAux(strokeAux),
      matchMaterial: matchMaterialSelectColorRef.current,
      fillSelectDiagonals: fillSelectDiagonalsRef.current,
      fillRespectsColor: fillRespectsColorRef.current,
      mirrorAxes,
      ...(lineStart
        ? {
            strokeLineStartNx: lineStart.nx,
            strokeLineStartNy: lineStart.ny,
          }
        : {}),
      ...(!lineStart && brushPrev
        ? {
            strokeSegmentPrevNx: brushPrev.nx,
            strokeSegmentPrevNy: brushPrev.ny,
          }
        : {}),
    };
    if (dispatch.kind === "edit") {
      const palette = selectedColorsRef.current;
      const multiColor = palette.length > 1;
      const editArgs = {
        ...sharedArgs,
        tool: dispatch.tool,
        color: activeColorRef.current,
        ...(multiColor
          ? {
              palette,
              paintColorDistrib: paintColorDistribRef.current,
              strokeSeed: currentStrokeSeedRef.current,
            }
          : {}),
        material: activeMaterialRef.current,
      };
      return invoke("voxel_edit_at_screen", { args: editArgs })
        .catch(async (e: unknown) => {
          if (isFill && typeof e === "string" && e === "confirm_large_fill") {
            if (await askFillConfirmation()) {
              beginFillOperation();
              return invoke("voxel_edit_at_screen", {
                args: { ...editArgs, confirmed: true },
              }).catch((e2: unknown) => {
                console.error("[voxelle] voxel_edit_at_screen error", e2);
              });
            }
            return;
          }
          console.error("[voxelle] voxel_edit_at_screen error", e);
        })
        .finally(() => {
          if (isFill) endFillOperation();
        }) as Promise<void>;
    } else {
      const selArgs = {
        ...sharedArgs,
        interaction: dispatch.interaction,
        ...(strokeShiftKeyRef.current ? { combineModeOverride: "add" } : {}),
      };
      return invoke<number>("selection_stroke_at_screen", { args: selArgs })
        .then((n) => {
          if (n > 0) {
            void invoke<number>("selection_get_count").then((c) => setSelectionCount(c));
          }
        })
        .catch(async (e: unknown) => {
          if (isFill && typeof e === "string" && e === "confirm_large_fill") {
            if (await askFillConfirmation()) {
              beginFillOperation();
              return invoke<number>("selection_stroke_at_screen", {
                args: { ...selArgs, confirmed: true },
              })
                .then((n) => {
                  if (n > 0) {
                    void invoke<number>("selection_get_count").then((c) => setSelectionCount(c));
                  }
                })
                .catch(() => {});
            }
            return;
          }
        })
        .finally(() => {
          if (isFill) endFillOperation();
        });
    }
  }

  /** Specialized selection single-click commands (selectByColor, selectCoplanar, selectCoplanarEmpty). */
  function invokeSelectionSpecialClick(interaction: string, nx: number, ny: number) {
    const cmd =
      interaction === "selectByColor"
        ? "selection_add_by_color_at_screen"
        : interaction === "selectCoplanar"
          ? "selection_add_coplanar_at_screen"
          : interaction === "selectCoplanarEmpty"
            ? "selection_add_coplanar_empty_at_screen"
            : null;
    if (!cmd) return;
    const args: Record<string, unknown> = { nx, ny };
    if (interaction === "selectByColor") {
      args.matchMaterial = matchMaterialSelectColorRef.current;
    }
    if (strokeShiftKeyRef.current) {
      args.combineModeOverride = "add";
    }
    void invoke<number>(cmd, { args })
      .then((n) => {
        if (n > 0) {
          void invoke<number>("selection_get_count").then((c) => setSelectionCount(c));
        }
      })
      .catch(() => {});
  }

  async function handleStrokeAnchorClick(nx: number, ny: number) {
    const dispatch = getStrokeDispatch(interactionModeRef.current);
    if (!dispatch) return;
    const tool = dispatch.kind === "edit" ? dispatch.tool : "remove";
    const c = await invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
      args: {
        nx,
        ny,
        tool,
        strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
      },
    });
    if (!c) return;
    const sm = drawStrokeModeRef.current;
    if (sm === "fill") {
      runStrokeAtScreen(nx, ny, {});
      return;
    }
    if (sm === "polygon" || sm === "polygonHull") {
      setStrokePolygonVerts((v) => {
        const idx = v.findIndex((p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2]);
        const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
        strokePolygonVertsRef.current = next;
        return next;
      });
      strokePolygonLastScreenRef.current = { nx, ny };
      queueMicrotask(() => {
        void invoke("sync_preview_input", {
          args: buildSyncPreviewPayload(nx, ny, previewModeForSync(interactionModeRef.current)),
        }).catch(() => {});
      });
      return;
    }
    if (sm === "circle") {
      const r = strokeClickRef.current;
      if (!r.circleCenter) {
        r.circleCenter = c;
      } else {
        runStrokeAtScreen(nx, ny, {
          circleCenter: r.circleCenter,
          circleEdge: c,
        });
        r.circleCenter = null;
      }
      return;
    }
  }

  async function handleWallSculptPolygonClick(nx: number, ny: number) {
    const c = await invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
      args: {
        nx,
        ny,
        tool: "add",
        strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
      },
    });
    if (!c) return;
    setWallSculptPolygonVerts((v) => {
      const idx = v.findIndex((p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2]);
      const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
      wallSculptPolygonVertsRef.current = next;
      return next;
    });
  }

  async function handleClothPinClick(nx: number, ny: number) {
    const c = await invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
      args: {
        nx,
        ny,
        tool: "add",
        strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
      },
    });
    if (!c) return;
    setClothPins((v) => {
      const idx = v.findIndex((p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2]);
      const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
      clothPinsRef.current = next;
      return next;
    });
  }

  function applyPolygonStrokeFill() {
    if (strokePolygonVerts.length < 3) return;
    const scr = strokePolygonLastScreenRef.current ?? lastViewportPickNormRef.current;
    const nx = scr?.nx ?? 0;
    const ny = scr?.ny ?? 0;
    if (strokeFamilyVariantRef.current === "solid") {
      polygonDepthRef.current = 1;
      setPolygonDepthUi(1);
      polygonPhase.enter("depth", { endNorm: { nx, ny } });
      return;
    }
    runStrokeAtScreen(nx, ny, {
      polygonVertices: strokePolygonVerts.map((v) => [v[0], v[1], v[2]]),
    });
  }

  useEffect(() => {
    sprayDensityRef.current = sprayDensity;
  }, [sprayDensity]);
  useEffect(() => {
    fillSelectDiagonalsRef.current = fillSelectDiagonals;
  }, [fillSelectDiagonals]);
  useEffect(() => {
    fillRespectsColorRef.current = fillRespectsColor;
  }, [fillRespectsColor]);
  useEffect(() => {
    surfacePlaneHollowRef.current = surfacePlaneHollow;
  }, [surfacePlaneHollow]);
  useEffect(() => {
    sprayConstrainToPlaneRef.current = sprayConstrainToPlane;
  }, [sprayConstrainToPlane]);
  useEffect(() => {
    spraySizeRangeRef.current = spraySizeRange;
  }, [spraySizeRange]);
  useEffect(() => {
    sprayScatterRef.current = sprayScatter;
  }, [sprayScatter]);
  useEffect(() => {
    sprayRadiusMinRef.current = sprayRadiusMin;
  }, [sprayRadiusMin]);
  useEffect(() => {
    sprayRadiusMaxRef.current = sprayRadiusMax;
  }, [sprayRadiusMax]);
  useEffect(() => {
    sprayBrushShapeRef.current = sprayBrushShape;
  }, [sprayBrushShape]);
  useEffect(() => {
    sprayConstrainToPlaneRefRef.current = sprayConstrainToPlaneRef_;
  }, [sprayConstrainToPlaneRef_]);
  useEffect(() => {
    fillConstrainToPlaneRef.current = fillConstrainToPlane;
  }, [fillConstrainToPlane]);
  useEffect(() => {
    if (interactionMode === "fly") {
      setToolsPane("fly");
      return;
    }
    if (interactionMode === "walk") {
      setToolsPane("walk");
      return;
    }
    if (interactionMode === "navigate") {
      setToolsPane("hand");
      return;
    }
    if (interactionMode === "sculpt") {
      setToolsPane("sculpt");
      return;
    }
    if (interactionMode === "generator") {
      setToolsPane("generators");
      return;
    }
    if (interactionMode === "squishy") {
      setToolsPane("squishy");
      return;
    }
    if (
      interactionMode === "add" ||
      interactionMode === "remove" ||
      interactionMode === "paint" ||
      interactionMode === "eyedropper" ||
      interactionMode === "select" ||
      interactionMode === "selectByColor" ||
      interactionMode === "selectCoplanar" ||
      interactionMode === "selectCoplanarEmpty" ||
      interactionMode === "stamp" ||
      interactionMode === "punch"
    ) {
      setToolsPane("draw");
    }
  }, [interactionMode]);

  useEffect(() => {
    if (
      selectionCount === 0 &&
      !stampBookPatternActive &&
      (interactionMode === "stamp" || interactionMode === "punch")
    ) {
      setInteractionMode("add");
    }
    // When user makes a new selection, clear the book stamp pattern
    if (selectionCount > 0 && stampBookPatternActive) {
      setStampBookPatternActive(false);
    }
  }, [selectionCount, interactionMode, stampBookPatternActive]);

  useEffect(() => {
    // Don't overwrite a book-loaded stamp clipboard with an empty selection
    if (!stampBookPatternActive && (interactionMode === "stamp" || interactionMode === "punch")) {
      void invoke("clipboard_copy_selection").catch(() => {});
    }
  }, [interactionMode, stampBookPatternActive]);

  // Cancel single-click generator placements when switching away from generator mode.
  useEffect(() => {
    if (interactionMode !== "generator") {
      rocksPhase.cancel();
      grassPhase.cancel();
      ashlarPhase.cancel();
      floraPhase.cancel();
      piscinaPhase.cancel();
      insectaPhase.cancel();
      faunaPhase.cancel();
    }
  }, [interactionMode]);

  const previewModeForSync = (m: InteractionMode): string => {
    if (m === "add") return "add";
    if (m === "remove") return "remove";
    if (m === "paint") return "paint";
    if (m === "sculpt") return "add";
    if (m === "generator") return "add";
    if (m === "stamp") return "stamp";
    if (m === "punch") return "punch";
    if (m === "fly") return "fly";
    if (m === "walk") return "walk";
    if (m === "squishy") return "squishy";
    if (
      m === "select" ||
      m === "selectByColor" ||
      m === "selectCoplanar" ||
      m === "selectCoplanarEmpty"
    ) {
      return "select";
    }
    if (m === "selectExtrude") return "selectExtrude";
    return "navigate";
  };

  /**
   * Normalized viewport coords of pointer down for strokes that need a drag origin.
   * Line style always; brush style for plane/circle/cuboid/etc. (Surface/Solid), but not Spray
   * (Spray uses segment prev only). Without this, Rust never sees `strokeLineStart*` for Surface.
   */
  function strokeViewportLineStartNorm(): { nx: number; ny: number } | null {
    const start = strokeViewportStartRef.current;
    if (!start) return null;
    if (strokeDrawStyleRef.current === "line") return start;
    if (strokeDrawStyleRef.current === "brush" && drawStrokeModeRef.current !== "spray") {
      return start;
    }
    return null;
  }

  function beginFillOperation() {
    fillOperationPendingRef.current = true;
    setFillOperationPending(true);
  }

  function endFillOperation() {
    fillOperationPendingRef.current = false;
    setFillOperationPending(false);
  }

  /** Show a non-blocking React confirmation modal and resolve true/false. */
  function askFillConfirmation(): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
      setPendingFillConfirm({
        resolve: (confirmed: boolean) => {
          setPendingFillConfirm(null);
          resolve(confirmed);
        },
      });
    });
  }

  useLayoutEffect(() => {
    applyAppearanceToDocument(loadPreferences().appearanceTheme);
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const onSchemeChange = () => {
      if (loadPreferences().appearanceTheme === "auto") {
        applyAppearanceToDocument("auto");
      }
    };
    mq.addEventListener("change", onSchemeChange);
    return () => mq.removeEventListener("change", onSchemeChange);
  }, []);

  useEffect(() => {
    loadingRef.current = loading;
    interactionBlockedRef.current = loading || workBusy || fillOperationPending;
  }, [loading, workBusy, fillOperationPending]);

  /** Escape cancels in-progress flood fill (Rust BFS checks `fill_operation_cancel`). */
  useEffect(() => {
    if (!workBusy && !fillOperationPending) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Escape") return;
      if (e.repeat) return;
      if (!fillOperationPendingRef.current && !/fill/i.test(workPhaseRef.current)) return;
      e.preventDefault();
      void invoke("voxel_fill_cancel").catch(() => {});
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [workBusy, fillOperationPending]);

  /** Any path that sets `loadError` must not leave `loading` stuck true (e.g. `collab-error`, invoke `.catch`). */
  useEffect(() => {
    if (loadError != null) setLoading(false);
  }, [loadError]);

  useEffect(() => {
    const p = loadPreferences();
    const next = preferencesWithCollabIdentity(p, displayName, accentColor);
    if (
      next.collabDisplayName === p.collabDisplayName &&
      next.collabAccentColor === p.collabAccentColor
    )
      return;
    savePreferences(next);
  }, [displayName, accentColor]);

  useEffect(() => {
    const p = loadPreferences();
    const n = normalizeCollabHostPort(hostPort);
    if (p.collabHostPort === n) return;
    savePreferences({ ...p, collabHostPort: n });
  }, [hostPort]);

  /** Keep roster / chat labels in sync when name or accent changes mid-session. */
  useEffect(() => {
    if (!collabActive) return;
    const rgb = hexToRgb(normalizeCollabAccentColor(accentColor));
    void invoke("collab_update_profile", {
      displayName: normalizeCollabDisplayName(displayName),
      colorRgb: rgb,
    }).catch(() => {});
  }, [displayName, accentColor, collabActive]);

  useEffect(() => {
    localStorage.setItem(LS_SIDEBAR_EXPANDED, sidebarExpanded ? "1" : "0");
  }, [sidebarExpanded]);

  useEffect(() => {
    localStorage.setItem(LS_RIGHT_SIDEBAR_EXPANDED, rightSidebarExpanded ? "1" : "0");
  }, [rightSidebarExpanded]);

  useEffect(() => {
    localStorage.setItem(LS_TOOLS_FLOATING, toolsPaneFloating ? "1" : "0");
  }, [toolsPaneFloating]);

  useEffect(() => {
    try {
      localStorage.setItem(
        LS_TOOLS_FLOAT_POS,
        JSON.stringify({ x: toolPanePos.x, y: toolPanePos.y }),
      );
    } catch {
      /* ignore */
    }
  }, [toolPanePos]);

  useEffect(() => {
    localStorage.setItem(LS_PALETTE_FLOATING, colorPaletteFloating ? "1" : "0");
  }, [colorPaletteFloating]);

  useEffect(() => {
    localStorage.setItem(
      LS_PALETTE_FLOAT_POS,
      JSON.stringify({ x: colorPalettePos.x, y: colorPalettePos.y }),
    );
  }, [colorPalettePos]);

  useEffect(() => {
    localStorage.setItem(
      LS_PALETTE_FLOAT_SIZE,
      JSON.stringify({ w: colorPaletteSize.w, h: colorPaletteSize.h }),
    );
  }, [colorPaletteSize]);

  useEffect(() => {
    const p = loadPreferences();
    void invoke("set_autosave_settings", autosaveSettingsInvokeArgs(p)).catch(() => {});
  }, []);

  useEffect(() => {
    void invoke<LastSessionInfo>("get_last_session_info")
      .then((info) => setLastSessionInfo(info))
      .catch(() => setLastSessionInfo(null))
      .finally(() => setLastSessionReady(true));
  }, []);

  useEffect(() => {
    if (!lastSessionReady) return;
    const p = loadPreferences();
    if (!p.reopenLastProject) {
      setPendingAutoReopen(false);
      return;
    }
    if (!lastSessionInfo?.lastDocumentPath) {
      setPendingAutoReopen(false);
      return;
    }
    const info = lastSessionInfo;
    const doc = info.lastDocumentPath;
    const auto = info.autosavePath;
    const useAutosave =
      info.autosaveExists &&
      auto != null &&
      auto !== "" &&
      (!info.documentExists || info.autosaveNewerThanDocument);
    if (useAutosave) {
      void invoke("load_voxelle_recovery", {
        args: { documentPath: doc, autosavePath: auto },
      }).catch((err) => {
        setLoadError(err instanceof Error ? err.message : String(err));
        setPendingAutoReopen(false);
      });
    } else if (info.documentExists) {
      void invoke("load_voxelle_path", { path: doc }).catch((err) => {
        setLoadError(err instanceof Error ? err.message : String(err));
        setPendingAutoReopen(false);
      });
    } else {
      setPendingAutoReopen(false);
    }
  }, [lastSessionReady, lastSessionInfo]);

  useEffect(() => {
    const p = loadPreferences();
    if (p.hdr) {
      void invoke("set_hdr_output", { enabled: true })
        .then(() => invoke("set_tone_mapping", { mode: 6 }))
        .catch(() => {});
    } else {
      void invoke("set_tone_mapping", {
        mode: toneMappingToGpuMode(p.toneMapping),
      }).catch(() => {});
    }
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(LS_RENDERING_MODE) as RenderingMode | null;
    const valid = saved && ["greedy", "marchingCubes", "dualContour"].includes(saved);
    void invoke<RenderingMode>("get_rendering_mode")
      .then((m) => {
        if (valid && saved !== m) {
          void invoke("set_rendering_mode", { mode: saved }).catch(() => {});
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!collabActive) return;
    const id = window.setInterval(() => {
      void invoke("collab_push_camera").catch(() => {});
    }, 150);
    return () => clearInterval(id);
  }, [collabActive]);

  useEffect(() => {
    if (!chatPanelOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setChatPanelOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chatPanelOpen]);

  // Fire a ping at the given normalized viewport coords with an optional emoji.
  const firePing = useCallback((p: { nx: number; ny: number }, emoji?: string) => {
    const dn = loadPreferences().collabDisplayName.trim();
    void invoke<{
      ok: boolean;
      x?: number;
      y?: number;
      z?: number;
    }>("ping_cursor_pick", {
      args: { nx: p.nx, ny: p.ny, displayName: dn, emoji: emoji ?? "" },
    })
      .then((r) => {
        if (!r?.ok || r.x == null || r.y == null || r.z == null) return;
        const name = dn.length > 0 ? dn : "You";
        pingHudRef.current = {
          name,
          wx: r.x + 0.5,
          wy: r.y + 0.5,
          wz: r.z + 0.5,
          until: Date.now() + PING_HUD_MS,
          emoji: emoji || undefined,
        };
        setPingHudTick((n) => n + 1);
        playPingSound();
        void invoke("collab_send_ping", {
          x: r.x,
          y: r.y,
          z: r.z,
          emoji: emoji ?? "",
        }).catch(() => {});
      })
      .catch(() => {});
  }, []);

  // Handle radial menu selection (emoji chosen or null = cancelled)
  const onRadialSelect = useCallback(
    (emoji: string | null) => {
      setRadialMenu((m) => ({ ...m, visible: false }));
      const p = pendingPingRef.current;
      if (!p) return;
      pendingPingRef.current = null;
      if (emoji) {
        firePing(p, emoji);
      }
    },
    [firePing],
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          void invoke("voxel_redo").catch(() => {});
        } else {
          void invoke("voxel_undo").catch(() => {});
        }
        return;
      }
      if (meta && e.key === "s") {
        e.preventDefault();
        void invoke("save_voxelle").catch(() => {
          void invoke("save_voxelle_as").catch(() => {});
        });
        return;
      }
      if (e.key !== "z" && e.key !== "Z") return;
      if (meta) return;
      if (e.repeat) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
        return;
      }
      if (preferencesOpen || stampBookOpen || joinModalOpen || newProjectOpen || collabJoinPending)
        return;
      const p = lastViewportPickNormRef.current;
      if (!p) return;
      e.preventDefault();
      // Stash the pick coords for the radial menu / quick-tap path
      pendingPingRef.current = { nx: p.nx, ny: p.ny };
      const scr = lastCursorScreenRef.current;
      // Start hold timer — if Z is held long enough, show radial menu
      if (radialHoldTimerRef.current) clearTimeout(radialHoldTimerRef.current);
      radialHoldTimerRef.current = setTimeout(() => {
        radialHoldTimerRef.current = null;
        setRadialMenu({ x: scr.x, y: scr.y, visible: true });
      }, RADIAL_HOLD_MS);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key !== "z" && e.key !== "Z") return;
      // If the hold timer is still pending, it was a quick tap → fire normal ping
      if (radialHoldTimerRef.current) {
        clearTimeout(radialHoldTimerRef.current);
        radialHoldTimerRef.current = null;
        const p = pendingPingRef.current;
        pendingPingRef.current = null;
        if (p) firePing(p);
      }
      // If radial menu is visible, RadialPingMenu handles keyup via onSelect
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      if (radialHoldTimerRef.current) {
        clearTimeout(radialHoldTimerRef.current);
        radialHoldTimerRef.current = null;
      }
    };
  }, [preferencesOpen, stampBookOpen, joinModalOpen, newProjectOpen, collabJoinPending, firePing]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return;
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          t.isContentEditable)
      ) {
        return;
      }
      if (
        preferencesOpen ||
        stampBookOpen ||
        joinModalOpen ||
        newProjectOpen ||
        collabJoinPending
      ) {
        return;
      }
      if (loading || workBusy) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (selectionCount === 0) return;
      if (e.code === "KeyX") {
        e.preventDefault();
        e.stopPropagation();
        void invoke<number>("selection_delete_selected_voxels").catch(() => {});
        return;
      }
      // Arrow keys: translate (plain) or rotate 90° (Shift+arrow).
      // X/Z plane: ← → move X. Shift+← → rotate around Y.
      // Y axis: no arrow for Y translate; use Shift+↑↓ for rotate around X/Z.
      const arrowMap: Record<string, [number, number, number]> = {
        ArrowLeft: [-1, 0, 0],
        ArrowRight: [1, 0, 0],
        ArrowUp: [0, 0, -1],
        ArrowDown: [0, 0, 1],
      };
      const rotateMap: Record<string, [number, number]> = {
        ArrowLeft: [1, -1],
        ArrowRight: [1, 1],
        ArrowUp: [0, -1],
        ArrowDown: [0, 1],
      };
      if (!e.shiftKey && arrowMap[e.code]) {
        e.preventDefault();
        e.stopPropagation();
        const [dx, dy, dz] = arrowMap[e.code];
        void invoke("selection_translate", { dx, dy, dz }).catch(() => {});
        return;
      }
      if (e.shiftKey && rotateMap[e.code]) {
        e.preventDefault();
        e.stopPropagation();
        const [axis, quarters] = rotateMap[e.code];
        void invoke("selection_rotate", { axis, quarters }).catch(() => {});
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    selectionCount,
    preferencesOpen,
    stampBookOpen,
    joinModalOpen,
    newProjectOpen,
    collabJoinPending,
    loading,
    workBusy,
  ]);

  const clearPreview = useCallback(() => {
    void invoke("sync_preview_input", {
      args: buildSyncPreviewPayload(-1, 0, "navigate"),
    }).catch(() => {});
    void invoke("squishy_gizmo_pointer_up").catch(() => {});
  }, []);

  const releaseFlyMouseLook = useCallback(async () => {
    flyMouseLookActiveRef.current = false;
    flyLastClientRef.current = null;
    flySkipNextFlyMoveRef.current = false;
    flyPendingLookDxRef.current = 0;
    flyPendingLookDyRef.current = 0;
    flyCapturedPointerIdRef.current = null;
    // Release pointer lock if active
    if (document.pointerLockElement) {
      try {
        document.exitPointerLock();
      } catch {
        /* */
      }
    }
    // Release Tauri-native cursor grab/visibility
    const w = getCurrentWindow();
    try {
      await w.setCursorGrab(false);
    } catch {
      /* e.g. Linux: grab unsupported */
    }
    try {
      await w.setCursorVisible(true);
    } catch {
      /* */
    }
  }, []);

  const activateFlyMouseLook = useCallback(async (_pointerId: number) => {
    const el = viewportRef.current;
    console.log("[walk-debug] activateFlyMouseLook called, el=", !!el);
    if (!el) return;
    flySkipNextFlyMoveRef.current = false;
    flyPendingLookDxRef.current = 0;
    flyPendingLookDyRef.current = 0;
    flyCapturedPointerIdRef.current = null;
    // Request pointer lock FIRST — must be called synchronously from user gesture
    // before any awaits, otherwise the browser drops the gesture context.
    try {
      await el.requestPointerLock();
      console.log(
        "[walk-debug] requestPointerLock succeeded, pointerLockElement=",
        document.pointerLockElement === el,
      );
    } catch (err) {
      console.warn("[walk-debug] requestPointerLock FAILED:", err);
    }
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    // Tauri-native fallback: grab + hide cursor if pointer lock didn't engage
    if (document.pointerLockElement !== el) {
      const w = getCurrentWindow();
      try {
        await w.setCursorPosition(new LogicalPosition(cx, cy));
      } catch {
        /* */
      }
      try {
        await w.setCursorGrab(true);
      } catch {
        /* Linux: unsupported */
      }
      try {
        await w.setCursorVisible(false);
      } catch {
        /* */
      }
    }
    flyLastClientRef.current = { x: cx, y: cy };
    flyMouseLookActiveRef.current = true;
  }, []);

  useEffect(() => {
    void invoke("sync_preview_input", {
      args: buildSyncPreviewPayload(-1, 0, previewModeForSync(interactionMode)),
    }).catch(() => {});
  }, [interactionMode]);

  /** Re-push brush/stroke params so hover preview updates when sliders change without moving the pointer. */
  useEffect(() => {
    const im = interactionModeRef.current;
    if (
      im !== "add" &&
      im !== "remove" &&
      im !== "paint" &&
      im !== "sculpt" &&
      im !== "select" &&
      im !== "selectByColor" &&
      im !== "selectCoplanar" &&
      im !== "selectCoplanarEmpty"
    ) {
      return;
    }
    if (interactionBlockedRef.current) return;
    const p = lastViewportPickNormRef.current;
    if (p == null) return;
    void invoke("sync_preview_input", {
      args: buildSyncPreviewPayload(p.nx, p.ny, previewModeForSync(im)),
    }).catch(() => {});
  }, [
    brushRadius,
    brushShape,
    drawStrokeMode,
    sprayDensity,
    planeAxis,
    surfacePlaneHollow,
    sprayConstrainToPlane,
    spraySizeRange,
    fillConstrainToPlane,
    activeColor,
    activeMaterial,
    matchMaterialSelectColor,
    brushClipBottomHalf,
    sculptBrushRadius,
    sculptBrushShapeUi,
    strokePolygonVerts,
  ]);

  /** Squishy: re-sync metaball preview when radius / hollow / mode change without moving the pointer. */
  useEffect(() => {
    const im = interactionModeRef.current;
    if (im !== "squishy") return;
    if (interactionBlockedRef.current) return;
    const p = lastViewportPickNormRef.current;
    if (p == null) return;
    void invoke("sync_preview_input", {
      args: buildSyncPreviewPayload(p.nx, p.ny, "squishy"),
    }).catch(() => {});
  }, [
    squishyMode,
    generatorSphereRadius,
    squishyHollow,
    squishyWallThickness,
    squishySnapToSurface,
  ]);

  /** Generators: hover + rope/cloth volume preview when parameters change without moving the pointer. */
  useEffect(() => {
    const im = interactionModeRef.current;
    if (im !== "generator") return;
    if (interactionBlockedRef.current) return;
    // During rope settings phase, buildSyncPreviewPayload overrides nx/ny to
    // the stored second endpoint, so even a dummy (0,0) works.  For cloth the
    // preview is pin-based and doesn't use hover coords.
    const p = lastViewportPickNormRef.current ?? { nx: 0, ny: 0 };
    void invoke("sync_preview_input", {
      args: buildSyncPreviewPayload(p.nx, p.ny, previewModeForSync(im)),
    }).catch(() => {});
  }, [
    generatorKind,
    ropeFirstScreen,
    ropeSag,
    ropeTension,
    ropeBrushRadiusIndex,
    ropeBrushShapeUi,
    clothPins,
    clothTension,
    clothGravityDirection,
    clothSimGravityPct,
    clothSimStiffnessPct,
    clothSimIterations,
    clothSimConstraintPasses,
    generatorSphereRadius,
    activeColor,
    rockRoughness,
    ashlarThickness,
    rockCount,
    rockClusterRadius,
    rockSinkDirection,
    rockSinkAmount,
    grassDensity,
    grassMaxHeight,
    roofPins,
    roofStyle,
    roofHeight,
    roofHollow,
  ]);

  useEffect(() => {
    if (interactionMode !== "squishy" || squishyMode !== "edit") {
      void invoke("squishy_gizmo_pointer_up").catch(() => {});
    }
  }, [interactionMode, squishyMode]);

  useEffect(() => {
    matchMaterialSelectColorRef.current = matchMaterialSelectColor;
  }, [matchMaterialSelectColor]);

  useEffect(() => {
    void invoke("selection_menu_sync_match_material", {
      checked: matchMaterialSelectColor,
    }).catch(() => {});
  }, [matchMaterialSelectColor]);

  useEffect(() => {
    void invoke("set_mood_params", { args: mood }).catch(() => {});
  }, [mood]);

  useEffect(() => {
    if (interactionMode !== "fly") {
      void invoke("set_fly_mode", { enabled: false }).catch(() => {});
      keysDownRef.current.clear();
      void releaseFlyMouseLook();
      return;
    }
    void invoke("set_fly_mode", { enabled: true }).catch(() => {});
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.code === "Escape" && flyMouseLookActiveRef.current) {
        void releaseFlyMouseLook();
        e.preventDefault();
        return;
      }
      keysDownRef.current.add(e.code);
      if (
        e.code === "KeyW" ||
        e.code === "KeyS" ||
        e.code === "KeyA" ||
        e.code === "KeyD" ||
        e.code === "KeyE" ||
        e.code === "KeyQ" ||
        e.code === "ShiftLeft" ||
        e.code === "ShiftRight"
      ) {
        e.preventDefault();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      keysDownRef.current.delete(e.code);
      if (
        e.code === "KeyW" ||
        e.code === "KeyS" ||
        e.code === "KeyA" ||
        e.code === "KeyD" ||
        e.code === "KeyE" ||
        e.code === "KeyQ" ||
        e.code === "ShiftLeft" ||
        e.code === "ShiftRight"
      ) {
        e.preventDefault();
      }
    };
    const dpr = () => window.devicePixelRatio || 1;
    const onFlyPointerMove = (e: PointerEvent) => {
      const vp = viewportRef.current;
      const s = dpr();
      if (!flyMouseLookActiveRef.current || !vp) return;

      // When pointer lock is active, movementX/Y give raw deltas directly —
      // no need to recenter or skip synthetic events.
      if (document.pointerLockElement === vp) {
        const dxCss = e.movementX;
        const dyCss = e.movementY;
        if (dxCss === 0 && dyCss === 0) return;
        flyPendingLookDxRef.current += dxCss * s;
        flyPendingLookDyRef.current += dyCss * s;
        return;
      }

      // Fallback: manual recentering when pointer lock is unavailable
      if (flySkipNextFlyMoveRef.current) {
        flySkipNextFlyMoveRef.current = false;
        flyLastClientRef.current = { x: e.clientX, y: e.clientY };
        return;
      }
      let dxCss = e.movementX;
      let dyCss = e.movementY;
      if (dxCss === 0 && dyCss === 0) {
        const last = flyLastClientRef.current;
        if (last == null) {
          flyLastClientRef.current = { x: e.clientX, y: e.clientY };
          return;
        }
        dxCss = e.clientX - last.x;
        dyCss = e.clientY - last.y;
        flyLastClientRef.current = { x: e.clientX, y: e.clientY };
        if (dxCss === 0 && dyCss === 0) return;
      } else {
        flyLastClientRef.current = { x: e.clientX, y: e.clientY };
      }
      flyPendingLookDxRef.current += dxCss * s;
      flyPendingLookDyRef.current += dyCss * s;
      const r = vp.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      void getCurrentWindow()
        .setCursorPosition(new LogicalPosition(cx, cy))
        .then(() => {
          flySkipNextFlyMoveRef.current = true;
          flyLastClientRef.current = { x: cx, y: cy };
        })
        .catch(() => {});
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("pointermove", onFlyPointerMove, true);
    const tick = () => {
      const pdx = flyPendingLookDxRef.current;
      const pdy = flyPendingLookDyRef.current;
      flyPendingLookDxRef.current = 0;
      flyPendingLookDyRef.current = 0;
      if (pdx !== 0 || pdy !== 0) {
        void invoke("camera_fly_look", {
          args: { dx: pdx, dy: pdy },
        }).catch(() => {});
      }
      const k = keysDownRef.current;
      let forward = 0;
      let right = 0;
      let up = 0;
      if (k.has("KeyW")) forward += 1;
      if (k.has("KeyS")) forward -= 1;
      if (k.has("KeyD")) right += 1;
      if (k.has("KeyA")) right -= 1;
      if (k.has("KeyE")) up += 1;
      if (k.has("KeyQ")) up -= 1;
      const slow = k.has("ShiftLeft") || k.has("ShiftRight");
      const speedScale = (slow ? 1 / 8 : 1) * flySpeedRef.current;
      void invoke("sync_fly_input", {
        args: { forward, right, up, speedScale },
      }).catch(() => {});
      // Recenter cursor each frame when using Tauri fallback (not pointer lock)
      if (flyMouseLookActiveRef.current && !document.pointerLockElement) {
        const vp = viewportRef.current;
        if (vp) {
          const r = vp.getBoundingClientRect();
          const cx = r.left + r.width / 2;
          const cy = r.top + r.height / 2;
          void getCurrentWindow()
            .setCursorPosition(new LogicalPosition(cx, cy))
            .then(() => {
              flySkipNextFlyMoveRef.current = true;
              flyLastClientRef.current = { x: cx, y: cy };
            })
            .catch(() => {});
        }
      }
      flyRafRef.current = requestAnimationFrame(tick);
    };
    flyRafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(flyRafRef.current);
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      document.removeEventListener("pointermove", onFlyPointerMove, true);
      void invoke("set_fly_mode", { enabled: false }).catch(() => {});
      void releaseFlyMouseLook();
    };
  }, [interactionMode, releaseFlyMouseLook]);

  // ── Walk mode: first-person with gravity, collision, jumping ──
  useEffect(() => {
    if (interactionMode !== "walk") {
      void invoke("set_walk_mode", { enabled: false }).catch(() => {});
      keysDownRef.current.clear();
      void releaseFlyMouseLook();
      return;
    }
    console.log("[walk-debug] walk useEffect SETUP — activating walk mode");
    void invoke("set_walk_mode", { enabled: true })
      .then(() => {
        console.log("[walk-debug] set_walk_mode(true) resolved OK");
      })
      .catch((err) => {
        console.error("[walk-debug] set_walk_mode(true) FAILED:", err);
      });
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.code === "Escape" && flyMouseLookActiveRef.current) {
        void releaseFlyMouseLook();
        e.preventDefault();
        return;
      }
      keysDownRef.current.add(e.code);
      if (
        e.code === "KeyW" ||
        e.code === "KeyS" ||
        e.code === "KeyA" ||
        e.code === "KeyD" ||
        e.code === "Space" ||
        e.code === "ShiftLeft" ||
        e.code === "ShiftRight"
      ) {
        e.preventDefault();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      keysDownRef.current.delete(e.code);
      if (
        e.code === "KeyW" ||
        e.code === "KeyS" ||
        e.code === "KeyA" ||
        e.code === "KeyD" ||
        e.code === "Space" ||
        e.code === "ShiftLeft" ||
        e.code === "ShiftRight"
      ) {
        e.preventDefault();
      }
    };
    const dpr = () => window.devicePixelRatio || 1;
    const onWalkPointerMove = (e: PointerEvent) => {
      const vp = viewportRef.current;
      const s = dpr();
      if (!flyMouseLookActiveRef.current || !vp) return;
      if (document.pointerLockElement === vp) {
        const dxCss = e.movementX;
        const dyCss = e.movementY;
        if (dxCss === 0 && dyCss === 0) return;
        flyPendingLookDxRef.current += dxCss * s;
        flyPendingLookDyRef.current += dyCss * s;
        return;
      }
      // Fallback: manual recentering when pointer lock is unavailable
      if (flySkipNextFlyMoveRef.current) {
        flySkipNextFlyMoveRef.current = false;
        flyLastClientRef.current = { x: e.clientX, y: e.clientY };
        return;
      }
      let dxCss = e.movementX;
      let dyCss = e.movementY;
      if (dxCss === 0 && dyCss === 0) {
        const last = flyLastClientRef.current;
        if (last == null) {
          flyLastClientRef.current = { x: e.clientX, y: e.clientY };
          return;
        }
        dxCss = e.clientX - last.x;
        dyCss = e.clientY - last.y;
        flyLastClientRef.current = { x: e.clientX, y: e.clientY };
        if (dxCss === 0 && dyCss === 0) return;
      } else {
        flyLastClientRef.current = { x: e.clientX, y: e.clientY };
      }
      flyPendingLookDxRef.current += dxCss * s;
      flyPendingLookDyRef.current += dyCss * s;
      const r = vp.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      void getCurrentWindow()
        .setCursorPosition(new LogicalPosition(cx, cy))
        .then(() => {
          flySkipNextFlyMoveRef.current = true;
          flyLastClientRef.current = { x: cx, y: cy };
        })
        .catch(() => {});
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("pointermove", onWalkPointerMove, true);
    const tick = () => {
      const pdx = flyPendingLookDxRef.current;
      const pdy = flyPendingLookDyRef.current;
      flyPendingLookDxRef.current = 0;
      flyPendingLookDyRef.current = 0;
      if (pdx !== 0 || pdy !== 0) {
        void invoke("camera_fly_look", {
          args: { dx: pdx, dy: pdy },
        }).catch(() => {});
      }
      const k = keysDownRef.current;
      let forward = 0;
      let right = 0;
      if (k.has("KeyW")) forward += 1;
      if (k.has("KeyS")) forward -= 1;
      if (k.has("KeyD")) right += 1;
      if (k.has("KeyA")) right -= 1;
      const jump = k.has("Space");
      const slow = k.has("ShiftLeft") || k.has("ShiftRight");
      const speedScale = slow ? 1 / 3 : 1;
      void invoke("sync_fly_input", {
        args: { forward, right, up: 0, speedScale, jump },
      }).catch(() => {});
      // Recenter cursor each frame when using Tauri fallback (not pointer lock)
      if (flyMouseLookActiveRef.current && !document.pointerLockElement) {
        const vp = viewportRef.current;
        if (vp) {
          const r = vp.getBoundingClientRect();
          const cx = r.left + r.width / 2;
          const cy = r.top + r.height / 2;
          void getCurrentWindow()
            .setCursorPosition(new LogicalPosition(cx, cy))
            .then(() => {
              flySkipNextFlyMoveRef.current = true;
              flyLastClientRef.current = { x: cx, y: cy };
            })
            .catch(() => {});
        }
      }
      flyRafRef.current = requestAnimationFrame(tick);
    };
    flyRafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(flyRafRef.current);
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      document.removeEventListener("pointermove", onWalkPointerMove, true);
      void invoke("set_walk_mode", { enabled: false }).catch(() => {});
      void releaseFlyMouseLook();
    };
  }, [interactionMode, releaseFlyMouseLook]);

  useEffect(() => {
    if (
      interactionMode !== "select" &&
      interactionMode !== "selectByColor" &&
      interactionMode !== "selectCoplanar" &&
      interactionMode !== "selectCoplanarEmpty"
    )
      return;
    void invoke<number>("selection_get_count")
      .then((n) => setSelectionCount(n))
      .catch(() => {});
  }, [interactionMode]);

  useEffect(() => {
    if (loading || workBusy) {
      clearPreview();
    }
  }, [loading, workBusy, clearPreview]);

  /**
   * Normalized coords (0–1) in the GPU viewport texture. The native layer stretches the W×H texture to
   * the `.viewport` CSS box, so linear fractions `relX/rect.width` and `relY/rect.height` match what
   * hits the eye — unlike `(clientY/innerHeight)*surfaceH − viewportY`, which breaks when `inner*` includes
   * the scrollbar gutter and does not match `documentElement.client*` / `clientY` (common on macOS windowed
   * WebKit; often disappears in fullscreen where inner ≈ client).
   *
   * **Full-window GPU viewport** (experimental): texture covers the full swapchain; normalize with
   * `layoutViewportCssSize` so the denominator matches `clientX`/`clientY` (same as `sendResize`).
   */
  const clientToViewportNormalized = useCallback((e: React.PointerEvent) => {
    const el = viewportRef.current;
    if (!el) return { nx: 0.5, ny: 0.5 };
    const rect = el.getBoundingClientRect();
    const rw = rect.width;
    const rh = rect.height;
    if (rw <= 0 || rh <= 0) return { nx: 0.5, ny: 0.5 };

    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;
    return {
      nx: Math.min(1, Math.max(0, relX / rw)),
      ny: Math.min(1, Math.max(0, relY / rh)),
    };
  }, []);

  const planeStrokeDebugEnabledRef = useRef(true);
  const logPlaneStrokeDebug = useCallback(
    (_phase: string, _e: React.PointerEvent, _extra?: Record<string, unknown>) => {
      if (!planeStrokeDebugEnabledRef.current) return;
      const mode = interactionModeRef.current;
      const sm = drawStrokeModeRef.current;
      if (!(sm === "plane" && (mode === "add" || mode === "remove" || mode === "paint"))) {
        return;
      }
      void gestureRef.current;
    },
    [],
  );

  const resetPointerGesture = useCallback(
    (reason: string, e?: React.PointerEvent) => {
      if (e) {
        logPlaneStrokeDebug(`gesture:reset:${reason}`, e);
      }
      gestureRef.current = null;
      activePointerIdRef.current = null;
      pointerStartRef.current = null;
      maxPointerMoveRef.current = 0;
      pendingPointerUpRef.current = null;
    },
    [logPlaneStrokeDebug],
  );

  useEffect(() => {
    if (newProjectOpen) {
      const p = loadPreferences();
      setNewGridSize(p.newProjectDefaultSize);
      setNewGridShape(p.newProjectDefaultShape);
    }
  }, [newProjectOpen]);

  const createNewProject = useCallback(() => {
    if (loading || workBusy) return;
    let size = Math.floor(Number(newGridSize));
    if (!Number.isFinite(size)) size = 32;
    size = Math.max(1, Math.min(MAX_GRID_SIZE, size));
    setNewGridSize(size);
    setNewProjectOpen(false);
    void invoke("create_new_project", {
      args: { gridSize: size, shape: newGridShape },
    }).catch((err) => {
      setLoadError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    });
  }, [loading, workBusy, newGridSize, newGridShape]);

  useEffect(() => {
    const w = getCurrentWindow();
    if (loadError) {
      void w.setTitle("Voxelle Desktop");
      return;
    }
    const name = pathLabel ? basename(pathLabel) : "";
    if (loading && name) {
      void w.setTitle(`Loading… ${name} — Voxelle Desktop`);
    } else if (name) {
      void w.setTitle(`${name} — Voxelle Desktop`);
    } else {
      void w.setTitle("Voxelle Desktop");
    }
  }, [pathLabel, loading, loadError]);

  const onPointerDown = async (e: React.PointerEvent) => {
    logPlaneStrokeDebug("down:received", e);
    const modeEarly = interactionModeRef.current;
    if ((modeEarly === "fly" || modeEarly === "walk") && (e.button === 0 || e.button === 2)) {
      console.log(
        "[walk-debug] pointer-down in",
        modeEarly,
        "mode, button=",
        e.button,
        "mouseLookActive=",
        flyMouseLookActiveRef.current,
      );
      e.preventDefault();
      if (flyMouseLookActiveRef.current) {
        void releaseFlyMouseLook();
      } else {
        void activateFlyMouseLook(e.pointerId);
      }
      probingRef.current = false;
      resetPointerGesture("fly-toggle", e);
      return;
    }

    const captureEl = e.currentTarget as HTMLElement;
    try {
      captureEl.setPointerCapture(e.pointerId);
      capturedPointerIdRef.current = e.pointerId;
    } catch (err) {
      capturedPointerIdRef.current = null;
      console.warn("[voxelle][plane-stroke] setPointerCapture failed", err);
    }
    activePointerIdRef.current = e.pointerId;
    pointerStartRef.current = { x: e.clientX, y: e.clientY };
    maxPointerMoveRef.current = 0;
    probingRef.current = true;
    gestureRef.current = null;

    // Extrude settings phase: re-drag to reposition endpoint instead of cancelling.
    if (
      extrudePhase.ref.current &&
      interactionModeRef.current === "sculpt" &&
      sculptStrokeModeRef.current === "extrude" &&
      e.button === 0
    ) {
      extrudeRedragRef.current = true;
      probingRef.current = false;
      return;
    }
    // Cancel extrude phase on left-click — but NOT for selectExtrude, where
    // we defer to the gizmo hit-test below so re-clicking a handle doesn't commit.
    if (
      extrudePhase.ref.current &&
      e.button === 0 &&
      interactionModeRef.current !== "selectExtrude"
    ) {
      extrudePhase.cancel();
    }

    const { nx, ny } = clientToViewportNormalized(e);
    const pointerId = e.pointerId;
    const shiftKey = e.shiftKey;
    const middleButton = e.button === 1;
    const mode = interactionModeRef.current;
    const navigate = mode === "navigate" || mode === "fly" || mode === "walk";
    const forceCamera =
      middleButton ||
      (mode === "add" && e.button !== 0) ||
      (mode === "remove" && e.button !== 0) ||
      (mode === "paint" && e.button !== 0) ||
      (mode === "eyedropper" && e.button !== 0) ||
      (mode === "select" && e.button !== 0) ||
      (mode === "selectByColor" && e.button !== 0) ||
      (mode === "selectCoplanar" && e.button !== 0) ||
      (mode === "selectCoplanarEmpty" && e.button !== 0) ||
      (mode === "stamp" && e.button !== 0 && e.button !== 2) ||
      (mode === "punch" && e.button !== 0 && e.button !== 2) ||
      (mode === "sculpt" && e.button !== 0) ||
      (mode === "generator" && e.button !== 0 && e.button !== 2) ||
      (mode === "squishy" && e.button !== 0);

    const logoSplashPointer =
      startScreenLogoLoadedRef.current && !loading && !workBusy && e.button === 0;

    if (
      mode === "squishy" &&
      squishyModeRef.current === "edit" &&
      e.button === 0 &&
      !loading &&
      !workBusy &&
      !logoSplashPointer
    ) {
      try {
        const consumed = await invoke<boolean>("squishy_gizmo_pointer_down", {
          args: { nx, ny },
        });
        if (consumed) {
          probingRef.current = false;
          gestureRef.current = { pointerId, mode: "squishyGizmo" };
          lastRef.current = { x: e.clientX, y: e.clientY };
          return;
        }
      } catch {
        /* fall through to pick / camera */
      }
    }

    // Extrude gizmo: check in selectExtrude mode before falling through to camera.
    if (
      e.button === 0 &&
      !loading &&
      !workBusy &&
      !logoSplashPointer &&
      !navigate &&
      !forceCamera &&
      mode === "selectExtrude"
    ) {
      try {
        const hit = await extrudeGizmoRef.current?.startDragIfHit(e.clientX, e.clientY);
        if (hit) {
          probingRef.current = false;
          gestureRef.current = { pointerId, mode: "extrudeGizmo" };
          lastRef.current = { x: e.clientX, y: e.clientY };
          return;
        }
      } catch {
        /* fall through */
      }
      // Gizmo wasn't hit — cancel the settings phase if active.
      if (extrudePhase.ref.current) {
        extrudePhase.cancel();
      }
    }

    // Selection gizmo: check before pick probe so arrow/ring drags don't fall through.
    // Exclude selectExtrude — in that mode we use the extrude gizmo instead.
    if (
      e.button === 0 &&
      !loading &&
      !workBusy &&
      !logoSplashPointer &&
      !navigate &&
      !forceCamera &&
      mode !== "selectExtrude"
    ) {
      try {
        const hit = await gizmoRef.current?.startDragIfHit(e.clientX, e.clientY);
        if (hit) {
          probingRef.current = false;
          gestureRef.current = { pointerId, mode: "selectionGizmo" };
          lastRef.current = { x: e.clientX, y: e.clientY };
          return;
        }
      } catch {
        /* fall through */
      }
    }

    // Stamp/punch right-click is handled in onContextMenu (fires reliably on all platforms).
    // Still return early here to avoid setting a camera gesture if pointerdown does fire.
    if ((mode === "stamp" || mode === "punch") && e.button === 2) {
      probingRef.current = false;
      resetPointerGesture("stamp-rotate-passthrough", e);
      return;
    }

    // Generator right-click: reseed preview (web parity)

    if (mode === "generator" && e.button === 2 && !loading && !workBusy) {
      e.preventDefault();
      rockPreviewSeedRef.current = (Math.random() * 1e9) | 0;
      ashlarPreviewSeedRef.current = (Math.random() * 1e9) | 0;
      floraPreviewSeedRef.current = (Math.random() * 1e9) | 0;
      piscinaPreviewSeedRef.current = (Math.random() * 1e9) | 0;
      // Trigger a preview sync so the new seed is sent to Rust
      const p = lastViewportPickNormRef.current ?? { nx: 0, ny: 0 };
      void invoke("sync_preview_input", {
        args: buildSyncPreviewPayload(p.nx, p.ny, previewModeForSync(mode)),
      }).catch(() => {});
      probingRef.current = false;
      resetPointerGesture("generator-reseed", e);
      return;
    }

    let hitSolid = false;
    const isDrawOrSelect =
      !logoSplashPointer &&
      !loading &&
      !workBusy &&
      !forceCamera &&
      !navigate &&
      (mode === "add" ||
        mode === "remove" ||
        mode === "paint" ||
        mode === "eyedropper" ||
        mode === "select" ||
        mode === "selectByColor" ||
        mode === "selectCoplanar" ||
        mode === "selectCoplanarEmpty" ||
        mode === "stamp" ||
        mode === "punch" ||
        mode === "sculpt" ||
        mode === "generator" ||
        mode === "squishy") &&
      e.button === 0;
    // All draw/select modes run the async pick probe. Pointer-up events that
    // arrive while probing are deferred and replayed after the probe resolves
    // (see pendingPointerUpRef), which fixes both the "click twice" race and
    // the "can't orbit on empty space" bug for all tools including fill and
    // selection modes.
    if (isDrawOrSelect) {
      try {
        hitSolid = await invoke<boolean>("voxel_pick_probe", {
          args: { nx, ny },
        });
      } catch {
        hitSolid = false;
      }
    }

    probingRef.current = false;

    if (activePointerIdRef.current !== pointerId) {
      return;
    }

    gestureRef.current = {
      pointerId,
      mode: forceCamera || navigate || !hitSolid ? "camera" : "voxel",
    };
    logPlaneStrokeDebug("down:gesture-assigned", e, {
      hitSolid,
      forceCamera,
      navigate,
      assignedGestureMode: gestureRef.current.mode,
    });
    lastRef.current = { x: e.clientX, y: e.clientY };

    if (gestureRef.current.mode === "voxel") {
      const dispatch = getStrokeDispatch(mode);
      const isSculptOrDispatch = dispatch || mode === "sculpt";
      if (isSculptOrDispatch) {
        if (drawStrokeModeRef.current === "cuboid" && cuboidPhase.ref.current) {
          cuboidPhase.cancel();
        }
        if (drawStrokeModeRef.current === "cylinder" && cylinderPhase.ref.current) {
          cylinderPhase.cancel();
        }
        dragDidEditRef.current = false;
        strokeViewportStartRef.current = { nx, ny };
        lastStrokeNormRef.current = { nx, ny };
        currentStrokeSeedRef.current = Math.floor(Math.random() * 0xffffffff) >>> 0;
        strokeShiftKeyRef.current = shiftKey;
        if (dispatch?.kind === "selection") {
          selectionStrokeBegunRef.current = true;
          void invoke("selection_stroke_begin").catch(() => {});
        } else {
          void invoke("voxel_stroke_begin").catch(() => {});
          // Immediately refresh the preview so the correct stroke seed and
          // line-start are reflected on click without requiring mouse movement.
          if (
            mode === "sculpt" &&
            sculptStrokeModeRef.current !== "extrude" &&
            !loading &&
            !workBusy
          ) {
            void invoke("voxel_sculpt_stroke_preview_at_screen", {
              args: buildSculptStrokeInvokeArgs(nx, ny, {
                strokeSegmentPrev: { nx, ny },
              }),
            }).catch(() => {});
          }
        }
      }
      // Roof square/circle: resolve first anchor for drag-to-define.
      if (
        mode === "generator" &&
        generatorKindRef.current === "roof" &&
        (roofAreaShapeRef.current === "square" || roofAreaShapeRef.current === "circle")
      ) {
        dragDidEditRef.current = false;
        void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
          args: {
            nx,
            ny,
            tool: "add",
            strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
          },
        })
          .then((c) => {
            if (c) {
              roofFirstClickRef.current = c;
              setRoofFirstClick(c);
            }
          })
          .catch(() => {});
      }
    }

    if (gestureRef.current.mode === "camera" && mode !== "fly") {
      void invoke("viewport_pointer", {
        ev: {
          kind: "down",
          nx,
          ny,
          dx: 0,
          dy: 0,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey,
        },
      });
    }

    // If pointer-up arrived while the probe was in-flight, replay it now that
    // the gesture is fully established.
    const pendingUp = pendingPointerUpRef.current;
    if (pendingUp && pendingUp.pointerId === pointerId) {
      pendingPointerUpRef.current = null;
      onPointerUpRef.current?.(pendingUp);
    }
  };

  /** Shared sculpt stroke payload for preview and apply (matches Rust `SculptStrokeAtScreenArgs`). */
  function buildSculptStrokeInvokeArgs(
    nx: number,
    ny: number,
    opts: {
      strokeSegmentPrev?: { nx: number; ny: number } | null;
      includeStrokeSeed?: boolean;
    } = {},
  ) {
    const sm = sculptStrokeModeRef.current;
    const includeStrokeSeed = opts.includeStrokeSeed !== false;
    const lineStart =
      sm === "wall" &&
      (wallAreaShapeRef.current === "circle" || wallAreaShapeRef.current === "brush") &&
      strokeViewportStartRef.current
        ? {
            strokeLineStartNx: strokeViewportStartRef.current.nx,
            strokeLineStartNy: strokeViewportStartRef.current.ny,
          }
        : {};
    const wallPoly =
      sm === "wall" &&
      wallAreaShapeRef.current === "polygon" &&
      wallSculptPolygonVertsRef.current.length >= 2
        ? {
            wallPolygonVertices: wallSculptPolygonVertsRef.current.map((v) => [v[0], v[1], v[2]]),
          }
        : {};
    const seg = opts.strokeSegmentPrev
      ? {
          strokeSegmentPrevNx: opts.strokeSegmentPrev.nx,
          strokeSegmentPrevNy: opts.strokeSegmentPrev.ny,
        }
      : {};
    return {
      nx,
      ny,
      sculptMode: sm,
      color: activeColorRef.current,
      material: activeMaterialRef.current,
      brushRadius: sculptBrushRadiusRef.current,
      brushShape: sculptBrushShapeToRust(sculptBrushShapeUiRef.current),
      sprayDensity: 0,
      brushClipBottomHalf: brushClipBottomHalfRef.current,
      ...seg,
      ...(sm === "terrain"
        ? {
            terrainOp: terrainSculptOpRef.current,
            terrainBaseY: terrainBaseYRef.current,
            terrainStrength: sculptBrushStrengthRef.current,
            terrainSmoothRadius: terrainSmoothRadiusRef.current,
            terrainFlattenUseBaseY: terrainFlattenUseBaseYRef.current,
            terrainSubVoxel: terrainSubVoxelRef.current,
          }
        : {}),
      ...(sm === "smooth"
        ? {
            smoothNeighborPasses: sculptSmoothPassesRef.current,
          }
        : {}),
      brushStrength: sculptBrushStrengthRef.current,
      brushFalloff: sculptBrushFalloffRef.current,
      ...(includeStrokeSeed
        ? {
            strokeSeed: Math.floor(Math.random() * 0x1_0000_0000) >>> 0,
          }
        : {}),
      wallAreaShape: wallAreaShapeRef.current,
      sprayDirection: sprayDirectionRef.current,
      wallWidthIndex: wallWidthIndexRef.current,
      wallHeightVox: wallHeightVoxRef.current,
      wallLockStartHeight: wallLockStartHeightRef.current,
      wallAxisAlign: wallAxisAlignRef.current,
      sculptSmoothVariant: sculptSmoothVariantRef.current,
      smoothNeighborRadius: smoothNeighborRadiusRef.current,
      smoothAggressiveness: smoothAggressivenessRef.current,
      smoothLaplacianIterations: smoothLaplacianIterationsRef.current,
      smoothLaplacianRelaxPct: smoothLaplacianRelaxPctRef.current,
      extrudeProfile: extrudeProfileRef.current,
      extrudeEndCap: extrudeEndCapRef.current,
      extrudeTaper: extrudeTaperRef.current,
      extrudeTaperStart: extrudeTaperRef.current ? extrudeTaperStartRef.current : 0,
      extrudeTaperEnd: extrudeTaperRef.current ? extrudeTaperEndRef.current : 0,
      ...lineStart,
      ...wallPoly,
    };
  }

  function commitWallSculptPolygonStroke() {
    const verts = wallSculptPolygonVertsRef.current;
    if (verts.length < 2) return;
    const scr = lastViewportPickNormRef.current ?? { nx: 0, ny: 0 };
    void invoke("voxel_sculpt_stroke_at_screen", {
      args: buildSculptStrokeInvokeArgs(scr.nx, scr.ny, {
        includeStrokeSeed: true,
      }),
    }).catch(() => {});
    setWallSculptPolygonVerts([]);
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const { nx: px, ny: py } = clientToViewportNormalized(e);
    lastViewportPickNormRef.current = { nx: px, ny: py };
    lastCursorScreenRef.current = { x: e.clientX, y: e.clientY };
    if (viewportCursorDebugEnabled) {
      const el = viewportRef.current;
      const rect = el?.getBoundingClientRect();
      const lv = layoutViewportCssSize();
      const scr: ViewportCursorDebugScreen | null = rect
        ? {
            clientX: e.clientX,
            clientY: e.clientY,
            relX: e.clientX - rect.left,
            relY: e.clientY - rect.top,
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight,
            layoutWidth: lv.w,
            layoutHeight: lv.h,
            rectLeft: rect.left,
            rectTop: rect.top,
            rectWidth: rect.width,
            rectHeight: rect.height,
          }
        : null;
      viewportCursorDebugScreenRef.current = scr;
      setViewportCursorDebugScreen(scr);
      setViewportCursorDebugJs({ nx: px, ny: py });
      if (viewportCursorDebugRafRef.current == null) {
        viewportCursorDebugRafRef.current = requestAnimationFrame(() => {
          viewportCursorDebugRafRef.current = null;
          void invoke<ViewportCursorDebugPayload>("get_viewport_cursor_debug")
            .then((d) => {
              setViewportCursorDebugRust(d);
              // #region agent log
              const vel = viewportRef.current;
              const wrap = vel?.parentElement;
              const rV = vel?.getBoundingClientRect();
              const rW = wrap?.getBoundingClientRect();
              const phys = viewportPhysRef.current;
              const surf = surfacePhysRef.current;
              const scrSnap = viewportCursorDebugScreenRef.current;
              const iw = scrSnap?.layoutWidth ?? layoutViewportCssSize().w;
              const ih = scrSnap?.layoutHeight ?? layoutViewportCssSize().h;
              fetch("http://127.0.0.1:7756/ingest/93734617-b27b-4379-bb59-e5971936c3d4", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  "X-Debug-Session-Id": "0e537f",
                },
                body: JSON.stringify({
                  sessionId: "0e537f",
                  runId: "rect-mapping",
                  hypothesisId: "H_innerY_vs_relY",
                  location: "App.tsx:viewportDebugRaf",
                  message: "pick uses rel/rect; compare inner*surface−origin Y vs relY/rh",
                  data: (() => {
                    const scr = viewportCursorDebugScreenRef.current;
                    const pick = lastViewportPickNormRef.current;
                    let nxFromRel: number | null = null;
                    let nxWindow: number | null = null;
                    let deltaWinVsRel: number | null = null;
                    let nyFromRel: number | null = null;
                    let nyFromInner: number | null = null;
                    let deltaInnerVsRelNy: number | null = null;
                    if (
                      scr &&
                      rV &&
                      rV.width > 0 &&
                      rV.height > 0 &&
                      phys.w > 0 &&
                      phys.h > 0 &&
                      iw > 0 &&
                      ih > 0 &&
                      surf.w > 0 &&
                      surf.h > 0
                    ) {
                      nxFromRel = scr.relX / rV.width;
                      nyFromRel = scr.relY / rV.height;
                      const ox = Math.max(0, Math.round((rV.left / iw) * surf.w));
                      const oy = Math.max(0, Math.round((rV.top / ih) * surf.h));
                      nxWindow = ((scr.clientX / iw) * surf.w - ox) / phys.w;
                      deltaWinVsRel = nxWindow - nxFromRel;
                      nyFromInner = ((scr.clientY / ih) * surf.h - oy) / phys.h;
                      deltaInnerVsRelNy = nyFromInner - nyFromRel;
                    }
                    return {
                      viewportRw: rV?.width,
                      viewportRh: rV?.height,
                      wrapRw: rW?.width,
                      wrapRh: rW?.height,
                      rectDeltaW: rV && rW ? rV.width - rW.width : null,
                      rectDeltaH: rV && rW ? rV.height - rW.height : null,
                      aspectDom: rV && rV.height > 0 ? rV.width / rV.height : null,
                      physW: phys.w,
                      physH: phys.h,
                      aspectPhys: phys.h > 0 ? phys.w / phys.h : null,
                      rustW: d.viewportWidth,
                      rustH: d.viewportHeight,
                      aspectRust: d.viewportHeight > 0 ? d.viewportWidth / d.viewportHeight : null,
                      surfaceW: surf.w,
                      surfaceH: surf.h,
                      vwPerRw: rV && rV.width > 0 ? phys.w / rV.width : null,
                      swPerIw: iw > 0 ? surf.w / iw : null,
                      shPerIh: ih > 0 ? surf.h / ih : null,
                      nxFromRel,
                      nxWindow,
                      deltaWinVsRel,
                      nyFromRel,
                      nyFromInner,
                      deltaInnerVsRelNy,
                      nxPick: pick?.nx ?? null,
                      nyPick: pick?.ny ?? null,
                      deltaPickVsRelNx: pick && nxFromRel != null ? pick.nx - nxFromRel : null,
                      deltaPickVsRelNy: pick && nyFromRel != null ? pick.ny - nyFromRel : null,
                    };
                  })(),
                  timestamp: Date.now(),
                }),
              }).catch(() => {});
              // #endregion
            })
            .catch(() => setViewportCursorDebugRust(null));
        });
      }
    }
    if (
      gestureRef.current?.mode === "squishyGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      void invoke("squishy_gizmo_pointer_move", {
        args: { nx: px, ny: py },
      }).catch(() => {});
      return;
    }
    if (
      gestureRef.current?.mode === "selectionGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      const last = lastRef.current;
      const cx = e.clientX;
      const cy = e.clientY;
      gizmoRef.current?.pointerMove(cx, cy, last.x, last.y);
      lastRef.current = { x: cx, y: cy };
      return;
    }
    if (
      gestureRef.current?.mode === "extrudeGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      const last = lastRef.current;
      const cx = e.clientX;
      const cy = e.clientY;
      extrudeGizmoRef.current?.pointerMove(
        cx,
        cy,
        last.x,
        last.y,
        activeColorRef.current,
        activeMaterialRef.current,
      );
      dragDidEditRef.current = true;
      lastRef.current = { x: cx, y: cy };
      return;
    }
    if (!gestureRef.current) {
      if (interactionModeRef.current === "selectExtrude") {
        extrudeGizmoRef.current
          ?.updateHover(e.clientX, e.clientY)
          .then((h) => {
            gizmoHoverRef.current = h ?? false;
          })
          .catch(() => {
            gizmoHoverRef.current = false;
          });
      } else {
        gizmoRef.current
          ?.updateHover(e.clientX, e.clientY)
          .then((h) => {
            gizmoHoverRef.current = h ?? false;
          })
          .catch(() => {
            gizmoHoverRef.current = false;
          });
      }
    } else {
      gizmoHoverRef.current = false;
    }
    const overGizmo = !gestureRef.current && gizmoHoverRef.current;
    const anyGenPhaseActive =
      rocksPhase.active ||
      grassPhase.active ||
      ashlarPhase.active ||
      floraPhase.active ||
      piscinaPhase.active ||
      insectaPhase.active ||
      faunaPhase.active;
    if (
      !overGizmo &&
      !probingRef.current &&
      (interactionModeRef.current === "add" ||
        interactionModeRef.current === "remove" ||
        interactionModeRef.current === "paint" ||
        interactionModeRef.current === "sculpt" ||
        interactionModeRef.current === "select" ||
        interactionModeRef.current === "selectByColor" ||
        interactionModeRef.current === "selectCoplanar" ||
        interactionModeRef.current === "selectCoplanarEmpty" ||
        interactionModeRef.current === "squishy" ||
        interactionModeRef.current === "generator" ||
        interactionModeRef.current === "stamp" ||
        interactionModeRef.current === "punch" ||
        interactionModeRef.current === "selectExtrude") &&
      !interactionBlockedRef.current &&
      !anyGenPhaseActive
    ) {
      const m = previewModeForSync(interactionModeRef.current);
      void invoke("sync_preview_input", {
        args: buildSyncPreviewPayload(px, py, m),
      }).catch(() => {});
    } else if (overGizmo) {
      // Preserve selectExtrude mode when hovering the extrude gizmo so the
      // GPU gizmo continues rendering in extrude style (balls, no rings).
      const hoverMode =
        interactionModeRef.current === "selectExtrude" ? "selectExtrude" : "navigate";
      void invoke("sync_preview_input", {
        args: buildSyncPreviewPayload(-1, 0, hoverMode),
      }).catch(() => {});
    }

    // Wall brush hover preview: show the full wall footprint under the cursor before any drag.
    // Pass strokeLineStart = current position so Rust uses a zero-length line anchor (single
    // surface voxel) and treats the union as non-accumulating, replacing each frame.
    // strokeSeed is fixed so the stochastic filter is stable and doesn't flicker on hover.
    if (
      !overGizmo &&
      e.buttons === 0 &&
      !probingRef.current &&
      !interactionBlockedRef.current &&
      !loading &&
      !workBusy &&
      interactionModeRef.current === "sculpt" &&
      sculptStrokeModeRef.current === "wall" &&
      wallAreaShapeRef.current === "brush"
    ) {
      const now = Date.now();
      if (now - lastWallHoverMsRef.current >= 40) {
        lastWallHoverMsRef.current = now;
        void invoke("voxel_sculpt_stroke_preview_at_screen", {
          args: {
            ...buildSculptStrokeInvokeArgs(px, py, {
              includeStrokeSeed: false,
            }),
            strokeLineStartNx: px,
            strokeLineStartNy: py,
            strokeSeed: 0,
          },
        }).catch(() => {});
      }
    }

    // Terrain hover: show surface Y under cursor when not stroking.
    if (
      e.buttons === 0 &&
      !probingRef.current &&
      !interactionBlockedRef.current &&
      !loading &&
      !workBusy &&
      interactionModeRef.current === "sculpt" &&
      sculptStrokeModeRef.current === "terrain"
    ) {
      const now = Date.now();
      if (now - lastTerrainHoverMsRef.current >= 80) {
        lastTerrainHoverMsRef.current = now;
        void invoke<number | null>("terrain_surface_y_at_screen", {
          nx: px,
          ny: py,
        })
          .then((r) => setTerrainHoverY(r))
          .catch(() => {});
      }
    }

    if (probingRef.current && activePointerIdRef.current === e.pointerId) {
      return;
    }
    // Extrude re-drag: reposition endpoint during settings phase.
    if (extrudeRedragRef.current && e.buttons && pointerStartRef.current) {
      const now = Date.now();
      if (now - lastStrokeEditMsRef.current >= 24) {
        lastStrokeEditMsRef.current = now;
        const startNorm = extrudeStartNormRef.current;
        if (startNorm) {
          const dpr = window.devicePixelRatio || 1;
          const screenDx = (e.clientX - pointerStartRef.current.x) * dpr;
          const screenDy = (pointerStartRef.current.y - e.clientY) * dpr;
          if (interactionModeRef.current === "selectExtrude") {
            void invoke("selection_extrude_preview", {
              args: {
                screenDx,
                screenDy,
                directionRef: "camera",
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch((err) => {
              console.error("[selection_extrude_preview re-drag]", err);
            });
          } else {
            void invoke("extrude_ray_preview", {
              args: {
                startNx: startNorm.nx,
                startNy: startNorm.ny,
                screenDx,
                screenDy,
                directionRef: extrudeDirectionRefRef.current,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
                brushRadius: sculptBrushRadiusRef.current,
                brushShape: sculptBrushShapeToRust(sculptBrushShapeUiRef.current),
                brushStrength: sculptBrushStrengthRef.current,
                brushFalloff: sculptBrushFalloffRef.current,
                strokeSeed: Math.floor(Math.random() * 0x1_0000_0000) >>> 0,
                extrudeProfile: extrudeProfileRef.current,
                extrudeEndCap: extrudeEndCapRef.current,
                extrudeTaper: extrudeTaperRef.current,
                extrudeTaperStart: extrudeTaperRef.current ? extrudeTaperStartRef.current : 0,
                extrudeTaperEnd: extrudeTaperRef.current ? extrudeTaperEndRef.current : 0,
              },
            }).catch((err) => {
              console.error("[extrude_ray_preview re-drag]", err);
            });
          }
        }
      }
      return;
    }
    if (
      gestureRef.current &&
      gestureRef.current.pointerId === e.pointerId &&
      gestureRef.current.mode === "voxel"
    ) {
      if (pointerStartRef.current) {
        const dx = e.clientX - pointerStartRef.current.x;
        const dy = e.clientY - pointerStartRef.current.y;
        maxPointerMoveRef.current = Math.max(maxPointerMoveRef.current, Math.hypot(dx, dy));
      }
      const m = interactionModeRef.current;
      {
        const dispatch = getStrokeDispatch(m);
        if (
          e.buttons &&
          dispatch &&
          !loading &&
          !workBusy &&
          !fillOperationPending &&
          !strokeModeSkipsDrag(drawStrokeModeRef.current) &&
          !(drawStrokeModeRef.current === "cuboid" && cuboidPhase.ref.current) &&
          !(drawStrokeModeRef.current === "cylinder" && cylinderPhase.ref.current)
        ) {
          const now = Date.now();
          if (now - lastStrokeEditMsRef.current >= 24) {
            lastStrokeEditMsRef.current = now;
            dragDidEditRef.current = true;
            const lineStart = strokeViewportLineStartNorm();
            const brushPrev =
              strokeDrawStyleRef.current === "brush" && lastStrokeNormRef.current
                ? lastStrokeNormRef.current
                : null;
            if (dispatch.kind === "edit") {
              const previewPalette = selectedColorsRef.current;
              const previewMultiColor = previewPalette.length > 1;
              void invoke("voxel_stroke_preview_at_screen", {
                args: {
                  nx: px,
                  ny: py,
                  tool: dispatch.tool,
                  color: activeColorRef.current,
                  ...(previewMultiColor
                    ? {
                        palette: previewPalette,
                        paintColorDistrib: paintColorDistribRef.current,
                        strokeSeed: currentStrokeSeedRef.current,
                      }
                    : {}),
                  material: activeMaterialRef.current,
                  brushRadius: brushRadiusRef.current,
                  brushShape: brushShapeRef.current,
                  sprayDensity: sprayDensityRef.current,
                  strokeMode: drawStrokeModeRef.current,
                  planeAxis: planeAxisRef.current,
                  strokeAux: mergedStrokeAux({}),
                  matchMaterial: matchMaterialSelectColorRef.current,
                  mirrorAxes:
                    (mirrorXRef.current ? 1 : 0) |
                    (mirrorYRef.current ? 2 : 0) |
                    (mirrorZRef.current ? 4 : 0),
                  ...(lineStart
                    ? {
                        strokeLineStartNx: lineStart.nx,
                        strokeLineStartNy: lineStart.ny,
                      }
                    : {}),
                  ...(!lineStart && brushPrev
                    ? {
                        strokeSegmentPrevNx: brushPrev.nx,
                        strokeSegmentPrevNy: brushPrev.ny,
                      }
                    : {}),
                },
              })
                .finally(() => {
                  if (strokeDrawStyleRef.current === "brush") {
                    lastStrokeNormRef.current = { nx: px, ny: py };
                  }
                })
                .catch(() => {});
              logPlaneStrokeDebug("move:preview", e, {
                nx: px,
                ny: py,
                tool: dispatch.tool,
                lineStart: lineStart ?? null,
                brushPrev: brushPrev ?? null,
              });
            } else {
              runStrokeAtScreen(px, py, {}, { lineStart, brushPrev });
              if (strokeDrawStyleRef.current === "brush") {
                lastStrokeNormRef.current = { nx: px, ny: py };
              }
            }
          }
        }
      }
      if (e.buttons && m === "sculpt" && !loading && !workBusy && !fillOperationPending) {
        const now = Date.now();
        if (now - lastStrokeEditMsRef.current >= 24) {
          lastStrokeEditMsRef.current = now;
          dragDidEditRef.current = true;
          if (
            sculptStrokeModeRef.current === "extrude" &&
            pointerStartRef.current &&
            (strokeViewportStartRef.current || extrudeRedragRef.current)
          ) {
            // Ray-based extrude: compute screen delta and send to Rust.
            // Use stored start position for re-drags during settings phase.
            const startNorm = extrudeStartNormRef.current ?? strokeViewportStartRef.current;
            if (startNorm) {
              // On first drag, persist the start position for later re-drags.
              if (!extrudeStartNormRef.current && strokeViewportStartRef.current) {
                extrudeStartNormRef.current = {
                  ...strokeViewportStartRef.current,
                };
              }
              const dpr = window.devicePixelRatio || 1;
              const screenDx = (e.clientX - pointerStartRef.current.x) * dpr;
              const screenDy = (pointerStartRef.current.y - e.clientY) * dpr; // screen up = +
              void invoke("extrude_ray_preview", {
                args: {
                  startNx: startNorm.nx,
                  startNy: startNorm.ny,
                  screenDx,
                  screenDy,
                  directionRef: extrudeDirectionRefRef.current,
                  color: activeColorRef.current,
                  material: activeMaterialRef.current,
                  brushRadius: sculptBrushRadiusRef.current,
                  brushShape: sculptBrushShapeToRust(sculptBrushShapeUiRef.current),
                  brushStrength: sculptBrushStrengthRef.current,
                  brushFalloff: sculptBrushFalloffRef.current,
                  strokeSeed: Math.floor(Math.random() * 0x1_0000_0000) >>> 0,
                  extrudeProfile: extrudeProfileRef.current,
                  extrudeEndCap: extrudeEndCapRef.current,
                  extrudeTaper: extrudeTaperRef.current,
                  extrudeTaperStart: extrudeTaperRef.current ? extrudeTaperStartRef.current : 0,
                  extrudeTaperEnd: extrudeTaperRef.current ? extrudeTaperEndRef.current : 0,
                },
              }).catch((err) => {
                console.error("[extrude_ray_preview]", err);
              });
            }
          } else {
            const sculptBrushPrev = lastStrokeNormRef.current;
            void invoke("voxel_sculpt_stroke_preview_at_screen", {
              args: buildSculptStrokeInvokeArgs(px, py, {
                strokeSegmentPrev: sculptBrushPrev,
              }),
            })
              .finally(() => {
                lastStrokeNormRef.current = { nx: px, ny: py };
              })
              .catch(() => {});
          }
        }
      }
      // selectExtrude drag: preview extruded selection along drag direction.
      if (e.buttons && m === "selectExtrude" && pointerStartRef.current && !loading && !workBusy) {
        const now = Date.now();
        if (now - lastStrokeEditMsRef.current >= 24) {
          lastStrokeEditMsRef.current = now;
          dragDidEditRef.current = true;
          const startNorm = extrudeStartNormRef.current ?? strokeViewportStartRef.current;
          if (startNorm) {
            if (!extrudeStartNormRef.current && strokeViewportStartRef.current) {
              extrudeStartNormRef.current = { ...strokeViewportStartRef.current };
            }
            const dpr = window.devicePixelRatio || 1;
            const screenDx = (e.clientX - pointerStartRef.current.x) * dpr;
            const screenDy = (pointerStartRef.current.y - e.clientY) * dpr;
            void invoke("selection_extrude_preview", {
              args: {
                screenDx,
                screenDy,
                directionRef: "camera",
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch((err) => {
              console.error("[selection_extrude_preview]", err);
            });
          }
        }
      }
      // Roof square/circle drag: compute pins during drag.
      if (
        e.buttons &&
        m === "generator" &&
        generatorKindRef.current === "roof" &&
        roofFirstClickRef.current &&
        !loading &&
        !workBusy
      ) {
        const shape = roofAreaShapeRef.current;
        if (shape === "square" || shape === "circle") {
          const now = Date.now();
          if (now - lastStrokeEditMsRef.current >= 40) {
            lastStrokeEditMsRef.current = now;
            dragDidEditRef.current = true;
            void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
              args: {
                nx: px,
                ny: py,
                tool: "add",
                strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
              },
            })
              .then((c) => {
                if (!c || !roofFirstClickRef.current) return;
                const first = roofFirstClickRef.current;
                let pins: [number, number, number][];
                if (shape === "square") {
                  const [x1, y1, z1] = first;
                  const [x2, , z2] = c;
                  pins = [
                    [x1, y1, z1],
                    [x2, y1, z1],
                    [x2, y1, z2],
                    [x1, y1, z2],
                  ];
                } else {
                  const [cx, cy, cz] = first;
                  const [ex, , ez] = c;
                  const dx = ex - cx;
                  const dz = ez - cz;
                  const r = Math.sqrt(dx * dx + dz * dz);
                  const N = 16;
                  pins = [];
                  for (let i = 0; i < N; i++) {
                    const angle = (2 * Math.PI * i) / N;
                    pins.push([
                      Math.round(cx + r * Math.cos(angle)),
                      cy,
                      Math.round(cz + r * Math.sin(angle)),
                    ]);
                  }
                }
                // Ensure pins wind so the roof grows upward (+Y).
                // In the XZ plane, positive signed area means CW from
                // above, which gives a downward normal — reverse to fix.
                let areaXZ = 0;
                for (let i = 0; i < pins.length; i++) {
                  const p = pins[i];
                  const q = pins[(i + 1) % pins.length];
                  areaXZ += p[0] * q[2] - q[0] * p[2];
                }
                if (areaXZ >= 0) {
                  pins.reverse();
                }
                setRoofPins(pins);
                roofPinsRef.current = pins;
                void invoke("sync_preview_input", {
                  args: buildSyncPreviewPayload(
                    px,
                    py,
                    previewModeForSync(interactionModeRef.current),
                  ),
                }).catch(() => {});
              })
              .catch(() => {});
          }
        }
      }
      return;
    }
    if (pointerStartRef.current) {
      const dx = e.clientX - pointerStartRef.current.x;
      const dy = e.clientY - pointerStartRef.current.y;
      maxPointerMoveRef.current = Math.max(maxPointerMoveRef.current, Math.hypot(dx, dy));
    }
    const dpr = window.devicePixelRatio || 1;
    const dx = (e.clientX - lastRef.current.x) * dpr;
    const dy = (e.clientY - lastRef.current.y) * dpr;
    lastRef.current = { x: e.clientX, y: e.clientY };
    if (e.buttons === 0) {
      logPlaneStrokeDebug("move:buttons-zero", e);
      if (startScreenLogoLoadedRef.current && interactionModeRef.current !== "fly") {
        const { nx, ny } = clientToViewportNormalized(e);
        void invoke("viewport_pointer", {
          ev: {
            kind: "move",
            nx,
            ny,
            dx: 0,
            dy: 0,
            button: e.button,
            buttons: 0,
            shiftKey: e.shiftKey,
          },
        }).catch(() => {});
      }
      return;
    }
    const { nx, ny } = clientToViewportNormalized(e);
    // Don't pan the camera during right-button drag in stamp/punch mode —
    // right-click in those modes rotates the stamp, not the camera.
    const stampPunchRightDrag =
      (interactionModeRef.current === "stamp" || interactionModeRef.current === "punch") &&
      (e.buttons & 2) !== 0 &&
      gestureRef.current?.mode !== "camera";
    if (interactionModeRef.current !== "fly" && !stampPunchRightDrag) {
      void invoke("viewport_pointer", {
        ev: {
          kind: "move",
          nx,
          ny,
          dx,
          dy,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey && gestureRef.current?.mode !== "voxel",
        },
      });
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("up:received", e);
    // Extrude re-drag: stay in settings phase after repositioning endpoint.
    if (extrudeRedragRef.current) {
      extrudeRedragRef.current = false;
      pointerStartRef.current = null;
      return;
    }
    if (probingRef.current && activePointerIdRef.current === e.pointerId) {
      // Probe is still in-flight — defer this up event so onPointerDown can
      // replay it once the gesture is established (see pendingPointerUpRef).
      pendingPointerUpRef.current = e;
      return;
    }

    if (
      gestureRef.current?.mode === "squishyGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      void invoke("squishy_gizmo_pointer_up").catch(() => {});
      resetPointerGesture("squishy-gizmo-up", e);
      return;
    }

    if (
      gestureRef.current?.mode === "selectionGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      gizmoRef.current?.pointerUp();
      resetPointerGesture("selection-gizmo-up", e);
      return;
    }
    if (
      gestureRef.current?.mode === "extrudeGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      extrudeGizmoRef.current?.pointerUp();
      // Restore selectExtrude preview_mode so the GPU gizmo keeps its extrude style.
      void invoke("sync_preview_input", {
        args: buildSyncPreviewPayload(-1, 0, "selectExtrude"),
      }).catch(() => {});
      if (dragDidEditRef.current) {
        extrudePhase.enter("settings", {} as Record<string, never>);
        lastStrokeNormRef.current = null;
      } else {
        void invoke("voxel_stroke_preview_reset").catch(() => {});
        lastStrokeNormRef.current = null;
      }
      resetPointerGesture("extrude-gizmo-up", e);
      return;
    }

    const g = gestureRef.current;
    const start = pointerStartRef.current;
    const moved = maxPointerMoveRef.current;
    const isThisPointer = g?.pointerId === e.pointerId;
    let hasPointerCaptureForUp = capturedPointerIdRef.current === e.pointerId;
    if (!hasPointerCaptureForUp && viewportRef.current) {
      try {
        hasPointerCaptureForUp = viewportRef.current.hasPointerCapture(e.pointerId);
      } catch {
        hasPointerCaptureForUp = false;
      }
    }

    if (
      isThisPointer &&
      g?.mode === "voxel" &&
      !loading &&
      !workBusy &&
      !fillOperationPending &&
      start &&
      e.button === 0 &&
      hasPointerCaptureForUp
    ) {
      const { nx, ny } = clientToViewportNormalized(e);
      const m = interactionModeRef.current;
      if (moved < 5) {
        if (m === "stamp") {
          void invoke("clipboard_stamp_at_screen", {
            args: {
              nx,
              ny,
              rotX: stampRotXRef.current,
              rotY: stampRotYRef.current,
              rotZ: stampRotZRef.current,
              originX: stampOriginXRef.current,
              originZ: stampOriginZRef.current,
            },
          }).catch(() => {});
        } else if (m === "punch") {
          void invoke("clipboard_punch_at_screen", {
            args: {
              nx,
              ny,
              rotX: stampRotXRef.current,
              rotY: stampRotYRef.current,
              rotZ: stampRotZRef.current,
              originX: stampOriginXRef.current,
              originZ: stampOriginZRef.current,
            },
          }).catch(() => {});
        } else if (m === "generator") {
          const gk = generatorKindRef.current;
          if (gk === "rocks") {
            if (!rocksPhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              rocksPhase.enter("settings", { nx, ny, seed: rockPreviewSeedRef.current });
            }
          } else if (gk === "grass") {
            if (!grassPhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              grassPhase.enter("settings", { nx, ny, seed: grassPreviewSeedRef.current });
            }
          } else if (gk === "cloth") {
            if (!clothPhase.active) {
              void handleClothPinClick(nx, ny);
            }
          } else if (gk === "rope") {
            if (ropePhase.active) {
              // Already in settings phase — ignore clicks
            } else if (!ropeFirstScreen) {
              setRopeFirstScreen({ nx, ny });
            } else {
              // Enter settings phase instead of immediately generating
              void invoke("lock_generator_preview_camera").catch(() => {});
              ropePhase.enter("settings", {
                nx1: ropeFirstScreen.nx,
                ny1: ropeFirstScreen.ny,
                nx2: nx,
                ny2: ny,
              });
            }
          } else if (gk === "ashlar") {
            if (!ashlarPhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              ashlarPhase.enter("settings", { nx, ny, seed: ashlarPreviewSeedRef.current });
            }
          } else if (gk === "flora") {
            if (!floraPhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              floraPhase.enter("settings", { nx, ny, seed: floraPreviewSeedRef.current });
            }
          } else if (gk === "roof") {
            // Square/circle use drag-to-define (handled in pointer move/up).
            // Polygon still uses click-to-add-pin.
            if (roofAreaShapeRef.current === "polygon") {
              void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
                args: {
                  nx,
                  ny,
                  tool: "add",
                  strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
                },
              })
                .then((c) => {
                  if (!c) return;
                  setRoofPins((v) => {
                    // Click on existing pin → remove it.
                    const idx = v.findIndex((p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2]);
                    const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
                    roofPinsRef.current = next;
                    return next;
                  });
                })
                .catch(() => {});
            }
          } else if (gk === "piscina") {
            if (!piscinaPhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              piscinaPhase.enter("settings", { nx, ny, seed: piscinaPreviewSeedRef.current });
            }
          } else if (gk === "insecta") {
            if (!insectaPhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              insectaPhase.enter("settings", { nx, ny });
            }
          } else if (gk === "fauna") {
            if (!faunaPhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              faunaPhase.enter("settings", { nx, ny });
            }
          }
        } else if (m === "squishy") {
          if (!squishyPhase.active) {
            squishyPhase.enter("settings", {});
          }
          const mode = squishyModeRef.current;
          void invoke("squishy_session_set_mode", { args: { mode } })
            .then(() => {
              if (mode === "add") {
                return invoke("squishy_metaball_add_at_screen", {
                  args: {
                    nx,
                    ny,
                    radius: Math.max(2, generatorSphereRadiusRef.current),
                  },
                });
              }
              return invoke<number | null>("squishy_pick_at_screen", {
                args: { nx, ny },
              }).then((id) => {
                if (id == null) return;
                if (mode === "delete") {
                  return invoke("squishy_metaball_remove", { args: { id } });
                }
                return invoke("squishy_metaball_select", { args: { id } });
              });
            })
            .then(() => invoke<{ balls: { id: number }[] }>("squishy_session_get"))
            .then((s) => setSquishyBallCount(s.balls?.length ?? 0))
            .catch(() => {});
        }
      }
      if (m === "eyedropper") {
        if (moved < 5) {
          void invoke<{
            color: number;
            material: string;
          } | null>("voxel_pick_color_at_screen", {
            args: {
              nx,
              ny,
              tool: "add",
              color: activeColorRef.current,
              material: activeMaterialRef.current,
              brushRadius: 0,
              brushShape: brushShapeRef.current,
            },
          })
            .then((r) => {
              if (r) {
                setActiveColor(r.color);
                setActiveMaterial(r.material);
                const back = eyedropperReturnModeRef.current;
                if (back != null && back !== "eyedropper") {
                  setInteractionMode(back);
                }
              }
            })
            .catch(() => {});
        }
      } else if (getStrokeDispatch(m)) {
        const dispatch = getStrokeDispatch(m)!;
        const sm = drawStrokeModeRef.current;
        // Depth phase activation after drag (cuboid/cylinder)
        if (
          (sm === "cuboid" || sm === "cylinder") &&
          (dragDidEditRef.current || moved >= 5) &&
          strokeViewportStartRef.current
        ) {
          const phase = sm === "cuboid" ? cuboidPhase : cylinderPhase;
          const depthRef = sm === "cuboid" ? cuboidDepthRef : cylinderDepthRef;
          const setDepthUi = sm === "cuboid" ? setCuboidDepthUi : setCylinderDepthUi;
          depthRef.current = 1;
          setDepthUi(1);
          const lineStart = { ...strokeViewportStartRef.current };
          phase.enter("depth", { lineStart, endNorm: { nx, ny }, frozenGeo: null });
          {
            const im = interactionModeRef.current;
            const tool = im === "remove" ? "remove" : im === "paint" ? "paint" : "add";
            void invoke<CuboidPlaneGeo | null>("query_cuboid_plane_geometry", {
              args: {
                nx,
                ny,
                tool,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
                brushRadius: brushRadiusRef.current,
                brushShape: brushShapeRef.current,
                sprayDensity: 0,
                strokeMode: sm,
                planeAxis: planeAxisRef.current,
                strokeAux: mergedStrokeAux({}),
                matchMaterial: false,
                strokeLineStartNx: lineStart.nx,
                strokeLineStartNy: lineStart.ny,
              },
            })
              .then((geo) => {
                phase.update({ frozenGeo: geo ?? null });
              })
              .catch(() => {});
          }
        }
        // Single-click handling (no drag)
        // For selection strokes, we must wait for the stroke invoke to
        // complete before calling selection_stroke_end, otherwise the
        // end command can race ahead and clear the accumulator.
        let clickStrokePromise: Promise<void> | null = null;
        if (!dragDidEditRef.current && moved < 5) {
          if ((sm === "cuboid" || sm === "cylinder") && strokeViewportStartRef.current) {
            // Single-click cuboid/cylinder: enter depth phase
            const phase = sm === "cuboid" ? cuboidPhase : cylinderPhase;
            const depthRef = sm === "cuboid" ? cuboidDepthRef : cylinderDepthRef;
            const setDepthUi = sm === "cuboid" ? setCuboidDepthUi : setCylinderDepthUi;
            depthRef.current = 1;
            setDepthUi(1);
            const lineStart = { ...strokeViewportStartRef.current };
            phase.enter("depth", { lineStart, endNorm: { nx, ny }, frozenGeo: null });
            {
              const im = interactionModeRef.current;
              const tool = im === "remove" ? "remove" : im === "paint" ? "paint" : "add";
              void invoke<CuboidPlaneGeo | null>("query_cuboid_plane_geometry", {
                args: {
                  nx,
                  ny,
                  tool,
                  color: activeColorRef.current,
                  material: activeMaterialRef.current,
                  brushRadius: brushRadiusRef.current,
                  brushShape: brushShapeRef.current,
                  sprayDensity: 0,
                  strokeMode: sm,
                  planeAxis: planeAxisRef.current,
                  strokeAux: mergedStrokeAux({}),
                  matchMaterial: false,
                  strokeLineStartNx: lineStart.nx,
                  strokeLineStartNy: lineStart.ny,
                },
              })
                .then((geo) => {
                  phase.update({ frozenGeo: geo ?? null });
                })
                .catch(() => {});
            }
          } else if (strokeModeSkipsDrag(sm)) {
            void handleStrokeAnchorClick(nx, ny);
          } else if (
            dispatch.kind === "selection" &&
            dispatch.interaction !== "select" &&
            sm !== "fill"
          ) {
            // Specialized selection click commands (selectByColor, etc.)
            void invokeSelectionSpecialClick(dispatch.interaction, nx, ny);
          } else {
            const lineStart = strokeViewportLineStartNorm();
            clickStrokePromise = runStrokeAtScreen(nx, ny, {}, { lineStart });
          }
        }
        logPlaneStrokeDebug("up:stroke-end", e, {
          moved,
          dragDidEdit: dragDidEditRef.current,
          strokeMode: sm,
          interactionMode: m,
        });
        if (dispatch.kind === "edit") {
          void invoke("voxel_stroke_end").catch(() => {});
        } else {
          const endSelection = () => {
            void invoke("selection_stroke_end").catch(() => {});
            selectionStrokeBegunRef.current = false;
          };
          if (clickStrokePromise) {
            void clickStrokePromise.then(endSelection);
          } else {
            endSelection();
          }
        }
        lastStrokeNormRef.current = null;
      } else if (m === "sculpt") {
        const sm = sculptStrokeModeRef.current;
        if (sm === "extrude" && (dragDidEditRef.current || moved >= 5)) {
          // Extrude phased tool: enter settings phase instead of committing.
          // The preview union is already accumulated from the drag. Keep it visible.
          extrudePhase.enter("settings", {} as Record<string, never>);
          lastStrokeNormRef.current = null;
          // Do NOT call voxel_stroke_end — preview must stay.
        } else {
          if (!dragDidEditRef.current && moved < 5) {
            const wa = wallAreaShapeRef.current;
            if (sm === "wall" && wa === "polygon") {
              void handleWallSculptPolygonClick(nx, ny);
            } else {
              void invoke("voxel_sculpt_stroke_at_screen", {
                args: buildSculptStrokeInvokeArgs(nx, ny),
              }).catch(() => {});
            }
          }
          void invoke("voxel_stroke_end").catch(() => {});
          lastStrokeNormRef.current = null;
        }
      } else if (m === "selectExtrude") {
        if (dragDidEditRef.current || moved >= 5) {
          // Enter settings phase — preview union stays visible.
          extrudePhase.enter("settings", {} as Record<string, never>);
          lastStrokeNormRef.current = null;
          // Do NOT call voxel_stroke_end — preview must stay.
        } else {
          void invoke("voxel_stroke_preview_reset").catch(() => {});
          lastStrokeNormRef.current = null;
        }
      } else if (m === "generator" && generatorKindRef.current === "roof") {
        // Roof square/circle drag complete: clear first-click anchor.
        // Pins are already set from the move handler during drag.
        const shape = roofAreaShapeRef.current;
        if (shape === "square" || shape === "circle") {
          roofFirstClickRef.current = null;
          setRoofFirstClick(null);
        }
      }
    } else if (isThisPointer && g?.mode === "voxel" && e.button === 0 && !hasPointerCaptureForUp) {
      logPlaneStrokeDebug("up:ignored-no-capture", e, {
        moved,
      });
    }

    if (isThisPointer && g?.mode === "camera" && interactionModeRef.current !== "fly") {
      const { nx, ny } = clientToViewportNormalized(e);
      void invoke("viewport_pointer", {
        ev: {
          kind: "up",
          nx,
          ny,
          dx: 0,
          dy: 0,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey,
        },
      });
    }

    if (isThisPointer) {
      resetPointerGesture("pointer-up-complete", e);
      if (capturedPointerIdRef.current === e.pointerId) {
        capturedPointerIdRef.current = null;
      }
    }
  };
  onPointerUpRef.current = onPointerUp;

  const onPointerLeave = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("leave", e);
    // Keep the select preview visible when the pointer moves to the sidebar / tool panel.
    // Also keep phased-tool previews alive — their coordinates are locked to
    // stored data anyway, so clearing on leave just creates a jarring flicker
    // when the user mouses over the settings overlay.
    const im = interactionModeRef.current;
    const anyPhaseActive =
      cuboidPhase.ref.current !== null ||
      cylinderPhase.ref.current !== null ||
      extrudePhase.ref.current !== null ||
      ropePhase.ref.current !== null ||
      clothPhase.ref.current !== null;
    if (
      !anyPhaseActive &&
      im !== "select" &&
      im !== "selectByColor" &&
      im !== "selectCoplanar" &&
      im !== "selectCoplanarEmpty" &&
      im !== "selectExtrude" &&
      im !== "squishy" &&
      im !== "generator"
    ) {
      clearPreview();
    }
    if (viewportCursorDebugEnabled) {
      setViewportCursorDebugJs(null);
      setViewportCursorDebugRust(null);
      viewportCursorDebugScreenRef.current = null;
      setViewportCursorDebugScreen(null);
    }
    if (startScreenLogoLoadedRef.current && interactionModeRef.current !== "fly") {
      void invoke("viewport_pointer", {
        ev: {
          kind: "leave",
          nx: 0.5,
          ny: 0.5,
          dx: 0,
          dy: 0,
          button: 0,
          buttons: 0,
          shiftKey: false,
        },
      }).catch(() => {});
    }
    // Never synthesize pointer-up from leave. Commit only on real pointer-up/cancel;
    // pointer capture keeps those events routed to the viewport during drags.
  };

  const onGotPointerCapture = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("capture:got", e);
  };

  const onLostPointerCapture = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("capture:lost", e);
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (interactionModeRef.current === "fly" || interactionModeRef.current === "walk") return;
    void invoke("viewport_wheel", {
      ev: { delta_x: e.deltaX, delta_y: e.deltaY },
    });
  };

  const startHost = () => {
    if (collabActive) return;
    setCollabBanner(null);
    setLoadError(null);
    const rgb = hexToRgb(normalizeCollabAccentColor(accentColor));
    void invoke("collab_host_start", {
      port: hostPort,
      displayName: normalizeCollabDisplayName(displayName),
      colorRgb: rgb,
      enableUpnp: prefsEnableUpnp,
    })
      .then((res) => {
        const r = res as { lanUrl: string; nat: string };
        setHostWsUrl(r.lanUrl);
        setHostWanUrl(null);
        setNatError(null);
        setNatPending(r.nat === "pending");
        setCollabActive(true);
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        setLoadError(
          `Couldn't start a session.\n\n${msg}\n\nLeave any open session first, or try a different port.`,
        );
      });
  };

  const joinSession = (urlOverride?: string) => {
    if (collabActive) return;
    setCollabBanner(null);
    setLoadError(null);
    const u = (urlOverride ?? joinUrl).trim();
    if (!u) {
      setLoadError("Enter a host address.");
      return;
    }
    setJoinUrl(u);
    pendingJoinUrlRef.current = u;
    setCollabJoinPending(true);
    const rgb = hexToRgb(normalizeCollabAccentColor(accentColor));
    void invoke("collab_join", {
      url: u,
      displayName: normalizeCollabDisplayName(displayName),
      colorRgb: rgb,
    });
  };

  const cancelJoin = () => {
    void invoke("collab_cancel_join").catch(() => {});
  };

  const leaveSession = () => {
    void invoke("collab_leave").catch(() => {});
  };

  collabActiveMenuRef.current = collabActive;
  startHostMenuRef.current = startHost;
  leaveSessionMenuRef.current = leaveSession;

  const copyHostingJoinAddress = () => {
    const url = hostWanUrl ?? hostWsUrl;
    if (!url) return;
    void navigator.clipboard.writeText(url).then(
      () => {
        setHostingCopied(true);
        window.setTimeout(() => setHostingCopied(false), 2000);
      },
      () => {},
    );
  };

  const amLeader = roster.some((r) => r.peerId === localPeerId && r.isLeader);

  /** Solo or host: can open files. Guests (session without hosting) cannot. */
  const collabGuest = collabActive && !hostWsUrl;
  /** Editor chrome (sidebars, tool HUD) once a document exists, collab is active, or load/join is in progress. */
  const showEditorChrome =
    Boolean(pathLabel) ||
    collabActive ||
    loading ||
    collabJoinPending ||
    (workBusy && !startScreenLogoLoaded);
  const showStartScreen = !showEditorChrome;
  const showEmptyOpenFile = showStartScreen && !pendingAutoReopen;
  /** Cold start: logo mesh still decoding (ignore `showStartScreen` while `workBusy` toggles editor chrome).
   *  Also shown while waiting to resolve auto-reopen so the start screen doesn't flash before loading begins. */
  const showStartScreenLogoSpinner =
    (!startScreenLogoLoaded && !pathLabel && !collabActive) ||
    (pendingAutoReopen && !loading && !pathLabel && !collabActive);

  const reopenLastProject = useCallback(() => {
    const info = lastSessionInfo;
    if (!info?.lastDocumentPath) return;
    const doc = info.lastDocumentPath;
    const auto = info.autosavePath;
    const useAutosave =
      info.autosaveExists &&
      auto != null &&
      auto !== "" &&
      (!info.documentExists || info.autosaveNewerThanDocument);

    if (useAutosave) {
      void invoke("load_voxelle_recovery", {
        args: { documentPath: doc, autosavePath: auto },
      }).catch((err) => {
        setLoadError(err instanceof Error ? err.message : String(err));
      });
      return;
    }
    if (info.documentExists) {
      void invoke("load_voxelle_path", { path: doc }).catch((err) => {
        setLoadError(err instanceof Error ? err.message : String(err));
      });
    }
  }, [lastSessionInfo]);

  const lastProjectBlurb = lastSessionInfo != null ? lastProjectReopenBlurb(lastSessionInfo) : null;

  /** Phase + progress for the top-center viewport HUD (load / mesh / fill / etc.). */
  const viewportTopCenterHud = (() => {
    if (loading && pathLabel) {
      const pct = Math.round(Math.min(1, Math.max(0, loadProgress)) * 100);
      const phase = loadPhase.trim();
      return {
        label: phase || `Loading ${basename(pathLabel)}…`,
        pct,
        showFillCancel: false,
      };
    }
    if (loading) {
      return {
        label: loadPhase.trim() || "Loading…",
        pct: Math.round(Math.min(1, Math.max(0, loadProgress)) * 100),
        showFillCancel: false,
      };
    }
    if (workBusy || fillOperationPending) {
      const pct = Math.round(Math.min(1, Math.max(0, workProgress)) * 100);
      const phase = workPhase.trim();
      const showFillCancel = fillOperationPending || /fill/i.test(workPhase);
      return {
        label: phase || (fillOperationPending ? "Fill…" : "Working…"),
        pct,
        showFillCancel,
      };
    }
    return null;
  })();

  const statusBarMessage = (() => {
    // Shorten footer only while loading (top HUD shows load detail). Keep mesh/edit phases in the bar during work.
    if (viewportTopCenterHud != null && loading) {
      if (pathLabel) {
        const base = basename(pathLabel);
        if (collabActive) return `${base} · Live`;
        return base;
      }
      return collabActive ? "Live session" : `v${VOXELLE_DESKTOP_VERSION}`;
    }
    if (loading && pathLabel) {
      const pct = Math.round(Math.min(1, Math.max(0, loadProgress)) * 100);
      const phase = loadPhase.trim();
      return phase
        ? `Loading ${basename(pathLabel)}… ${pct}% — ${phase}`
        : `Loading ${basename(pathLabel)}… ${pct}%`;
    }
    if (loading) return "Loading…";
    if (workBusy || fillOperationPending) {
      const pct = Math.round(Math.min(1, Math.max(0, workProgress)) * 100);
      const phase = workPhase.trim();
      return phase
        ? `${phase} ${pct}%`
        : fillOperationPending
          ? `Fill… ${pct}%`
          : `Working… ${pct}%`;
    }
    if (pathLabel) {
      const base = basename(pathLabel);
      if (collabActive) return `${base} · Live`;
      return base;
    }
    return `v${VOXELLE_DESKTOP_VERSION}`;
  })();

  const sendChat = () => {
    const t = chatInput.trim();
    if (!t) return;
    void invoke("collab_send_chat", { text: t }).catch(() => {});
    setChatInput("");
  };

  const onRosterSnapCamera = (peerId: number) => {
    void invoke("collab_snap_camera", { peerId }).catch(() => {});
  };

  const setCanEdit = (peerId: number, canEdit: boolean) => {
    void invoke("collab_set_can_edit", { targetPeer: peerId, canEdit }).catch(() => {});
  };

  const isSelectionInteractionMode =
    interactionMode === "select" ||
    interactionMode === "selectByColor" ||
    interactionMode === "selectCoplanar" ||
    interactionMode === "selectCoplanarEmpty";

  const isDrawVoxelEditMode =
    interactionMode === "add" || interactionMode === "remove" || interactionMode === "paint";

  const showDrawPaneToolMatrix =
    (toolsPane === "draw" || toolsPane === "select") &&
    (isDrawVoxelEditMode || isSelectionInteractionMode);

  const showPolygonPhaseHud =
    showEditorChrome &&
    showDrawPaneToolMatrix &&
    (drawStrokeMode === "polygon" || drawStrokeMode === "polygonHull");

  const showWallSculptPolygonHud =
    showEditorChrome &&
    interactionMode === "sculpt" &&
    toolsPane === "draw" &&
    sculptStrokeMode === "wall" &&
    wallAreaShape === "polygon";

  const showViewportTopCenterStack =
    showEditorChrome &&
    (viewportTopCenterHud != null ||
      cuboidPhase.active ||
      cylinderPhase.active ||
      extrudePhase.active ||
      ropePhase.active ||
      clothPhase.active ||
      rocksPhase.active ||
      grassPhase.active ||
      ashlarPhase.active ||
      floraPhase.active ||
      piscinaPhase.active ||
      insectaPhase.active ||
      faunaPhase.active ||
      (generatorKind === "cloth" && clothPins.length > 0) ||
      (generatorKind === "roof" && (roofPins.length > 0 || roofFirstClick !== null)) ||
      showPolygonPhaseHud ||
      showWallSculptPolygonHud);

  const selectionMethod = deriveSelectionMethod({
    drawStrokeMode,
    strokeDrawStyle,
    sprayDensity,
    strokeFamilyVariant,
  });

  const showToolOptionsPanel =
    showEditorChrome &&
    !loading &&
    !workBusy &&
    (toolsPane === "sculpt" ||
      toolsPane === "generators" ||
      toolsPane === "squishy" ||
      toolsPane === "mood" ||
      (toolsPane === "draw" &&
        (interactionMode === "add" ||
          interactionMode === "remove" ||
          interactionMode === "paint" ||
          interactionMode === "eyedropper" ||
          interactionMode === "stamp" ||
          interactionMode === "punch" ||
          isSelectionInteractionMode)) ||
      (toolsPane === "select" &&
        (interactionMode === "stamp" ||
          interactionMode === "punch" ||
          interactionMode === "selectExtrude" ||
          isSelectionInteractionMode)));

  return (
    <div
      className={`app${loading && !loadError ? " app-loading-cursor" : ""}${hideUI ? " app--ui-hidden" : ""}`}
    >
      <div className="app-main">
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
                        toolsPane === "generators"
                          ? "sidebar-pane-tab is-active"
                          : "sidebar-pane-tab"
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
                        toolsPane === "mood" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"
                      }
                      aria-selected={toolsPane === "mood"}
                      disabled={loading || workBusy}
                      onClick={() => setToolsPane("mood")}
                    >
                      Mood
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
                  </div>
                  <div className="sidebar-expanded-slot" aria-label="Tool pane options">
                    {toolsPane === "hand" ? (
                      <p className="sidebar-pane-hint">Drag in viewport to orbit/pan.</p>
                    ) : null}

                    {toolsPane === "fly" ? (
                      <>
                        <p className="sidebar-pane-hint">
                          Click viewport to capture pointer. WASD move, E/Q up/down, Shift slow.
                          Mouse looks. Esc releases pointer.
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
                        <div
                          className="sidebar-tool-selection-row"
                          role="group"
                          aria-label="Tool and selection"
                        >
                          <div className="sidebar-tool-selection-col">
                            <div className="sidebar-section-label">Tool</div>
                            <div className="sidebar-mode-grid sidebar-mode-grid-stacked">
                              {(["add", "remove", "paint"] as const).map((m) => (
                                <button
                                  key={m}
                                  type="button"
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
                          </div>
                          <div className="sidebar-tool-selection-col">
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
                                  loading ||
                                  workBusy ||
                                  (selectionCount === 0 && !stampBookPatternActive)
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
                            </div>
                          </div>
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
                              loading ||
                              workBusy ||
                              (selectionCount === 0 && !stampBookPatternActive)
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
                              ["piscina", "Fish"],
                              ["insecta", "Insect"],
                              ["fauna", "Creature"],
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
                                piscinaPhase.cancel();
                                insectaPhase.cancel();
                                faunaPhase.cancel();
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
                          Size and shape in the tool options panel. Rope: two clicks. Cloth:
                          multi-pin + Apply.
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
                      { pane: "draw", label: "Draw", mode: "add" },
                      { pane: "select", label: "Sel", mode: "select" },
                      { pane: "sculpt", label: "Sculpt", mode: "sculpt" },
                      { pane: "generators", label: "Gen", mode: "generator" },
                      { pane: "squishy", label: "Sqsh", mode: "squishy" },
                      { pane: "mood", label: "Mood", mode: null },
                      { pane: "fly", label: "Fly", mode: "fly" },
                      { pane: "walk", label: "Walk", mode: "walk" },
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
                          className={`sidebar-collapsed-sub-btn${interactionMode === m ? " is-active" : ""}`}
                          disabled={loading || workBusy}
                          onClick={() => setInteractionMode(m)}
                        >
                          {m[0].toUpperCase() + m.slice(1)}
                        </button>
                      ))}
                      <div className="sidebar-collapsed-section-label">Select</div>
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
                          ["piscina", "Fish"],
                          ["insecta", "Insect"],
                          ["fauna", "Creature"],
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
                            piscinaPhase.cancel();
                            insectaPhase.cancel();
                            faunaPhase.cancel();
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
        {colorPaletteFloating && showEditorChrome ? (
          <div
            className="floating-palette-panel"
            style={{
              left: colorPalettePos.x,
              top: colorPalettePos.y,
              width: colorPaletteSize.w,
              height: colorPaletteSize.h,
            }}
            role="region"
            aria-label="Floating color palette"
          >
            <div className="floating-palette-header">
              <div
                className="floating-palette-drag-handle"
                onPointerDown={(e) => {
                  const pid = e.pointerId;
                  const startX = e.clientX;
                  const startY = e.clientY;
                  const origX = colorPalettePos.x;
                  const origY = colorPalettePos.y;

                  const handleMove = (moveE: PointerEvent) => {
                    if (moveE.pointerId !== pid) return;
                    const dx = moveE.clientX - startX;
                    const dy = moveE.clientY - startY;
                    setColorPalettePos({
                      x: origX + dx,
                      y: origY + dy,
                    });
                  };

                  const handleUp = (upE: PointerEvent) => {
                    if (upE.pointerId !== pid) return;
                    document.removeEventListener("pointermove", handleMove as EventListener);
                    document.removeEventListener("pointerup", handleUp as EventListener);
                  };

                  document.addEventListener("pointermove", handleMove as EventListener);
                  document.addEventListener("pointerup", handleUp as EventListener);
                }}
                aria-label="Drag to move palette"
              >
                <span className="floating-palette-grip" aria-hidden>
                  ⋮⋮
                </span>
              </div>
              <div className="floating-palette-header-actions">
                <button
                  type="button"
                  className="floating-palette-dock-btn"
                  onClick={() => setColorPaletteFloating(false)}
                  title="Dock palette"
                >
                  Dock
                </button>
              </div>
            </div>
            <div className="floating-palette-content">
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
                    interactionMode === "eyedropper"
                      ? "sidebar-mode-btn is-active"
                      : "sidebar-mode-btn"
                  }
                  disabled={loading || workBusy}
                  onClick={() => setInteractionMode("eyedropper")}
                >
                  <span className="sidebar-mode-label">Eyedropper</span>
                </button>
              </div>
              <PaletteSwatches
                activeColor={activeColor}
                selectedColors={selectedColors}
                setActiveColor={setActiveColor}
                setSelectedColors={setSelectedColors}
                disabled={loading || workBusy}
                palette={MATERIAL_BUILTIN_PALETTE_HEX}
              />
            </div>
            <div
              className="floating-palette-resize-handle"
              onPointerDown={(e) => {
                const pid = e.pointerId;
                const startX = e.clientX;
                const startY = e.clientY;
                const origW = colorPaletteSize.w;
                const origH = colorPaletteSize.h;

                const handleMove = (moveE: PointerEvent) => {
                  if (moveE.pointerId !== pid) return;
                  const dx = moveE.clientX - startX;
                  const dy = moveE.clientY - startY;
                  setColorPaletteSize({
                    w: Math.max(140, origW + dx),
                    h: Math.max(120, origH + dy),
                  });
                };

                const handleUp = (upE: PointerEvent) => {
                  if (upE.pointerId !== pid) return;
                  document.removeEventListener("pointermove", handleMove as EventListener);
                  document.removeEventListener("pointerup", handleUp as EventListener);
                };

                document.addEventListener("pointermove", handleMove as EventListener);
                document.addEventListener("pointerup", handleUp as EventListener);
              }}
              aria-label="Resize palette"
            />
          </div>
        ) : null}
        <div
          className={`viewport-wrap${showStartScreen ? " is-start-screen" : ""}${
            showEditorChrome && !rightSidebarExpanded ? " is-right-sidebar-collapsed" : ""
          }`}
        >
          {loading || workBusy ? (
            <div className="load-bar" aria-hidden>
              <div
                className="load-bar-fill"
                style={{
                  width: `${Math.round(
                    Math.min(1, Math.max(0, loading ? loadProgress : workProgress)) * 100,
                  )}%`,
                }}
              />
            </div>
          ) : null}
          {showStartScreenLogoSpinner ? (
            <div
              className="viewport-start-screen-spinner"
              role="status"
              aria-live="polite"
              aria-label="Loading scene"
            >
              <div className="viewport-start-screen-spinner-ring" aria-hidden />
            </div>
          ) : null}
          <div
            ref={viewportRef}
            className={
              interactionMode === "navigate"
                ? "viewport viewport-mode-navigate"
                : interactionMode === "fly" || interactionMode === "walk"
                  ? "viewport viewport-mode-fly"
                  : "viewport viewport-mode-edit"
            }
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerLeave}
            onGotPointerCapture={onGotPointerCapture}
            onLostPointerCapture={onLostPointerCapture}
            onContextMenu={(ev) => {
              ev.preventDefault();
              const m = interactionModeRef.current;
              if ((m === "stamp" || m === "punch") && !loading && !workBusy) {
                const el = viewportRef.current;
                const rect = el?.getBoundingClientRect();
                if (rect && rect.width > 0 && rect.height > 0) {
                  const nx = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
                  const ny = Math.min(1, Math.max(0, (ev.clientY - rect.top) / rect.height));
                  void invoke<[number, number, number] | null>("stamp_face_normal_at_screen", {
                    args: { nx, ny },
                  })
                    .then((normal) => {
                      if (!normal) return;
                      const [fnx, fny, fnz] = normal;
                      if (fnx !== 0) setStampRotX((v) => v + 90);
                      else if (fny !== 0) setStampRotY((v) => v + 90);
                      else if (fnz !== 0) setStampRotZ((v) => v + 90);
                    })
                    .catch(() => {});
                }
              }
            }}
            onWheel={onWheel}
            role="application"
            aria-label="3D viewport"
          >
            {showEditorChrome && viewportCursorDebugEnabled ? (
              <div className="viewport-cursor-debug-overlay" aria-hidden>
                {(() => {
                  const jsPct = viewportCursorDebugJs
                    ? viewportCursorOverlayPercent(
                        viewportCursorDebugJs.nx,
                        viewportCursorDebugJs.ny,
                      )
                    : null;
                  const rustPct =
                    viewportCursorDebugRust?.previewNx != null &&
                    viewportCursorDebugRust.previewNy != null
                      ? viewportCursorOverlayPercent(
                          viewportCursorDebugRust.previewNx,
                          viewportCursorDebugRust.previewNy,
                        )
                      : null;
                  return (
                    <>
                      {jsPct ? (
                        <div
                          className="viewport-cursor-debug-mark viewport-cursor-debug-mark-js"
                          style={{
                            left: `${jsPct.leftPct}%`,
                            top: `${jsPct.topPct}%`,
                          }}
                        />
                      ) : null}
                      {rustPct ? (
                        <div
                          className="viewport-cursor-debug-mark viewport-cursor-debug-mark-rust"
                          style={{
                            left: `${rustPct.leftPct}%`,
                            top: `${rustPct.topPct}%`,
                          }}
                        />
                      ) : null}
                    </>
                  );
                })()}
                <div className="viewport-cursor-debug-legend">
                  <div>
                    JS{" "}
                    {viewportCursorDebugJs
                      ? `${viewportCursorDebugJs.nx.toFixed(5)}, ${viewportCursorDebugJs.ny.toFixed(5)}`
                      : "—"}
                  </div>
                  <div>
                    Rust preview{" "}
                    {viewportCursorDebugRust?.previewNx != null &&
                    viewportCursorDebugRust.previewNy != null
                      ? `${viewportCursorDebugRust.previewNx.toFixed(5)}, ${viewportCursorDebugRust.previewNy.toFixed(5)}`
                      : "—"}
                  </div>
                  <div>
                    Δn{" "}
                    {viewportCursorDebugJs &&
                    viewportCursorDebugRust?.previewNx != null &&
                    viewportCursorDebugRust.previewNy != null
                      ? `${(viewportCursorDebugJs.nx - viewportCursorDebugRust.previewNx).toFixed(5)}, ${(viewportCursorDebugJs.ny - viewportCursorDebugRust.previewNy).toFixed(5)}`
                      : "—"}
                  </div>
                  <div>
                    proj cube (face hit){" "}
                    {viewportCursorDebugRust?.projCubeNx != null &&
                    viewportCursorDebugRust.projCubeNy != null
                      ? `${viewportCursorDebugRust.projCubeNx.toFixed(5)}, ${viewportCursorDebugRust.projCubeNy.toFixed(5)}`
                      : "—"}
                    {viewportCursorDebugJs &&
                    viewportCursorDebugRust?.projCubeNx != null &&
                    viewportCursorDebugRust.projCubeNy != null
                      ? ` · Δ ${(viewportCursorDebugJs.nx - viewportCursorDebugRust.projCubeNx).toFixed(5)}, ${(viewportCursorDebugJs.ny - viewportCursorDebugRust.projCubeNy).toFixed(5)}`
                      : ""}
                  </div>
                  <div>
                    proj center (voxel ctr, debug){" "}
                    {viewportCursorDebugRust?.projCenterNx != null &&
                    viewportCursorDebugRust.projCenterNy != null
                      ? `${viewportCursorDebugRust.projCenterNx.toFixed(5)}, ${viewportCursorDebugRust.projCenterNy.toFixed(5)}`
                      : "—"}
                    {viewportCursorDebugJs &&
                    viewportCursorDebugRust?.projCenterNx != null &&
                    viewportCursorDebugRust.projCenterNy != null
                      ? ` · Δ ${(viewportCursorDebugJs.nx - viewportCursorDebugRust.projCenterNx).toFixed(5)}, ${(viewportCursorDebugJs.ny - viewportCursorDebugRust.projCenterNy).toFixed(5)}`
                      : ""}
                  </div>
                  <div>
                    viewport{" "}
                    {viewportCursorDebugRust
                      ? `${viewportCursorDebugRust.viewportWidth}×${viewportCursorDebugRust.viewportHeight}`
                      : "—"}
                    {viewportCursorDebugRust?.texelSx != null &&
                    viewportCursorDebugRust.texelSy != null
                      ? ` · texel ${viewportCursorDebugRust.texelSx.toFixed(2)}, ${viewportCursorDebugRust.texelSy.toFixed(2)}`
                      : ""}
                  </div>
                  <div>
                    screen client{" "}
                    {viewportCursorDebugScreen
                      ? `${viewportCursorDebugScreen.clientX.toFixed(1)}, ${viewportCursorDebugScreen.clientY.toFixed(1)}`
                      : "—"}
                    {" · rel "}
                    {viewportCursorDebugScreen
                      ? `${viewportCursorDebugScreen.relX.toFixed(2)}, ${viewportCursorDebugScreen.relY.toFixed(2)}`
                      : "—"}
                  </div>
                  <div>
                    layout→surface origin{" "}
                    {viewportCursorDebugRust &&
                    viewportCursorDebugScreen &&
                    viewportCursorDebugScreen.layoutWidth > 0 &&
                    viewportCursorDebugScreen.layoutHeight > 0
                      ? (() => {
                          const s = viewportCursorDebugScreen;
                          const r = viewportCursorDebugRust;
                          const expX = Math.round((s.rectLeft / s.layoutWidth) * r.surfaceWidth);
                          const expY = Math.round((s.rectTop / s.layoutHeight) * r.surfaceHeight);
                          const dx = expX - r.viewportOriginX;
                          const dy = expY - r.viewportOriginY;
                          return `expect (${expX}, ${expY}) · Rust (${r.viewportOriginX}, ${r.viewportOriginY}) · Δ (${dx}, ${dy}) · surface ${r.surfaceWidth}×${r.surfaceHeight} · layout ${s.layoutWidth}×${s.layoutHeight} · inner ${s.innerWidth}×${s.innerHeight} · rect @ ${s.rectLeft.toFixed(0)},${s.rectTop.toFixed(0)}`;
                        })()
                      : "—"}
                  </div>
                  <div>
                    world ray (Rust, preview texels) o{" "}
                    {viewportCursorDebugRust?.rayOriginX != null &&
                    viewportCursorDebugRust.rayOriginY != null &&
                    viewportCursorDebugRust.rayOriginZ != null
                      ? `${viewportCursorDebugRust.rayOriginX.toFixed(4)}, ${viewportCursorDebugRust.rayOriginY.toFixed(4)}, ${viewportCursorDebugRust.rayOriginZ.toFixed(4)}`
                      : "—"}
                    {" d "}
                    {viewportCursorDebugRust?.rayDirX != null &&
                    viewportCursorDebugRust.rayDirY != null &&
                    viewportCursorDebugRust.rayDirZ != null
                      ? `${viewportCursorDebugRust.rayDirX.toFixed(4)}, ${viewportCursorDebugRust.rayDirY.toFixed(4)}, ${viewportCursorDebugRust.rayDirZ.toFixed(4)}`
                      : "—"}
                  </div>
                </div>
              </div>
            ) : null}
            {showViewportTopCenterStack ? (
              <div className="viewport-top-center-hud" onPointerDown={(e) => e.stopPropagation()}>
                {viewportTopCenterHud ? (
                  <div className="viewport-work-phase-chip" role="status" aria-live="polite">
                    <span className="viewport-work-phase-text">{viewportTopCenterHud.label}</span>
                    <span className="viewport-work-phase-pct">{viewportTopCenterHud.pct}%</span>
                    {viewportTopCenterHud.showFillCancel ? (
                      <button
                        type="button"
                        className="viewport-work-phase-cancel"
                        onClick={() => void invoke("voxel_fill_cancel").catch(() => {})}
                      >
                        Cancel
                      </button>
                    ) : null}
                  </div>
                ) : null}
                {cuboidPhase.active || cylinderPhase.active || polygonPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label={
                      cuboidPhase.active
                        ? "Cuboid extrusion depth"
                        : polygonPhase.active
                          ? "Polygon extrusion depth"
                          : "Cylinder extrusion depth"
                    }
                  >
                    <span>Depth</span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => {
                        if (cuboidPhase.active) {
                          const n = Math.max(-256, cuboidDepthUi - 1);
                          cuboidDepthRef.current = n;
                          setCuboidDepthUi(n);
                          if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
                        } else if (polygonPhase.active) {
                          const n = Math.max(-256, polygonDepthUi - 1);
                          polygonDepthRef.current = n;
                          setPolygonDepthUi(n);
                          if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
                        } else {
                          const n = Math.max(-256, cylinderDepthUi - 1);
                          cylinderDepthRef.current = n;
                          setCylinderDepthUi(n);
                          if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
                        }
                      }}
                    >
                      −
                    </button>
                    <input
                      type="text"
                      inputMode="numeric"
                      className="viewport-cuboid-depth-input"
                      aria-label="Extrusion depth"
                      autoComplete="off"
                      value={
                        extrusionDepthEditing
                          ? extrusionDepthDraft
                          : String(
                              cuboidPhase.active
                                ? cuboidDepthUi
                                : polygonPhase.active
                                  ? polygonDepthUi
                                  : cylinderDepthUi,
                            )
                      }
                      onChange={(e) => {
                        const v = e.target.value;
                        if (v === "" || /^-?\d*$/.test(v)) {
                          setExtrusionDepthDraft(v);
                        }
                      }}
                      onFocus={(e) => {
                        const cur = cuboidPhase.active
                          ? cuboidDepthUi
                          : polygonPhase.active
                            ? polygonDepthUi
                            : cylinderDepthUi;
                        setExtrusionDepthEditing(true);
                        setExtrusionDepthDraft(String(cur));
                        e.target.select();
                      }}
                      onBlur={() => {
                        const current = cuboidPhase.active
                          ? cuboidDepthUi
                          : polygonPhase.active
                            ? polygonDepthUi
                            : cylinderDepthUi;
                        let n = parseInt(extrusionDepthDraft, 10);
                        if (Number.isNaN(n)) n = current;
                        n = Math.max(-256, Math.min(256, n));
                        if (cuboidPhase.active) {
                          setCuboidDepthUi(n);
                          cuboidDepthRef.current = n;
                        } else if (polygonPhase.active) {
                          setPolygonDepthUi(n);
                          polygonDepthRef.current = n;
                        } else {
                          setCylinderDepthUi(n);
                          cylinderDepthRef.current = n;
                        }
                        setExtrusionDepthEditing(false);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          (e.target as HTMLInputElement).blur();
                        }
                      }}
                    />
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => {
                        if (cuboidPhase.active) {
                          const n = Math.min(256, cuboidDepthUi + 1);
                          cuboidDepthRef.current = n;
                          setCuboidDepthUi(n);
                          if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
                        } else if (polygonPhase.active) {
                          const n = Math.min(256, polygonDepthUi + 1);
                          polygonDepthRef.current = n;
                          setPolygonDepthUi(n);
                          if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
                        } else {
                          const n = Math.min(256, cylinderDepthUi + 1);
                          cylinderDepthRef.current = n;
                          setCylinderDepthUi(n);
                          if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
                        }
                      }}
                    >
                      +
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => {
                        if (cuboidPhase.active) void commitCuboidSolidAtScreen();
                        else if (polygonPhase.active) void commitPolygonSolid();
                        else void commitCylinderSolidAtScreen();
                      }}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {extrudePhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Extrude settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Extrude</span>
                    <span
                      style={{
                        fontSize: "0.78rem",
                        color: "var(--app-text-muted)",
                      }}
                    >
                      {extrudeProfile === "cylinder" ? "Cylinder" : "Cube"}
                      {extrudeTaper ? " tapered" : ""}
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => extrudePhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => extrudePhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {/* Roof placing phase: pin/anchor count + Cancel */}
                {generatorKind === "roof" && (roofPins.length > 0 || roofFirstClick !== null) ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Roof placement"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Roof</span>
                    <span
                      style={{
                        fontSize: "0.78rem",
                        color: "var(--app-text-muted)",
                      }}
                    >
                      {roofFirstClick !== null
                        ? roofAreaShape === "circle"
                          ? "click edge"
                          : "click opposite corner"
                        : `${roofPins.length} pin${roofPins.length !== 1 ? "s" : ""}`}
                    </span>
                    <div className="viewport-polygon-phase-actions">
                      <button
                        type="button"
                        className="tool-options-shape-btn"
                        disabled={loading || workBusy || roofPins.length < 3}
                        onClick={() => {
                          void invoke("generator_roof_from_pins_cmd", {
                            args: {
                              pins: roofPins,
                              style: roofStyle,
                              height: roofHeight,
                              thickness: 1,
                              shedEdgeIndex: 0,
                              gableOrientation: 0,
                              breakRatio: 0.5,
                              wallHeight: 3,
                              parapetHeight: 2,
                              saltSkew: 0,
                              hollow: roofHollow,
                              color: activeColor,
                              material: activeMaterialRef.current,
                            },
                          })
                            .then(() => {
                              setRoofPins([]);
                              roofPinsRef.current = [];
                              void invoke("voxel_stroke_preview_reset").catch(() => {});
                            })
                            .catch(() => {});
                        }}
                      >
                        Done
                      </button>
                      <button
                        type="button"
                        className="tool-options-shape-btn"
                        onClick={() => {
                          setRoofPins([]);
                          roofPinsRef.current = [];
                          roofFirstClickRef.current = null;
                          setRoofFirstClick(null);
                          void invoke("voxel_stroke_preview_reset").catch(() => {});
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : null}
                {/* Cloth placing phase: pin count + Done/Cancel */}
                {generatorKind === "cloth" && !clothPhase.active && clothPins.length > 0 ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Cloth pin placement"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Cloth</span>
                    <span
                      style={{
                        fontSize: "0.78rem",
                        color: "var(--app-text-muted)",
                      }}
                    >
                      {clothPins.length} pin{clothPins.length !== 1 ? "s" : ""}
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => {
                        setClothPins([]);
                        clothPinsRef.current = [];
                        void invoke("voxel_stroke_preview_reset").catch(() => {});
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      disabled={clothPins.length < 3}
                      onClick={() => clothPhase.enter("settings", {})}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {/* Cloth settings phase: tension slider + Done/Cancel */}
                {clothPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Cloth settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Cloth</span>
                    <span
                      style={{
                        fontSize: "0.78rem",
                        color: "var(--app-text-muted)",
                      }}
                    >
                      Tension
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={clothTension}
                      onChange={(ev) => setClothTension(Number(ev.target.value))}
                      style={{ width: 80 }}
                      title="0 = loose drape, 1 = stiff"
                    />
                    <span
                      style={{
                        fontSize: "0.78rem",
                        minWidth: "2.2em",
                        textAlign: "right",
                      }}
                    >
                      {clothTension.toFixed(2)}
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => clothPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => clothPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {/* Rope settings phase: tension + sag sliders + Done/Cancel */}
                {ropePhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Rope settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Rope</span>
                    <span
                      style={{
                        fontSize: "0.78rem",
                        color: "var(--app-text-muted)",
                      }}
                    >
                      Tension
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={ropeTension}
                      onChange={(ev) => setRopeTension(Number(ev.target.value))}
                      style={{ width: 60 }}
                      title="0 = loose, 1 = taut"
                    />
                    <span
                      style={{
                        fontSize: "0.78rem",
                        minWidth: "2.2em",
                        textAlign: "right",
                      }}
                    >
                      {ropeTension.toFixed(2)}
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => ropePhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => ropePhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {rocksPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Rocks settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Rocks</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => rocksPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => rocksPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {grassPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Grass settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Grass</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => grassPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => grassPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {ashlarPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Ashlar settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Ashlar</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => ashlarPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => ashlarPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {floraPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Flora settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Flora</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => floraPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => floraPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {piscinaPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Piscina settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Fish</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => piscinaPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => piscinaPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {insectaPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Insecta settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Insect</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => insectaPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => insectaPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {faunaPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Fauna settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Fauna</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => faunaPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => faunaPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {squishyPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Squishy session"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Squishy</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to commit, Esc to cancel
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => squishyPhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => squishyPhase.commit()}
                    >
                      Done
                    </button>
                  </div>
                ) : null}
                {showWallSculptPolygonHud ? (
                  <div
                    className="viewport-polygon-phase-bar"
                    role="dialog"
                    aria-label="Wall polygon outline"
                  >
                    <p className="viewport-polygon-phase-hint">
                      Wall outline: {wallSculptPolygonVerts.length} corners. Click the surface to
                      add; Done applies (min 2 corners).
                    </p>
                    <div className="viewport-polygon-phase-actions">
                      <button
                        type="button"
                        className="tool-options-shape-btn"
                        disabled={loading || workBusy}
                        onClick={() => setWallSculptPolygonVerts([])}
                      >
                        Clear
                      </button>
                      <button
                        type="button"
                        className="tool-options-shape-btn"
                        disabled={loading || workBusy || wallSculptPolygonVerts.length < 2}
                        onClick={() => commitWallSculptPolygonStroke()}
                      >
                        Done
                      </button>
                    </div>
                  </div>
                ) : null}
                {showPolygonPhaseHud && !polygonPhase.active ? (
                  <div
                    className="viewport-polygon-phase-bar"
                    role="dialog"
                    aria-label="Polygon area"
                  >
                    <p className="viewport-polygon-phase-hint">
                      Vertices: {strokePolygonVerts.length}. Click to add corners; Apply with three
                      or more.
                    </p>
                    <div className="viewport-polygon-phase-actions">
                      <button
                        type="button"
                        className="tool-options-shape-btn"
                        disabled={loading || workBusy}
                        onClick={() => {
                          setStrokePolygonVerts([]);
                          strokePolygonLastScreenRef.current = null;
                        }}
                      >
                        Clear
                      </button>
                      <button
                        type="button"
                        className="tool-options-shape-btn"
                        disabled={loading || workBusy || strokePolygonVerts.length < 3}
                        onClick={() => {
                          if (isDrawVoxelEditMode) applyPolygonStrokeFill();
                          else applyPolygonStrokeFill();
                        }}
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}
            <SelectionGizmo
              ref={gizmoRef}
              selectionCount={selectionCount}
              flyMode={interactionMode === "fly" || interactionMode === "walk"}
              loadingOrBusy={loading || workBusy}
              stampOrPunch={interactionMode === "stamp" || interactionMode === "punch"}
              viewportEl={viewportRef.current}
            />
            <ExtrudeGizmo ref={extrudeGizmoRef} viewportEl={viewportRef.current} />
          </div>
          {showEditorChrome ? (
            <ViewportCameraHud
              flyMode={interactionMode === "fly" || interactionMode === "walk"}
              loadingOrBusy={loading || workBusy}
            />
          ) : null}
          {showToolOptionsPanel ? (
            <div
              className={`tool-options-panel${toolsPaneFloating ? " is-tools-floating" : ""}${
                !toolsPaneFloating && sidebarExpanded ? " is-sidebar-expanded" : ""
              }${!toolsPaneFloating && !sidebarExpanded ? " is-sidebar-collapsed" : ""}${
                toolsPane === "generators" ? " is-generator-wide" : ""
              }`}
              role="dialog"
              aria-label="Tool options"
              onPointerDown={(e) => e.stopPropagation()}
              onPointerUp={(e) => e.stopPropagation()}
            >
              {selectionCount > 0 ? (
                <div className="tool-options-section tool-panel-selection-toolbar">
                  <div
                    className="tool-options-shape-row"
                    style={{
                      justifyContent: "space-between",
                      alignItems: "center",
                      flexWrap: "wrap",
                      gap: "0.5rem",
                    }}
                  >
                    <span className="tool-panel-selection-count" role="status" aria-live="polite">
                      {selectionCount} selected
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      disabled={loading || workBusy}
                      onClick={() => {
                        void invoke("selection_clear").catch(() => {});
                      }}
                    >
                      Deselect
                    </button>
                  </div>
                </div>
              ) : null}
              {(toolsPane === "draw" || toolsPane === "select") &&
              drawStrokeMode === "fill" &&
              (interactionMode === "add" ||
                interactionMode === "remove" ||
                interactionMode === "paint" ||
                isSelectionInteractionMode) ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Fill</div>
                  <p className="tool-options-hint">
                    Click a solid voxel. The connected region is filled, recolored, or added to the
                    selection per your current tool and the options below.
                  </p>
                </div>
              ) : null}
              {(toolsPane === "draw" || toolsPane === "select") && isSelectionInteractionMode ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Combine</div>
                  <div
                    className="tool-options-shape-row tool-options-shape-row-two"
                    role="group"
                    aria-label="Selection combine mode"
                  >
                    {(
                      [
                        ["replace", "Replace"],
                        ["intersect", "Intersect"],
                        ["add", "Add"],
                        ["subtract", "Subtract"],
                      ] as const
                    ).map(([id, label]) => (
                      <button
                        key={id}
                        type="button"
                        className={
                          selectionCombineMode === id
                            ? "tool-options-shape-btn is-active"
                            : "tool-options-shape-btn"
                        }
                        disabled={loading || workBusy}
                        onClick={() => {
                          setSelectionCombineMode(id);
                          void invoke("selection_set_combine_mode", {
                            mode: id,
                          }).catch(() => {});
                        }}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}
              {showDrawPaneToolMatrix ? (
                <>
                  <DrawPaneSelectionToolOptions
                    loading={loading}
                    workBusy={workBusy}
                    selectionMethod={selectionMethod}
                    drawStrokeMode={drawStrokeMode}
                    setDrawStrokeMode={setDrawStrokeMode}
                    strokeDrawStyle={strokeDrawStyle}
                    setStrokeDrawStyle={setStrokeDrawStyle}
                    strokeFamilyVariant={strokeFamilyVariant}
                    setStrokeFamilyVariant={setStrokeFamilyVariant}
                    planeAxis={planeAxis}
                    setPlaneAxis={setPlaneAxis}
                    fillSelectDiagonals={fillSelectDiagonals}
                    setFillSelectDiagonals={setFillSelectDiagonals}
                    fillRespectsColor={fillRespectsColor}
                    setFillRespectsColor={setFillRespectsColor}
                    sprayDensity={sprayDensity}
                    setSprayDensity={setSprayDensity}
                    brushShape={brushShape}
                    setBrushShape={setBrushShape}
                    brushClipBottomHalf={brushClipBottomHalf}
                    setBrushClipBottomHalf={setBrushClipBottomHalf}
                    brushRadius={brushRadius}
                    setBrushRadius={setBrushRadius}
                    selectionStrokeSnapToSurface={selectionStrokeSnapToSurface}
                    setSelectionStrokeSnapToSurface={setSelectionStrokeSnapToSurface}
                    selectionStrokeAxisAlign={selectionStrokeAxisAlign}
                    setSelectionStrokeAxisAlign={setSelectionStrokeAxisAlign}
                    surfacePlaneHollow={surfacePlaneHollow}
                    setSurfacePlaneHollow={setSurfacePlaneHollow}
                    sprayConstrainToPlane={sprayConstrainToPlane}
                    setSprayConstrainToPlane={setSprayConstrainToPlane}
                    spraySizeRange={spraySizeRange}
                    setSpraySizeRange={setSpraySizeRange}
                    sprayScatter={sprayScatter}
                    setSprayScatter={setSprayScatter}
                    sprayRadiusMin={sprayRadiusMin}
                    setSprayRadiusMin={setSprayRadiusMin}
                    sprayRadiusMax={sprayRadiusMax}
                    setSprayRadiusMax={setSprayRadiusMax}
                    sprayBrushShape={sprayBrushShape}
                    setSprayBrushShape={setSprayBrushShape}
                    sprayConstrainToPlaneRef={sprayConstrainToPlaneRef_}
                    setSprayConstrainToPlaneRef={setSprayConstrainToPlaneRef_}
                    fillConstrainToPlane={fillConstrainToPlane}
                    setFillConstrainToPlane={setFillConstrainToPlane}
                  />
                </>
              ) : null}
              {(toolsPane === "draw" || toolsPane === "select") && isSelectionInteractionMode ? (
                <>
                  {interactionMode === "selectByColor" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">By color</div>
                      <p className="tool-options-hint">
                        Click a voxel to select all connected voxels of the same color.
                      </p>
                      <label
                        className="tool-options-range-label"
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.35rem",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={matchMaterialSelectColor}
                          onChange={(ev) => setMatchMaterialSelectColor(ev.target.checked)}
                          disabled={loading || workBusy}
                        />
                        <span>Match material when matching color</span>
                      </label>
                    </div>
                  ) : null}
                  {interactionMode === "selectCoplanar" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">Coplanar</div>
                      <p className="tool-options-hint">
                        Click a solid voxel to extend the selection along the same face plane.
                      </p>
                    </div>
                  ) : null}
                  {interactionMode === "selectCoplanarEmpty" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">Coplanar void</div>
                      <p className="tool-options-hint">
                        Click empty space on a plane to select the coplanar empty region.
                      </p>
                    </div>
                  ) : null}
                </>
              ) : null}
              {toolsPane === "sculpt" && interactionMode === "sculpt" ? (
                <>
                  {sculptStrokeMode === "draw" ||
                  sculptStrokeMode === "smooth" ||
                  sculptStrokeMode === "gouge" ||
                  sculptStrokeMode === "extrude" ||
                  sculptStrokeMode === "terrain" ? (
                    <div className="tool-options-section" aria-label="Sculpt">
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Brush</span>
                        <input
                          type="range"
                          min={0}
                          max={SCULPT_BRUSH_MAX_INDEX}
                          value={sculptBrushRadius}
                          onChange={(ev) => setSculptBrushRadius(Number(ev.target.value))}
                          disabled={loading || workBusy}
                          title="Brush size (1–64 voxels)"
                        />
                        <span className="tool-options-range-value">{sculptBrushRadius + 1}</span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Strength</span>
                        <input
                          type="range"
                          min={1}
                          max={100}
                          value={sculptBrushStrength}
                          onChange={(ev) => setSculptBrushStrength(Number(ev.target.value))}
                          disabled={loading || workBusy}
                          title="How strongly the brush applies (with falloff)"
                        />
                        <span className="tool-options-range-value">{sculptBrushStrength}</span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Falloff</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={sculptBrushFalloff}
                          onChange={(ev) => setSculptBrushFalloff(Number(ev.target.value))}
                          disabled={loading || workBusy}
                          title="0 = hard edge; higher = softer falloff toward brush radius"
                        />
                        <span className="tool-options-range-value">{sculptBrushFalloff}</span>
                      </label>
                      {sculptStrokeMode === "draw" || sculptStrokeMode === "smooth" ? (
                        <label
                          className="tool-options-checkbox-row"
                          style={{ marginTop: "0.35rem" }}
                        >
                          <input
                            type="checkbox"
                            checked={brushClipBottomHalf}
                            onChange={(ev) => setBrushClipBottomHalf(ev.target.checked)}
                            disabled={loading || workBusy}
                          />
                          <span title="Uses the clicked face outward normal (world +Y if no solid hit)">
                            Outer half (face)
                          </span>
                        </label>
                      ) : null}
                      {sculptStrokeMode === "draw" ||
                      sculptStrokeMode === "smooth" ||
                      sculptStrokeMode === "gouge" ? (
                        <>
                          <div className="tool-options-heading" style={{ marginTop: "0.35rem" }}>
                            Brush shape
                          </div>
                          <div
                            className="tool-options-shape-row tool-options-shape-row-two"
                            role="group"
                            aria-label="Sculpt brush shape"
                          >
                            {(
                              [
                                ["circle", "Circle"],
                                ["square", "Square"],
                              ] as const
                            ).map(([id, label]) => (
                              <button
                                key={id}
                                type="button"
                                className={
                                  sculptBrushShapeUi === id
                                    ? "tool-options-shape-btn is-active"
                                    : "tool-options-shape-btn"
                                }
                                disabled={loading || workBusy}
                                onClick={() => setSculptBrushShapeUi(id)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                          <div
                            className="tool-options-shape-row tool-options-shape-row-two"
                            role="group"
                            aria-label="Sculpt brush shape 3D"
                          >
                            {(
                              [
                                ["sphere", "Sphere"],
                                ["cube", "Cube"],
                              ] as const
                            ).map(([id, label]) => (
                              <button
                                key={id}
                                type="button"
                                className={
                                  sculptBrushShapeUi === id
                                    ? "tool-options-shape-btn is-active"
                                    : "tool-options-shape-btn"
                                }
                                disabled={loading || workBusy}
                                onClick={() => setSculptBrushShapeUi(id)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </>
                      ) : null}
                      {sculptStrokeMode === "extrude" ? (
                        <>
                          <div className="tool-options-heading" style={{ marginTop: "0.35rem" }}>
                            Direction
                          </div>
                          <div
                            className="tool-options-shape-row"
                            role="group"
                            aria-label="Extrude direction reference"
                            style={{
                              display: "grid",
                              gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr",
                              gap: "0.25rem",
                            }}
                          >
                            {(
                              [
                                ["auto", "Auto"],
                                ["camera", "Cam"],
                                ["x", "X"],
                                ["y", "Y"],
                                ["z", "Z"],
                              ] as const
                            ).map(([id, label]) => (
                              <button
                                key={id}
                                type="button"
                                className={
                                  extrudeDirectionRef === id
                                    ? "tool-options-shape-btn is-active"
                                    : "tool-options-shape-btn"
                                }
                                disabled={loading || workBusy}
                                onClick={() => setExtrudeDirectionRef(id)}
                                title={
                                  id === "auto"
                                    ? "Along dominant axis of start face"
                                    : id === "camera"
                                      ? "View plane: drag maps through camera right/up"
                                      : `World ±${id.toUpperCase()} (sign from drag)`
                                }
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                          <div className="tool-options-heading" style={{ marginTop: "0.35rem" }}>
                            Profile
                          </div>
                          <div
                            className="tool-options-shape-row tool-options-shape-row-two"
                            role="group"
                            aria-label="Extrude profile"
                          >
                            {(
                              [
                                ["cube", "Cube"],
                                ["cylinder", "Cylinder"],
                              ] as const
                            ).map(([id, label]) => (
                              <button
                                key={id}
                                type="button"
                                className={
                                  extrudeProfile === id
                                    ? "tool-options-shape-btn is-active"
                                    : "tool-options-shape-btn"
                                }
                                disabled={loading || workBusy}
                                onClick={() => setExtrudeProfile(id)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                          {extrudeProfile === "cylinder" ? (
                            <>
                              <div
                                className="tool-options-heading"
                                style={{ marginTop: "0.35rem" }}
                              >
                                End caps
                              </div>
                              <div
                                className="tool-options-shape-row"
                                role="group"
                                aria-label="Extrude end cap"
                                style={{
                                  display: "grid",
                                  gridTemplateColumns: "1fr 1fr 1fr",
                                  gap: "0.25rem",
                                }}
                              >
                                {(
                                  [
                                    ["flat", "Flat"],
                                    ["rounded", "Rounded"],
                                    ["pointed", "Pointed"],
                                  ] as const
                                ).map(([id, label]) => (
                                  <button
                                    key={id}
                                    type="button"
                                    className={
                                      extrudeEndCap === id
                                        ? "tool-options-shape-btn is-active"
                                        : "tool-options-shape-btn"
                                    }
                                    disabled={loading || workBusy}
                                    onClick={() => setExtrudeEndCap(id)}
                                  >
                                    {label}
                                  </button>
                                ))}
                              </div>
                            </>
                          ) : null}
                          <label
                            className="tool-options-checkbox-row"
                            style={{ marginTop: "0.35rem" }}
                          >
                            <input
                              type="checkbox"
                              checked={extrudeTaper}
                              onChange={(ev) => setExtrudeTaper(ev.target.checked)}
                              disabled={loading || workBusy}
                            />
                            <span>Taper</span>
                          </label>
                          {extrudeTaper ? (
                            <>
                              <label className="tool-options-range-label tool-options-range-with-value">
                                <span>Start</span>
                                <input
                                  type="range"
                                  min={0}
                                  max={24}
                                  value={extrudeTaperStart}
                                  onChange={(ev) => setExtrudeTaperStart(Number(ev.target.value))}
                                  disabled={loading || workBusy}
                                />
                                <span className="tool-options-range-value">
                                  {extrudeTaperStart + 1}
                                </span>
                              </label>
                              <label className="tool-options-range-label tool-options-range-with-value">
                                <span>End</span>
                                <input
                                  type="range"
                                  min={0}
                                  max={24}
                                  value={extrudeTaperEnd}
                                  onChange={(ev) => setExtrudeTaperEnd(Number(ev.target.value))}
                                  disabled={loading || workBusy}
                                />
                                <span className="tool-options-range-value">
                                  {extrudeTaperEnd + 1}
                                </span>
                              </label>
                            </>
                          ) : null}
                        </>
                      ) : null}
                      {sculptStrokeMode === "terrain" ? (
                        <>
                          <div className="tool-options-heading" style={{ marginTop: "0.35rem" }}>
                            Brush shape
                          </div>
                          <div
                            className="tool-options-shape-row tool-options-shape-row-two"
                            role="group"
                            aria-label="Terrain brush shape (horizontal XZ)"
                          >
                            <button
                              type="button"
                              className={
                                sculptBrushShapeUi === "circle" || sculptBrushShapeUi === "sphere"
                                  ? "tool-options-shape-btn is-active"
                                  : "tool-options-shape-btn"
                              }
                              disabled={loading || workBusy}
                              onClick={() => setSculptBrushShapeUi("circle")}
                              title="Circular footprint in XZ"
                            >
                              Circle
                            </button>
                            <button
                              type="button"
                              className={
                                sculptBrushShapeUi === "square" || sculptBrushShapeUi === "cube"
                                  ? "tool-options-shape-btn is-active"
                                  : "tool-options-shape-btn"
                              }
                              disabled={loading || workBusy}
                              onClick={() => setSculptBrushShapeUi("square")}
                              title="Square footprint in XZ"
                            >
                              Square
                            </button>
                          </div>
                          <div className="tool-options-heading" style={{ marginTop: "0.35rem" }}>
                            Terrain
                          </div>
                          <div
                            className="tool-options-shape-row"
                            style={{
                              display: "grid",
                              gridTemplateColumns: "1fr 1fr 1fr",
                              gap: "0.25rem",
                            }}
                            role="group"
                            aria-label="Terrain operation"
                          >
                            {(
                              [
                                ["raise", "Raise"],
                                ["lower", "Lower"],
                                ["smooth", "Smooth"],
                                ["flatten", "Flatten"],
                                ["erode", "Erode"],
                              ] as const
                            ).map(([id, label]) => (
                              <button
                                key={id}
                                type="button"
                                className={
                                  terrainSculptOp === id
                                    ? "tool-options-shape-btn is-active"
                                    : "tool-options-shape-btn"
                                }
                                disabled={loading || workBusy}
                                onClick={() => setTerrainSculptOp(id)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                          {terrainHoverY !== null ? (
                            <div
                              className="tool-options-hint"
                              style={{ marginTop: "0.25rem", opacity: 0.7 }}
                            >
                              Surface Y: <strong>{terrainHoverY}</strong>
                            </div>
                          ) : null}
                          {terrainSculptOp !== "erode" ? (
                            <label
                              className="tool-options-range-label"
                              style={{ marginTop: "0.35rem" }}
                            >
                              <span>Base Y</span>
                              <input
                                type="number"
                                value={terrainBaseY}
                                min={-512}
                                max={512}
                                step={1}
                                onChange={(ev) => {
                                  const n = Number(ev.target.value);
                                  if (Number.isNaN(n)) return;
                                  setTerrainBaseY(Math.max(-512, Math.min(512, n)));
                                }}
                                disabled={loading || workBusy}
                              />
                            </label>
                          ) : null}
                          {terrainSculptOp === "raise" || terrainSculptOp === "lower" ? (
                            <label
                              className="tool-options-checkbox-row"
                              style={{ marginTop: "0.2rem" }}
                            >
                              <input
                                type="checkbox"
                                checked={terrainSubVoxel}
                                onChange={(ev) => setTerrainSubVoxel(ev.target.checked)}
                                disabled={loading || workBusy}
                              />
                              <span title="Accumulate fractional changes for gentle sub-voxel sculpting">
                                Sub-voxel precision
                              </span>
                            </label>
                          ) : null}
                          {terrainSculptOp === "smooth" ? (
                            <label className="tool-options-range-label tool-options-range-with-value">
                              <span>Smooth reach</span>
                              <input
                                type="range"
                                min={0}
                                max={8}
                                value={terrainSmoothRadius}
                                onChange={(ev) => setTerrainSmoothRadius(Number(ev.target.value))}
                                disabled={loading || workBusy}
                              />
                              <span className="tool-options-range-value">
                                {terrainSmoothRadius}
                              </span>
                            </label>
                          ) : null}
                          {terrainSculptOp === "flatten" ? (
                            <label
                              className="tool-options-checkbox-row"
                              style={{ marginTop: "0.35rem" }}
                            >
                              <input
                                type="checkbox"
                                checked={terrainFlattenUseBaseY}
                                onChange={(ev) => setTerrainFlattenUseBaseY(ev.target.checked)}
                                disabled={loading || workBusy}
                              />
                              <span title="Flatten to the Base Y value instead of the average surface height">
                                Use explicit Base Y
                              </span>
                            </label>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                  ) : null}
                  {sculptStrokeMode === "smooth" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">Smooth</div>
                      <div
                        className="tool-options-shape-row"
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr",
                          gap: "0.25rem",
                          marginBottom: "0.35rem",
                        }}
                        role="group"
                        aria-label="Smooth algorithm"
                      >
                        {(
                          [
                            ["majority", "Majority"],
                            ["meshLaplacian", "Mesh Laplacian"],
                          ] as const
                        ).map(([id, label]) => (
                          <button
                            key={id}
                            type="button"
                            className={
                              sculptSmoothVariant === id
                                ? "tool-options-shape-btn is-active"
                                : "tool-options-shape-btn"
                            }
                            disabled={loading || workBusy}
                            onClick={() => setSculptSmoothVariant(id)}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      {sculptSmoothVariant === "majority" ? (
                        <>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Passes</span>
                            <input
                              type="range"
                              min={1}
                              max={8}
                              value={sculptSmoothPasses}
                              onChange={(ev) => setSculptSmoothPasses(Number(ev.target.value))}
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">{sculptSmoothPasses}</span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Neighbor radius</span>
                            <input
                              type="range"
                              min={0}
                              max={6}
                              value={smoothNeighborRadius}
                              onChange={(ev) => setSmoothNeighborRadius(Number(ev.target.value))}
                              disabled={loading || workBusy}
                              title="0 = six face neighbors only"
                            />
                            <span className="tool-options-range-value">{smoothNeighborRadius}</span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Aggressiveness</span>
                            <input
                              type="range"
                              min={0}
                              max={100}
                              value={smoothAggressiveness}
                              onChange={(ev) => setSmoothAggressiveness(Number(ev.target.value))}
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">{smoothAggressiveness}</span>
                          </label>
                        </>
                      ) : (
                        <>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Iterations</span>
                            <input
                              type="range"
                              min={1}
                              max={20}
                              value={smoothLaplacianIterations}
                              onChange={(ev) =>
                                setSmoothLaplacianIterations(Number(ev.target.value))
                              }
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">
                              {smoothLaplacianIterations}
                            </span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Relax</span>
                            <input
                              type="range"
                              min={0}
                              max={100}
                              value={smoothLaplacianRelaxPct}
                              onChange={(ev) => setSmoothLaplacianRelaxPct(Number(ev.target.value))}
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">
                              {smoothLaplacianRelaxPct}
                            </span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Majority fallback radius</span>
                            <input
                              type="range"
                              min={0}
                              max={6}
                              value={smoothNeighborRadius}
                              onChange={(ev) => setSmoothNeighborRadius(Number(ev.target.value))}
                              disabled={loading || workBusy}
                              title="Neighborhood margin + mesh fallback"
                            />
                            <span className="tool-options-range-value">{smoothNeighborRadius}</span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Fallback aggressiveness</span>
                            <input
                              type="range"
                              min={0}
                              max={100}
                              value={smoothAggressiveness}
                              onChange={(ev) => setSmoothAggressiveness(Number(ev.target.value))}
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">{smoothAggressiveness}</span>
                          </label>
                        </>
                      )}
                    </div>
                  ) : null}
                  {sculptStrokeMode === "wall" ? (
                    <div className="tool-options-section" aria-label="Sculpt wall">
                      <div className="tool-options-heading">Area shape</div>
                      <div
                        className="tool-options-shape-row"
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr 1fr",
                          gap: "0.25rem",
                        }}
                        role="group"
                        aria-label="Wall area shape"
                      >
                        {(
                          [
                            ["brush", "Brush"],
                            ["circle", "Circle"],
                            ["polygon", "Polygon"],
                          ] as const
                        ).map(([id, label]) => (
                          <button
                            key={id}
                            type="button"
                            className={
                              wallAreaShape === id
                                ? "tool-options-shape-btn is-active"
                                : "tool-options-shape-btn"
                            }
                            disabled={loading || workBusy}
                            title={
                              id === "brush"
                                ? "Drag a freehand stroke on the surface"
                                : id === "circle"
                                  ? "Drag from center to edge on the face"
                                  : "Click corners for a closed outline, then Done (web)"
                            }
                            onClick={() => setWallAreaShape(id)}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      <label
                        className="tool-options-range-label"
                        style={{
                          marginTop: "0.45rem",
                          flexDirection: "row",
                          alignItems: "center",
                          gap: "0.5rem",
                        }}
                      >
                        <span style={{ minWidth: "4.5rem" }}>Direction</span>
                        <select
                          className="sidebar-material-select"
                          style={{ flex: 1, maxWidth: "12rem" }}
                          value={sprayDirection}
                          onChange={(ev) => setSprayDirection(ev.target.value as SprayDirectionApi)}
                          disabled={loading || workBusy}
                          title="Auto = face normal; or pick a world axis"
                          aria-label="Wall extrusion direction"
                        >
                          <option value="auto">Auto</option>
                          <option value="none">None</option>
                          <option value="right">X+</option>
                          <option value="left">X−</option>
                          <option value="up">Y+</option>
                          <option value="down">Y−</option>
                          <option value="back">Z+</option>
                          <option value="forward">Z−</option>
                        </select>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Width</span>
                        <input
                          type="range"
                          min={0}
                          max={SCULPT_BRUSH_MAX_INDEX}
                          value={wallWidthIndex}
                          onChange={(ev) => setWallWidthIndex(Number(ev.target.value))}
                          disabled={loading || workBusy}
                          title="Path thickness (1–64 voxels)"
                        />
                        <span className="tool-options-range-value">{wallWidthIndex + 1}</span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Height</span>
                        <input
                          type="range"
                          min={2}
                          max={20}
                          value={wallHeightVox}
                          onChange={(ev) => setWallHeightVox(Number(ev.target.value))}
                          disabled={loading || workBusy}
                          title="Voxels to extend along direction (min 2)"
                        />
                        <span className="tool-options-range-value">{wallHeightVox}</span>
                      </label>
                      <label className="tool-options-checkbox-row" style={{ marginTop: "0.35rem" }}>
                        <input
                          type="checkbox"
                          checked={wallLockStartHeight}
                          onChange={(ev) => setWallLockStartHeight(ev.target.checked)}
                          disabled={loading || workBusy}
                        />
                        <span>Lock start height</span>
                      </label>
                      <label className="tool-options-checkbox-row">
                        <input
                          type="checkbox"
                          checked={wallAxisAlign}
                          onChange={(ev) => setWallAxisAlign(ev.target.checked)}
                          disabled={loading || workBusy}
                        />
                        <span>Axis-align</span>
                      </label>
                    </div>
                  ) : null}
                </>
              ) : toolsPane === "sculpt" ? (
                <div className="tool-options-section">
                  <p className="tool-options-hint">Select Sculpt mode in the sidebar.</p>
                </div>
              ) : null}
              {toolsPane === "generators" ? (
                <GeneratorToolOptions
                  loading={loading}
                  workBusy={workBusy}
                  generatorKind={generatorKind}
                  generatorSphereRadius={generatorSphereRadius}
                  setGeneratorSphereRadius={setGeneratorSphereRadius}
                  rockRoughness={rockRoughness}
                  setRockRoughness={setRockRoughness}
                  rockCount={rockCount}
                  setRockCount={setRockCount}
                  rockClusterRadius={rockClusterRadius}
                  setRockClusterRadius={setRockClusterRadius}
                  rockSinkDirection={rockSinkDirection}
                  setRockSinkDirection={setRockSinkDirection}
                  rockSinkAmount={rockSinkAmount}
                  setRockSinkAmount={setRockSinkAmount}
                  grassDensity={grassDensity}
                  setGrassDensity={setGrassDensity}
                  grassMaxHeight={grassMaxHeight}
                  setGrassMaxHeight={setGrassMaxHeight}
                  clothGravityDirection={clothGravityDirection}
                  setClothGravityDirection={setClothGravityDirection}
                  ropeBrushShapeUi={ropeBrushShapeUi}
                  setRopeBrushShapeUi={setRopeBrushShapeUi}
                  ropeBrushRadiusIndex={ropeBrushRadiusIndex}
                  setRopeBrushRadiusIndex={setRopeBrushRadiusIndex}
                  clothSimGravityPct={clothSimGravityPct}
                  setClothSimGravityPct={setClothSimGravityPct}
                  clothSimStiffnessPct={clothSimStiffnessPct}
                  setClothSimStiffnessPct={setClothSimStiffnessPct}
                  clothSimIterations={clothSimIterations}
                  setClothSimIterations={setClothSimIterations}
                  clothSimConstraintPasses={clothSimConstraintPasses}
                  setClothSimConstraintPasses={setClothSimConstraintPasses}
                  ashlarThickness={ashlarThickness}
                  setAshlarThickness={setAshlarThickness}
                  floraPreset={floraPreset}
                  setFloraPreset={setFloraPreset}
                  floraHeight={floraHeight}
                  setFloraHeight={setFloraHeight}
                  floraGirth={floraGirth}
                  setFloraGirth={setFloraGirth}
                  floraWobble={floraWobble}
                  setFloraWobble={setFloraWobble}
                  floraTaper={floraTaper}
                  setFloraTaper={setFloraTaper}
                  floraStemCount={floraStemCount}
                  setFloraStemCount={setFloraStemCount}
                  floraClusterRadius={floraClusterRadius}
                  setFloraClusterRadius={setFloraClusterRadius}
                  floraBranchCount={floraBranchCount}
                  setFloraBranchCount={setFloraBranchCount}
                  floraBranchDepth={floraBranchDepth}
                  setFloraBranchDepth={setFloraBranchDepth}
                  floraBranchStart={floraBranchStart}
                  setFloraBranchStart={setFloraBranchStart}
                  floraBranchSpread={floraBranchSpread}
                  setFloraBranchSpread={setFloraBranchSpread}
                  floraBraidStrands={floraBraidStrands}
                  setFloraBraidStrands={setFloraBraidStrands}
                  floraBraidTwist={floraBraidTwist}
                  setFloraBraidTwist={setFloraBraidTwist}
                  floraCanopy={floraCanopy}
                  setFloraCanopy={setFloraCanopy}
                  roofAreaShape={roofAreaShape}
                  setRoofAreaShape={setRoofAreaShape}
                  roofPins={roofPins}
                  setRoofPins={setRoofPins}
                  roofPinsRef={roofPinsRef}
                  roofFirstClickRef={roofFirstClickRef}
                  setRoofFirstClick={setRoofFirstClick}
                  roofStyle={roofStyle}
                  setRoofStyle={setRoofStyle}
                  roofHeight={roofHeight}
                  setRoofHeight={setRoofHeight}
                  roofHollow={roofHollow}
                  setRoofHollow={setRoofHollow}
                  piscinaSpecies={piscinaSpecies}
                  setPiscinaSpecies={setPiscinaSpecies}
                  piscinaLength={piscinaLength}
                  setPiscinaLength={setPiscinaLength}
                  piscinaWidth={piscinaWidth}
                  setPiscinaWidth={setPiscinaWidth}
                  piscinaThickness={piscinaThickness}
                  setPiscinaThickness={setPiscinaThickness}
                  piscinaSpineBend={piscinaSpineBend}
                  setPiscinaSpineBend={setPiscinaSpineBend}
                  piscinaSpineSCurve={piscinaSpineSCurve}
                  setPiscinaSpineSCurve={setPiscinaSpineSCurve}
                  piscinaAnchorU={piscinaAnchorU}
                  setPiscinaAnchorU={setPiscinaAnchorU}
                  piscinaAnchorV={piscinaAnchorV}
                  setPiscinaAnchorV={setPiscinaAnchorV}
                  piscinaShowFinDorsal={piscinaShowFinDorsal}
                  setPiscinaShowFinDorsal={setPiscinaShowFinDorsal}
                  piscinaFinDorsal={piscinaFinDorsal}
                  setPiscinaFinDorsal={setPiscinaFinDorsal}
                  piscinaShowFinAnal={piscinaShowFinAnal}
                  setPiscinaShowFinAnal={setPiscinaShowFinAnal}
                  piscinaFinAnal={piscinaFinAnal}
                  setPiscinaFinAnal={setPiscinaFinAnal}
                  piscinaShowFinCaudal={piscinaShowFinCaudal}
                  setPiscinaShowFinCaudal={setPiscinaShowFinCaudal}
                  piscinaFinCaudal={piscinaFinCaudal}
                  setPiscinaFinCaudal={setPiscinaFinCaudal}
                  piscinaShowFinPectoral={piscinaShowFinPectoral}
                  setPiscinaShowFinPectoral={setPiscinaShowFinPectoral}
                  piscinaFinPectoral={piscinaFinPectoral}
                  setPiscinaFinPectoral={setPiscinaFinPectoral}
                  piscinaShowFinPelvic={piscinaShowFinPelvic}
                  setPiscinaShowFinPelvic={setPiscinaShowFinPelvic}
                  piscinaFinPelvic={piscinaFinPelvic}
                  setPiscinaFinPelvic={setPiscinaFinPelvic}
                  piscinaShowFinAdipose={piscinaShowFinAdipose}
                  setPiscinaShowFinAdipose={setPiscinaShowFinAdipose}
                  piscinaFinAdipose={piscinaFinAdipose}
                  setPiscinaFinAdipose={setPiscinaFinAdipose}
                  insectaSpecies={insectaSpecies}
                  setInsectaSpecies={setInsectaSpecies}
                  insectaTotalLength={insectaTotalLength}
                  setInsectaTotalLength={setInsectaTotalLength}
                  insectaHeadRatio={insectaHeadRatio}
                  setInsectaHeadRatio={setInsectaHeadRatio}
                  insectaThoraxRatio={insectaThoraxRatio}
                  setInsectaThoraxRatio={setInsectaThoraxRatio}
                  insectaAbdomenRatio={insectaAbdomenRatio}
                  setInsectaAbdomenRatio={setInsectaAbdomenRatio}
                  insectaBodyHalfWidth={insectaBodyHalfWidth}
                  setInsectaBodyHalfWidth={setInsectaBodyHalfWidth}
                  insectaBodyHalfHeight={insectaBodyHalfHeight}
                  setInsectaBodyHalfHeight={setInsectaBodyHalfHeight}
                  insectaAbdomenTaper={insectaAbdomenTaper}
                  setInsectaAbdomenTaper={setInsectaAbdomenTaper}
                  insectaHeadShape={insectaHeadShape}
                  setInsectaHeadShape={setInsectaHeadShape}
                  insectaBodyYawDeg={insectaBodyYawDeg}
                  setInsectaBodyYawDeg={setInsectaBodyYawDeg}
                  insectaBodyArch={insectaBodyArch}
                  setInsectaBodyArch={setInsectaBodyArch}
                  insectaAnchorU={insectaAnchorU}
                  setInsectaAnchorU={setInsectaAnchorU}
                  insectaAnchorV={insectaAnchorV}
                  setInsectaAnchorV={setInsectaAnchorV}
                  insectaAntennaLength={insectaAntennaLength}
                  setInsectaAntennaLength={setInsectaAntennaLength}
                  insectaAntennaSpread={insectaAntennaSpread}
                  setInsectaAntennaSpread={setInsectaAntennaSpread}
                  insectaAntennaPitch={insectaAntennaPitch}
                  setInsectaAntennaPitch={setInsectaAntennaPitch}
                  insectaAntennaRoot={insectaAntennaRoot}
                  setInsectaAntennaRoot={setInsectaAntennaRoot}
                  insectaMandibleLength={insectaMandibleLength}
                  setInsectaMandibleLength={setInsectaMandibleLength}
                  insectaMandibleSpread={insectaMandibleSpread}
                  setInsectaMandibleSpread={setInsectaMandibleSpread}
                  insectaMandibleForward={insectaMandibleForward}
                  setInsectaMandibleForward={setInsectaMandibleForward}
                  insectaWingShape={insectaWingShape}
                  setInsectaWingShape={setInsectaWingShape}
                  insectaShowWingFore={insectaShowWingFore}
                  setInsectaShowWingFore={setInsectaShowWingFore}
                  insectaWingForeLength={insectaWingForeLength}
                  setInsectaWingForeLength={setInsectaWingForeLength}
                  insectaWingForeWidth={insectaWingForeWidth}
                  setInsectaWingForeWidth={setInsectaWingForeWidth}
                  insectaWingForeSpread={insectaWingForeSpread}
                  setInsectaWingForeSpread={setInsectaWingForeSpread}
                  insectaWingForePitch={insectaWingForePitch}
                  setInsectaWingForePitch={setInsectaWingForePitch}
                  insectaWingForeOffset={insectaWingForeOffset}
                  setInsectaWingForeOffset={setInsectaWingForeOffset}
                  insectaWingForeForwardCant={insectaWingForeForwardCant}
                  setInsectaWingForeForwardCant={setInsectaWingForeForwardCant}
                  insectaShowWingHind={insectaShowWingHind}
                  setInsectaShowWingHind={setInsectaShowWingHind}
                  insectaWingHindLength={insectaWingHindLength}
                  setInsectaWingHindLength={setInsectaWingHindLength}
                  insectaWingHindWidth={insectaWingHindWidth}
                  setInsectaWingHindWidth={setInsectaWingHindWidth}
                  insectaWingHindSpread={insectaWingHindSpread}
                  setInsectaWingHindSpread={setInsectaWingHindSpread}
                  insectaWingHindPitch={insectaWingHindPitch}
                  setInsectaWingHindPitch={setInsectaWingHindPitch}
                  insectaWingHindOffset={insectaWingHindOffset}
                  setInsectaWingHindOffset={setInsectaWingHindOffset}
                  faunaStance={faunaStance}
                  setFaunaStance={setFaunaStance}
                  faunaArchetype={faunaArchetype}
                  setFaunaArchetype={setFaunaArchetype}
                  faunaBodyYawDeg={faunaBodyYawDeg}
                  setFaunaBodyYawDeg={setFaunaBodyYawDeg}
                  faunaBodyArch={faunaBodyArch}
                  setFaunaBodyArch={setFaunaBodyArch}
                  faunaSpineSegments={faunaSpineSegments}
                  setFaunaSpineSegments={setFaunaSpineSegments}
                  faunaBodyLength={faunaBodyLength}
                  setFaunaBodyLength={setFaunaBodyLength}
                  faunaBodyHalfWidth={faunaBodyHalfWidth}
                  setFaunaBodyHalfWidth={setFaunaBodyHalfWidth}
                  faunaBodyHalfHeight={faunaBodyHalfHeight}
                  setFaunaBodyHalfHeight={setFaunaBodyHalfHeight}
                  faunaNeckLength={faunaNeckLength}
                  setFaunaNeckLength={setFaunaNeckLength}
                  faunaNeckHalfWidth={faunaNeckHalfWidth}
                  setFaunaNeckHalfWidth={setFaunaNeckHalfWidth}
                  faunaNeckHalfHeight={faunaNeckHalfHeight}
                  setFaunaNeckHalfHeight={setFaunaNeckHalfHeight}
                  faunaHeadLength={faunaHeadLength}
                  setFaunaHeadLength={setFaunaHeadLength}
                  faunaHeadHalfWidth={faunaHeadHalfWidth}
                  setFaunaHeadHalfWidth={setFaunaHeadHalfWidth}
                  faunaHeadHalfHeight={faunaHeadHalfHeight}
                  setFaunaHeadHalfHeight={setFaunaHeadHalfHeight}
                  faunaTailLength={faunaTailLength}
                  setFaunaTailLength={setFaunaTailLength}
                  faunaShoulderOffsetForward={faunaShoulderOffsetForward}
                  setFaunaShoulderOffsetForward={setFaunaShoulderOffsetForward}
                  faunaHipOffsetForward={faunaHipOffsetForward}
                  setFaunaHipOffsetForward={setFaunaHipOffsetForward}
                  faunaFrontUpperLength={faunaFrontUpperLength}
                  setFaunaFrontUpperLength={setFaunaFrontUpperLength}
                  faunaFrontLowerLength={faunaFrontLowerLength}
                  setFaunaFrontLowerLength={setFaunaFrontLowerLength}
                  faunaHindUpperLength={faunaHindUpperLength}
                  setFaunaHindUpperLength={setFaunaHindUpperLength}
                  faunaHindLowerLength={faunaHindLowerLength}
                  setFaunaHindLowerLength={setFaunaHindLowerLength}
                  faunaAnchorU={faunaAnchorU}
                  setFaunaAnchorU={setFaunaAnchorU}
                  faunaAnchorV={faunaAnchorV}
                  setFaunaAnchorV={setFaunaAnchorV}
                  faunaAutoFootPlacement={faunaAutoFootPlacement}
                  setFaunaAutoFootPlacement={setFaunaAutoFootPlacement}
                />
              ) : null}
              {toolsPane === "squishy" && interactionMode === "squishy" ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Squishy session</div>
                  <div className="tool-options-shape-row" role="group" aria-label="Squishy mode">
                    <button
                      type="button"
                      className={
                        squishyMode === "add"
                          ? "tool-options-shape-btn is-active"
                          : "tool-options-shape-btn"
                      }
                      disabled={loading || workBusy}
                      onClick={() => setSquishyMode("add")}
                    >
                      Add
                    </button>
                    <button
                      type="button"
                      className={
                        squishyMode === "edit"
                          ? "tool-options-shape-btn is-active"
                          : "tool-options-shape-btn"
                      }
                      disabled={loading || workBusy}
                      onClick={() => setSquishyMode("edit")}
                    >
                      Pick
                    </button>
                    <button
                      type="button"
                      className={
                        squishyMode === "delete"
                          ? "tool-options-shape-btn is-active"
                          : "tool-options-shape-btn"
                      }
                      disabled={loading || workBusy}
                      onClick={() => setSquishyMode("delete")}
                    >
                      Delete
                    </button>
                  </div>
                  <p
                    style={{
                      fontSize: "0.85rem",
                      opacity: 0.85,
                      margin: "0.25rem 0",
                    }}
                  >
                    Metaballs: {squishyBallCount}. Click viewport to add/pick/delete; Commit
                    voxelizes the combined field.
                  </p>
                  <label className="tool-options-range-label">
                    <span>Blob radius (add)</span>
                    <input
                      type="range"
                      min={2}
                      max={10}
                      value={Math.min(10, Math.max(2, generatorSphereRadius))}
                      onChange={(ev) => setGeneratorSphereRadius(Number(ev.target.value))}
                      disabled={loading || workBusy}
                    />
                  </label>
                  <label
                    className="tool-options-range-label"
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={squishyHollow}
                      onChange={(ev) => setSquishyHollow(ev.target.checked)}
                      disabled={loading || workBusy}
                    />
                    <span>Hollow shell</span>
                  </label>
                  {squishyHollow ? (
                    <label className="tool-options-range-label">
                      <span>Shell thickness (voxels)</span>
                      <input
                        type="range"
                        min={1}
                        max={8}
                        step={1}
                        value={Math.min(8, Math.max(1, squishyWallThickness))}
                        onChange={(ev) => setSquishyWallThickness(Number(ev.target.value))}
                        disabled={loading || workBusy}
                      />
                    </label>
                  ) : null}
                  <label
                    className="tool-options-range-label"
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={squishySnapToSurface}
                      onChange={(ev) => setSquishySnapToSurface(ev.target.checked)}
                      disabled={loading || workBusy}
                    />
                    <span>Snap add to surface</span>
                  </label>
                  <div className="tool-options-shape-row" style={{ marginTop: "0.35rem" }}>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      disabled={loading || workBusy}
                      onClick={() => {
                        if (squishyPhase.active) {
                          squishyPhase.commit();
                        } else {
                          void invoke("squishy_session_commit", {
                            args: {
                              color: activeColorRef.current,
                              material: activeMaterialRef.current,
                            },
                          })
                            .then(() => invoke<{ balls: { id: number }[] }>("squishy_session_get"))
                            .then((s) => setSquishyBallCount(s.balls?.length ?? 0))
                            .catch(() => {});
                        }
                      }}
                    >
                      Commit to voxels
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      disabled={loading || workBusy}
                      onClick={() => {
                        if (squishyPhase.active) {
                          squishyPhase.cancel();
                        } else {
                          void invoke("squishy_session_clear")
                            .then(() => setSquishyBallCount(0))
                            .catch(() => {});
                        }
                      }}
                    >
                      Clear session
                    </button>
                  </div>
                </div>
              ) : toolsPane === "squishy" ? (
                <div className="tool-options-section">
                  <p className="tool-options-hint">Select Squishy mode in the sidebar.</p>
                </div>
              ) : null}
              {toolsPane === "mood" ? (
                <div className="tool-options-section">
                  {/* ── Grain ────────────────────────────── */}
                  <div className="tool-options-heading">Grain</div>
                  <label
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={mood.grainEnabled}
                      onChange={(ev) =>
                        setMood((p) => moodWith(p, { grainEnabled: ev.target.checked }))
                      }
                      disabled={loading || workBusy}
                    />
                    <span>Enable grain</span>
                  </label>
                  {mood.grainEnabled && (
                    <>
                      <label className="tool-options-range-label">
                        <span>Strength</span>
                        <input
                          type="range"
                          min={0}
                          max={0.5}
                          step={0.01}
                          value={mood.grainStrength}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                grainStrength: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.5rem",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={mood.grainAnimated}
                          onChange={(ev) =>
                            setMood((p) => moodWith(p, { grainAnimated: ev.target.checked }))
                          }
                          disabled={loading || workBusy}
                        />
                        <span>Animated</span>
                      </label>
                      {mood.grainAnimated && (
                        <label className="tool-options-range-label">
                          <span>Speed</span>
                          <input
                            type="range"
                            min={0}
                            max={4}
                            step={0.1}
                            value={mood.grainSpeed}
                            onChange={(ev) =>
                              setMood((p) =>
                                moodWith(p, {
                                  grainSpeed: Number(ev.target.value),
                                }),
                              )
                            }
                            disabled={loading || workBusy}
                          />
                        </label>
                      )}
                      <label
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.5rem",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={mood.grainColorful}
                          onChange={(ev) =>
                            setMood((p) => moodWith(p, { grainColorful: ev.target.checked }))
                          }
                          disabled={loading || workBusy}
                        />
                        <span>Colorful</span>
                      </label>
                    </>
                  )}

                  {/* ── Bloom ────────────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>
                    Bloom
                  </div>
                  <label className="tool-options-range-label">
                    <span>Strength</span>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.02}
                      value={mood.bloomStrength}
                      onChange={(ev) =>
                        setMood((p) =>
                          moodWith(p, {
                            bloomStrength: Number(ev.target.value),
                          }),
                        )
                      }
                      disabled={loading || workBusy}
                    />
                  </label>

                  {/* ── Vignette ─────────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>
                    Vignette
                  </div>
                  <label className="tool-options-range-label">
                    <span>Strength</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={mood.vignette}
                      onChange={(ev) =>
                        setMood((p) => moodWith(p, { vignette: Number(ev.target.value) }))
                      }
                      disabled={loading || workBusy}
                    />
                  </label>

                  {/* ── Atmosphere ────────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>
                    Atmosphere
                  </div>
                  <label
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={mood.atmEnabled}
                      onChange={(ev) =>
                        setMood((p) => moodWith(p, { atmEnabled: ev.target.checked }))
                      }
                      disabled={loading || workBusy}
                    />
                    <span>Enable atmosphere</span>
                  </label>
                  {mood.atmEnabled && (
                    <>
                      <label className="tool-options-range-label">
                        <span>Color</span>
                        <input
                          type="color"
                          value={mood.atmColor}
                          onChange={(ev) =>
                            setMood((p) => moodWith(p, { atmColor: ev.target.value }))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Thickness</span>
                        <input
                          type="range"
                          min={1}
                          max={200}
                          step={1}
                          value={mood.atmThickness}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                atmThickness: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Density</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={mood.atmDensity}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                atmDensity: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <div
                        style={{
                          display: "flex",
                          gap: "0.75rem",
                          marginTop: "0.25rem",
                        }}
                      >
                        <label
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "0.3rem",
                          }}
                        >
                          <input
                            type="radio"
                            name="atm-spatial"
                            checked={mood.atmAerial}
                            onChange={() => setMood((p) => moodWith(p, { atmAerial: true }))}
                            disabled={loading || workBusy}
                          />
                          <span>Aerial</span>
                        </label>
                        <label
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "0.3rem",
                          }}
                        >
                          <input
                            type="radio"
                            name="atm-spatial"
                            checked={!mood.atmAerial}
                            onChange={() => setMood((p) => moodWith(p, { atmAerial: false }))}
                            disabled={loading || workBusy}
                          />
                          <span>Plane</span>
                        </label>
                      </div>
                      {!mood.atmAerial && (
                        <div
                          style={{
                            display: "flex",
                            gap: "0.75rem",
                            marginTop: "0.25rem",
                          }}
                        >
                          <label
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "0.3rem",
                            }}
                          >
                            <input
                              type="radio"
                              name="atm-mode"
                              checked={!mood.atmPositiveSide}
                              onChange={() =>
                                setMood((p) => moodWith(p, { atmPositiveSide: false }))
                              }
                              disabled={loading || workBusy}
                            />
                            <span>Layer (slab)</span>
                          </label>
                          <label
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "0.3rem",
                            }}
                          >
                            <input
                              type="radio"
                              name="atm-mode"
                              checked={mood.atmPositiveSide}
                              onChange={() =>
                                setMood((p) => moodWith(p, { atmPositiveSide: true }))
                              }
                              disabled={loading || workBusy}
                            />
                            <span>Above face</span>
                          </label>
                        </div>
                      )}
                      <label className="tool-options-range-label">
                        <span>Height bias</span>
                        <input
                          type="range"
                          min={-200}
                          max={200}
                          step={1}
                          value={mood.atmHeightBias}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                atmHeightBias: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Height falloff</span>
                        <input
                          type="range"
                          min={10}
                          max={400}
                          step={1}
                          value={mood.atmHeightFalloff}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                atmHeightFalloff: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.5rem",
                          marginTop: "0.25rem",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={mood.atmDriftEnabled}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                atmDriftEnabled: ev.target.checked,
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                        <span>Drift</span>
                      </label>
                      {mood.atmDriftEnabled && (
                        <>
                          <label className="tool-options-range-label">
                            <span>Amount</span>
                            <input
                              type="range"
                              min={0}
                              max={1}
                              step={0.02}
                              value={mood.atmDriftAmount}
                              onChange={(ev) =>
                                setMood((p) =>
                                  moodWith(p, {
                                    atmDriftAmount: Number(ev.target.value),
                                  }),
                                )
                              }
                              disabled={loading || workBusy}
                            />
                          </label>
                          <label className="tool-options-range-label">
                            <span>Scale</span>
                            <input
                              type="range"
                              min={0.001}
                              max={0.1}
                              step={0.001}
                              value={mood.atmDriftScale}
                              onChange={(ev) =>
                                setMood((p) =>
                                  moodWith(p, {
                                    atmDriftScale: Number(ev.target.value),
                                  }),
                                )
                              }
                              disabled={loading || workBusy}
                            />
                          </label>
                          <label className="tool-options-range-label">
                            <span>Speed</span>
                            <input
                              type="range"
                              min={0}
                              max={2}
                              step={0.02}
                              value={mood.atmDriftSpeed}
                              onChange={(ev) =>
                                setMood((p) =>
                                  moodWith(p, {
                                    atmDriftSpeed: Number(ev.target.value),
                                  }),
                                )
                              }
                              disabled={loading || workBusy}
                            />
                          </label>
                        </>
                      )}
                    </>
                  )}

                  {/* ── Distance tint ────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>
                    Distance tint
                  </div>
                  <label
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={mood.dtEnabled}
                      onChange={(ev) =>
                        setMood((p) => moodWith(p, { dtEnabled: ev.target.checked }))
                      }
                      disabled={loading || workBusy}
                    />
                    <span>Enable distance tint</span>
                  </label>
                  {mood.dtEnabled && (
                    <>
                      <label className="tool-options-range-label">
                        <span>Near color</span>
                        <input
                          type="color"
                          value={mood.dtNearColor}
                          onChange={(ev) =>
                            setMood((p) => moodWith(p, { dtNearColor: ev.target.value }))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Mid color</span>
                        <input
                          type="color"
                          value={mood.dtMidColor}
                          onChange={(ev) =>
                            setMood((p) => moodWith(p, { dtMidColor: ev.target.value }))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Far color</span>
                        <input
                          type="color"
                          value={mood.dtFarColor}
                          onChange={(ev) =>
                            setMood((p) => moodWith(p, { dtFarColor: ev.target.value }))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Near distance</span>
                        <input
                          type="range"
                          min={1}
                          max={200}
                          step={1}
                          value={mood.dtNearDist}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                dtNearDist: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Far distance</span>
                        <input
                          type="range"
                          min={1}
                          max={400}
                          step={1}
                          value={mood.dtFarDist}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                dtFarDist: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Strength</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={mood.dtStrength}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                dtStrength: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  )}

                  {/* ── Sun shafts ────────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>
                    Sun shafts
                  </div>
                  <label
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={mood.ssEnabled}
                      onChange={(ev) =>
                        setMood((p) => moodWith(p, { ssEnabled: ev.target.checked }))
                      }
                      disabled={loading || workBusy}
                    />
                    <span>Enable sun shafts</span>
                  </label>
                  {mood.ssEnabled && (
                    <>
                      <label className="tool-options-range-label">
                        <span>Strength</span>
                        <input
                          type="range"
                          min={0}
                          max={10}
                          step={0.1}
                          value={mood.ssStrength}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                ssStrength: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Decay</span>
                        <input
                          type="range"
                          min={0.5}
                          max={0.99}
                          step={0.01}
                          value={mood.ssDecay}
                          onChange={(ev) =>
                            setMood((p) => moodWith(p, { ssDecay: Number(ev.target.value) }))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Density</span>
                        <input
                          type="range"
                          min={0.1}
                          max={1.5}
                          step={0.02}
                          value={mood.ssDensity}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                ssDensity: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Weight</span>
                        <input
                          type="range"
                          min={0}
                          max={1.5}
                          step={0.02}
                          value={mood.ssWeight}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                ssWeight: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Samples</span>
                        <input
                          type="range"
                          min={20}
                          max={56}
                          step={1}
                          value={mood.ssSamples}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                ssSamples: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  )}

                  {/* ── Screen-space reflections ──────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>
                    Reflections
                  </div>
                  <label
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={mood.ssrEnabled}
                      onChange={(ev) =>
                        setMood((p) => moodWith(p, { ssrEnabled: ev.target.checked }))
                      }
                      disabled={loading || workBusy}
                    />
                    <span>Screen-space reflections</span>
                  </label>
                  {mood.ssrEnabled && (
                    <>
                      <label className="tool-options-range-label">
                        <span>Strength</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={mood.ssrStrength}
                          onChange={(ev) =>
                            setMood((p) =>
                              moodWith(p, {
                                ssrStrength: Number(ev.target.value),
                              }),
                            )
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  )}
                </div>
              ) : null}
              {(interactionMode === "stamp" || interactionMode === "punch") &&
              (toolsPane === "draw" || toolsPane === "select") ? (
                <div className="tool-options-section" aria-label="Stamp orientation">
                  <div className="tool-options-heading">Orientation</div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "auto 1fr 1fr 1fr",
                      gap: "0.25rem",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ fontSize: "0.72rem", color: "var(--app-text-faint)" }}>Rot</span>
                    {(
                      [
                        ["X", stampRotX, setStampRotX],
                        ["Y", stampRotY, setStampRotY],
                        ["Z", stampRotZ, setStampRotZ],
                      ] as const
                    ).map(([axis, val, set]) => (
                      <input
                        key={axis}
                        type="number"
                        step={1}
                        value={val}
                        title={`${axis} rotation (degrees)`}
                        style={{ width: "100%", minWidth: 0 }}
                        onInput={(e) => set(Number((e.target as HTMLInputElement).value))}
                      />
                    ))}
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "auto 1fr 1fr",
                      gap: "0.2rem",
                      alignItems: "center",
                    }}
                  >
                    {(
                      [
                        ["X", setStampRotX],
                        ["Y", setStampRotY],
                        ["Z", setStampRotZ],
                      ] as const
                    ).map(([axis, set]) => (
                      <>
                        <span
                          key={`${axis}-label`}
                          style={{ fontSize: "0.72rem", color: "var(--app-text-faint)" }}
                        >
                          {axis}
                        </span>
                        <button
                          key={`${axis}-ccw`}
                          type="button"
                          className="tool-options-shape-btn"
                          title={`CCW ${axis} (−15°)`}
                          onClick={() => set((v) => v - 15)}
                        >
                          CCW
                        </button>
                        <button
                          key={`${axis}-cw`}
                          type="button"
                          className="tool-options-shape-btn"
                          title={`CW ${axis} (+15°)`}
                          onClick={() => set((v) => v + 15)}
                        >
                          CW
                        </button>
                      </>
                    ))}
                  </div>
                  <div style={{ marginTop: "0.4rem" }}>
                    <span style={{ fontSize: "0.72rem", color: "var(--app-text-faint)" }}>
                      Origin
                    </span>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(3, 1fr)",
                        gap: "0.15rem",
                        marginTop: "0.2rem",
                      }}
                    >
                      {([0, 1, 2] as const).flatMap((oz) =>
                        ([0, 1, 2] as const).map((ox) => {
                          const labels = ["left", "center", "right"];
                          const labelsZ = ["front", "center", "back"];
                          return (
                            <button
                              key={`${ox}-${oz}`}
                              type="button"
                              className={`tool-options-shape-btn${stampOriginX === ox && stampOriginZ === oz ? " is-active" : ""}`}
                              style={{
                                padding: "0.3rem 0",
                                minWidth: 0,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                              }}
                              title={`Origin: ${labels[ox]} / ${labelsZ[oz]}`}
                              onClick={() => {
                                setStampOriginX(ox);
                                setStampOriginZ(oz);
                              }}
                            >
                              <span
                                style={{
                                  width: "0.35rem",
                                  height: "0.35rem",
                                  borderRadius: "50%",
                                  background: "currentColor",
                                  display: "block",
                                  opacity: stampOriginX === ox && stampOriginZ === oz ? 1 : 0.35,
                                }}
                              />
                            </button>
                          );
                        }),
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="tool-options-shape-btn"
                    style={{ width: "100%" }}
                    onClick={() => {
                      setStampRotX(0);
                      setStampRotY(0);
                      setStampRotZ(0);
                    }}
                  >
                    Reset
                  </button>
                </div>
              ) : null}
            </div>
          ) : null}
          {showEmptyOpenFile ? (
            <div className="viewport-empty-open" role="region" aria-label="No file open">
              <div className="viewport-empty-open-stack">
                {lastSessionReady &&
                lastSessionInfo?.lastDocumentPath &&
                (lastSessionInfo.documentExists || lastSessionInfo.autosaveExists) ? (
                  <div
                    className="viewport-empty-last"
                    role="group"
                    aria-label="Continue last project"
                  >
                    <div className="viewport-empty-last-title">Continue where you left off</div>
                    {lastSessionInfo.documentBasename ? (
                      <div
                        className="viewport-empty-last-filename"
                        title={lastSessionInfo.lastDocumentPath ?? undefined}
                      >
                        {lastSessionInfo.documentBasename}
                      </div>
                    ) : null}
                    {lastProjectBlurb ? (
                      <p id="viewport-empty-last-desc" className="viewport-empty-last-blurb">
                        {lastProjectBlurb}
                      </p>
                    ) : null}
                    <div className="viewport-empty-last-actions">
                      <button
                        type="button"
                        className="viewport-empty-open-btn"
                        onClick={() => setNewProjectOpen(true)}
                        disabled={loading || workBusy}
                      >
                        Start new project
                      </button>
                      <button
                        type="button"
                        className="viewport-empty-open-btn is-secondary"
                        onClick={reopenLastProject}
                        disabled={loading || workBusy}
                        aria-describedby={lastProjectBlurb ? "viewport-empty-last-desc" : undefined}
                      >
                        Reopen last project
                      </button>
                    </div>
                  </div>
                ) : lastSessionReady ? (
                  <button
                    type="button"
                    className="viewport-empty-open-btn"
                    onClick={() => setNewProjectOpen(true)}
                    disabled={loading || workBusy}
                  >
                    New Project
                  </button>
                ) : null}
                <button
                  type="button"
                  className="viewport-empty-open-btn is-secondary"
                  onClick={() => void invoke("open_voxelle_dialog").catch(() => {})}
                >
                  Open file…
                </button>
                <div className="viewport-empty-session-row">
                  <button
                    type="button"
                    className="viewport-empty-open-btn is-secondary"
                    onClick={() => setJoinModalOpen(true)}
                    disabled={collabActive}
                    title={collabActive ? "Leave your session first" : "Paste a host link"}
                  >
                    Join Session
                  </button>
                  <button
                    type="button"
                    className="viewport-empty-open-btn"
                    onClick={collabActive ? leaveSession : startHost}
                    title={
                      collabActive
                        ? hostWsUrl
                          ? "End the session for everyone"
                          : "Leave session"
                        : undefined
                    }
                  >
                    {hostWsUrl ? "Stop hosting" : collabGuest ? "Leave" : "Start Session"}
                  </button>
                </div>
              </div>
            </div>
          ) : null}
          {/* Debug: Logo light controls (toggle via Debug menu) */}
          {showStartScreen && startScreenLogoLoaded && logoLightControlsVisible ? (
            <div
              style={{
                position: "absolute",
                bottom: 12,
                right: 12,
                zIndex: 20,
                background: "rgba(0,0,0,0.75)",
                color: "#fff",
                padding: "10px 14px",
                borderRadius: 8,
                fontSize: 12,
                fontFamily: "system-ui, sans-serif",
                display: "flex",
                flexDirection: "column",
                gap: 6,
                minWidth: 220,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 2 }}>Camera</div>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 60 }}>Angle</span>
                <input
                  type="range"
                  min={0}
                  max={360}
                  step={1}
                  value={logoCamAzimuth}
                  style={{ flex: 1 }}
                  onChange={(e) => {
                    const az = Number(e.target.value);
                    setLogoCamAzimuth(az);
                    void invoke("logo_set_camera_angle", {
                      azimuth: az,
                      elevation: logoCamElevation,
                    });
                  }}
                />
                <span style={{ width: 36, textAlign: "right", fontFamily: "monospace" }}>
                  {Math.round(logoCamAzimuth)}°
                </span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 60 }}>Elevation</span>
                <input
                  type="range"
                  min={-85}
                  max={85}
                  step={1}
                  value={logoCamElevation}
                  style={{ flex: 1 }}
                  onChange={(e) => {
                    const el = Number(e.target.value);
                    setLogoCamElevation(el);
                    void invoke("logo_set_camera_angle", {
                      azimuth: logoCamAzimuth,
                      elevation: el,
                    });
                  }}
                />
                <span style={{ width: 36, textAlign: "right", fontFamily: "monospace" }}>
                  {Math.round(logoCamElevation)}°
                </span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 60 }}>Zoom</span>
                <input
                  type="range"
                  min={1.0}
                  max={8.0}
                  step={0.1}
                  value={logoCamDist}
                  style={{ flex: 1 }}
                  onChange={(e) => {
                    const d = Number(e.target.value);
                    setLogoCamDist(d);
                    void invoke("logo_set_camera_dist", { dist: d });
                  }}
                />
                <span style={{ width: 36, textAlign: "right", fontFamily: "monospace" }}>
                  {logoCamDist.toFixed(1)}
                </span>
              </label>
              <div style={{ borderTop: "1px solid rgba(255,255,255,0.15)", margin: "4px 0" }} />
              <div style={{ fontWeight: 600, marginBottom: 2 }}>Light</div>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 60 }}>Angle</span>
                <input
                  type="range"
                  min={0}
                  max={360}
                  step={1}
                  value={logoLightAzimuth}
                  style={{ flex: 1 }}
                  onChange={(e) => {
                    const az = Number(e.target.value);
                    setLogoLightAzimuth(az);
                    void invoke("logo_set_light_dir", {
                      azimuth: az,
                      elevation: logoLightElevation,
                    });
                  }}
                />
                <span style={{ width: 36, textAlign: "right", fontFamily: "monospace" }}>
                  {Math.round(logoLightAzimuth)}°
                </span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 60 }}>Elevation</span>
                <input
                  type="range"
                  min={5}
                  max={90}
                  step={1}
                  value={logoLightElevation}
                  style={{ flex: 1 }}
                  onChange={(e) => {
                    const el = Number(e.target.value);
                    setLogoLightElevation(el);
                    void invoke("logo_set_light_dir", { azimuth: logoLightAzimuth, elevation: el });
                  }}
                />
                <span style={{ width: 36, textAlign: "right", fontFamily: "monospace" }}>
                  {Math.round(logoLightElevation)}°
                </span>
              </label>
            </div>
          ) : null}
          {loadError ? (
            <div className="viewport-error" role="alert">
              <span className="viewport-notice-text" title={loadError}>
                {loadError}
              </span>
              <button
                type="button"
                className="viewport-notice-dismiss"
                aria-label="Dismiss error"
                onClick={() => setLoadError(null)}
              >
                Dismiss
              </button>
            </div>
          ) : null}
          {collabBanner ? (
            <div
              className={
                collabBanner.tone === "alert" ? "viewport-notice is-alert" : "viewport-notice"
              }
              role={collabBanner.tone === "alert" ? "alert" : "status"}
            >
              <span className="viewport-notice-text">{collabBanner.text}</span>
              <button
                type="button"
                className="viewport-notice-dismiss"
                onClick={() => setCollabBanner(null)}
              >
                Dismiss
              </button>
            </div>
          ) : null}
          {collabActive && chatToasts.length > 0 ? (
            <div className="chat-toast-stack" aria-live="polite" aria-label="New chat messages">
              {chatToasts.map((t) => (
                <div
                  key={t.id}
                  className="chat-toast"
                  role="status"
                  onClick={() => setChatPanelOpen(true)}
                >
                  <span className="chat-toast-text">{t.text}</span>
                  <button
                    type="button"
                    className="chat-toast-dismiss"
                    aria-label="Dismiss notification"
                    onClick={(e) => {
                      e.stopPropagation();
                      setChatToasts((prev) => prev.filter((x) => x.id !== t.id));
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          {chatPanelOpen ? (
            <div className="chat-float-panel" role="dialog" aria-label="Collaboration chat">
              <div className="chat-float-header">
                <h3 className="chat-float-title">Chat</h3>
                <button
                  type="button"
                  className="chat-float-close"
                  onClick={() => setChatPanelOpen(false)}
                  aria-label="Close chat"
                >
                  ×
                </button>
              </div>
              <div className="collab-chat-log chat-float-log" role="log">
                {chatLines.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
              <div className="collab-row chat-float-input-row">
                <input
                  className="collab-grow"
                  type="text"
                  value={chatInput}
                  placeholder={collabActive ? "Message…" : "Join or host to chat"}
                  disabled={!collabActive}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => collabActive && e.key === "Enter" && sendChat()}
                />
                <button type="button" onClick={sendChat} disabled={!collabActive}>
                  Send
                </button>
              </div>
            </div>
          ) : null}
        </div>
        {showEditorChrome ? (
          <aside
            className={
              rightSidebarExpanded
                ? "app-sidebar app-sidebar-right is-expanded"
                : "app-sidebar app-sidebar-right is-collapsed"
            }
            aria-label="Inspector"
          >
            <div className="sidebar-header sidebar-header-right">
              <button
                type="button"
                className="sidebar-expand-toggle sidebar-expand-toggle-right"
                onClick={() => setRightSidebarExpanded((v) => !v)}
                aria-expanded={rightSidebarExpanded}
                title={rightSidebarExpanded ? "Collapse inspector" : "Expand inspector"}
              >
                {rightSidebarExpanded ? (
                  <>
                    <span className="sidebar-expand-toggle-label">Inspector</span>
                    <span className="sidebar-expand-toggle-icon" aria-hidden>
                      »
                    </span>
                  </>
                ) : (
                  <span className="sidebar-expand-toggle-icon" aria-hidden>
                    «
                  </span>
                )}
              </button>
            </div>
            {rightSidebarExpanded ? (
              <div className="sidebar-scroll">
                <div
                  className="sidebar-expanded-slot sidebar-expanded-slot-right"
                  aria-label="Inspector content"
                >
                  <div className="inspector-objects">
                    <h4 className="inspector-heading">Objects</h4>
                    {sceneObjectsErr ? <p className="inspector-hint">{sceneObjectsErr}</p> : null}
                    <ul className="inspector-object-list">
                      {sceneObjects
                        .slice()
                        .sort((a, b) => a.sortOrder - b.sortOrder || a.id - b.id)
                        .map((o) => (
                          <li key={o.id} className="inspector-object-row">
                            <label className="inspector-active">
                              <input
                                type="radio"
                                name="activeObject"
                                checked={activeObjectId === o.id}
                                onChange={() => {
                                  void invoke("set_active_object", {
                                    id: o.id,
                                  }).then(() => {
                                    setActiveObjectId(o.id);
                                    refreshSceneObjects();
                                  });
                                }}
                              />
                              <span className="inspector-object-name">{o.name}</span>
                            </label>
                            <label className="inspector-visible">
                              <input
                                type="checkbox"
                                checked={o.visible}
                                onChange={(e) => {
                                  void invoke("set_object_visible", {
                                    id: o.id,
                                    visible: e.target.checked,
                                  }).then(() => refreshSceneObjects());
                                }}
                              />
                              Visible
                            </label>
                          </li>
                        ))}
                    </ul>
                    <button
                      type="button"
                      className="inspector-new-object"
                      onClick={() => {
                        void invoke<number>("create_scene_object", {
                          name: "",
                        }).then(() => refreshSceneObjects());
                      }}
                    >
                      New object
                    </button>
                  </div>
                  {collabActive ? (
                    <div className="inspector-collaboration">
                      <h4 className="inspector-heading">Session</h4>
                      {hostWsUrl ? (
                        <>
                          <button
                            type="button"
                            className="inspector-copy-invite-btn"
                            onClick={copyHostingJoinAddress}
                            title={hostingCopied ? "Copied" : "Copy invite link"}
                          >
                            <span className="inspector-copy-invite-label">
                              {hostingCopied ? "Copied!" : "Copy invite link"}
                            </span>
                            <code className="inspector-copy-invite-url">
                              {hostWanUrl ?? hostWsUrl}
                            </code>
                          </button>
                          {hostWanUrl ? (
                            <p className="collab-hint inspector-collab-hint">
                              Nearby: <code>{hostWsUrl}</code>
                            </p>
                          ) : null}
                          {prefsEnableUpnp && natPending ? (
                            <p
                              className="collab-hint collab-hint-muted inspector-collab-hint"
                              role="status"
                            >
                              Checking your router…
                            </p>
                          ) : null}
                          {natError ? (
                            <p
                              className="collab-hint collab-hint-warn inspector-collab-hint"
                              role="alert"
                            >
                              {natError} You can forward port {hostPort} in your router settings.
                              Some networks won&apos;t allow guests over the internet.
                            </p>
                          ) : null}
                        </>
                      ) : null}
                      <h4 className="inspector-heading inspector-roster-heading">Roster</h4>
                      <ul className="collab-roster inspector-collab-roster">
                        {roster.map((r) => (
                          <li key={r.peerId}>
                            <button
                              type="button"
                              className="collab-roster-name"
                              onClick={() => onRosterSnapCamera(r.peerId)}
                              title="Jump to their view"
                            >
                              <span
                                className="collab-swatch"
                                style={{
                                  background: `#${(r.colorRgb & 0xffffff)
                                    .toString(16)
                                    .padStart(6, "0")}`,
                                }}
                              />
                              {r.displayName}
                              {r.isLeader ? " (leader)" : ""}
                            </button>
                            {!r.isLeader && amLeader ? (
                              <>
                                <label className="collab-can-edit">
                                  <input
                                    type="checkbox"
                                    checked={r.canEdit}
                                    onChange={(e) => setCanEdit(r.peerId, e.target.checked)}
                                  />
                                  Edit
                                </label>
                                <button
                                  type="button"
                                  className="collab-kick"
                                  title="Remove guest"
                                  onClick={() =>
                                    void invoke("collab_kick_peer", {
                                      targetPeer: r.peerId,
                                    })
                                  }
                                >
                                  Kick
                                </button>
                              </>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
          </aside>
        ) : null}
      </div>
      <StatusBar
        showStartScreen={showStartScreen}
        statusBarMessage={statusBarMessage}
        pathLabel={pathLabel}
        collabActive={collabActive}
        hostWsUrl={hostWsUrl}
        hostingCopied={hostingCopied}
        copyHostingJoinAddress={copyHostingJoinAddress}
        roster={roster}
        setLeaveConfirmOpen={setLeaveConfirmOpen}
        startHost={startHost}
        showFpsCounter={showFpsCounter}
        showEditorChrome={showEditorChrome}
        fpsDisplayed={fpsDisplayed}
        showPingLatency={showPingLatency}
        pingMs={pingMs}
      />
      {leaveConfirmOpen && (
        <div className="modal-overlay" onClick={() => setLeaveConfirmOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{hostWsUrl ? "End session?" : "Leave session?"}</h3>
            <p style={{ margin: "0 0 0.75rem", fontSize: "0.875rem" }}>
              {hostWsUrl
                ? "This will end the session for everyone."
                : "You will leave the current session."}
            </p>
            <div className="modal-buttons">
              <button type="button" onClick={() => setLeaveConfirmOpen(false)}>
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setLeaveConfirmOpen(false);
                  leaveSession();
                }}
              >
                {hostWsUrl ? "End session" : "Leave"}
              </button>
            </div>
          </div>
        </div>
      )}
      <JoinSessionModal
        open={joinModalOpen}
        onClose={() => setJoinModalOpen(false)}
        joinUrl={joinUrl}
        onJoinUrlChange={setJoinUrl}
        onJoin={joinSession}
        collabActive={collabActive}
        connecting={collabJoinPending}
      />
      <CollabJoinProgressModal
        open={collabJoinPending}
        loading={loading}
        loadProgress={loadProgress}
        loadPhase={loadPhase}
        pathLabel={pathLabel}
        onCancel={cancelJoin}
      />
      <StampBookModal
        open={stampBookOpen}
        onClose={() => setStampBookOpen(false)}
        selectionCount={selectionCount}
        onUseStamp={(entries: StampBookEntryTuple[]) => {
          void invoke("stamp_book_load_entries", {
            entries: entries.map(([dx, dy, dz, color, mat]) => ({
              dx,
              dy,
              dz,
              color,
              material: mat ?? "plastic",
            })),
          })
            .then(() => {
              void invoke("selection_clear").catch(() => {});
              setStampBookPatternActive(true);
              setInteractionMode("stamp");
            })
            .catch(() => {});
        }}
      />
      {pendingFillConfirm && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onKeyDown={(e) => {
            if (e.key === "Escape") pendingFillConfirm.resolve(false);
          }}
        >
          <div className="modal">
            <h3>Large fill</h3>
            <p style={{ fontSize: "0.85rem", margin: "0 0 0.75rem" }}>
              This fill covers a large area and may take a while. Continue?
            </p>
            <div className="modal-buttons">
              <button onClick={() => pendingFillConfirm.resolve(true)} autoFocus>
                Fill
              </button>
              <button onClick={() => pendingFillConfirm.resolve(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
      <PreferencesModal
        open={preferencesOpen}
        onClose={() => setPreferencesOpen(false)}
        onFpsCounterChange={setShowFpsCounter}
        onPingLatencyChange={setShowPingLatency}
        onEnableUpnpChange={setPrefsEnableUpnp}
        onCollabDisplayNameChange={setDisplayName}
        onCollabAccentColorChange={setAccentColor}
        onCollabHostPortChange={setHostPort}
        collabHosting={hostWsUrl != null}
      />
      {rotateDialogOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onClick={(e) => e.target === e.currentTarget && setRotateDialogOpen(false)}
          onKeyDown={(e) => e.key === "Escape" && setRotateDialogOpen(false)}
        >
          <div className="modal">
            <h3>Rotate selection</h3>
            <label className="modal-field">
              Axis
              <select
                value={rotateDialogAxis}
                onChange={(e) => setRotateDialogAxis(Number(e.target.value) as 0 | 1 | 2)}
              >
                <option value={0}>X</option>
                <option value={1}>Y</option>
                <option value={2}>Z</option>
              </select>
            </label>
            <label className="modal-field">
              Degrees
              <select
                value={rotateDialogDegrees}
                onChange={(e) => setRotateDialogDegrees(Number(e.target.value))}
              >
                <option value={90}>90°</option>
                <option value={180}>180°</option>
                <option value={270}>270°</option>
              </select>
            </label>
            <div className="modal-buttons">
              <button
                type="button"
                onClick={() => {
                  const quarters = rotateDialogDegrees / 90;
                  void invoke("selection_rotate", {
                    axis: rotateDialogAxis,
                    quarters,
                  }).catch(() => {});
                  setRotateDialogOpen(false);
                }}
              >
                Rotate
              </button>
              <button type="button" onClick={() => setRotateDialogOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {scaleDialogOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onClick={(e) => e.target === e.currentTarget && setScaleDialogOpen(false)}
          onKeyDown={(e) => e.key === "Escape" && setScaleDialogOpen(false)}
        >
          <div className="modal">
            <h3>Scale selection</h3>
            <label className="modal-field">
              Factor
              <input
                type="number"
                min={0.1}
                max={8}
                step={0.25}
                value={scaleDialogFactor}
                onChange={(e) =>
                  setScaleDialogFactor(Math.max(0.1, Math.min(8, Number(e.target.value))))
                }
              />
            </label>
            <div className="modal-buttons">
              <button
                type="button"
                onClick={() => {
                  void invoke("selection_scale", {
                    factor: scaleDialogFactor,
                  }).catch(() => {});
                  setScaleDialogOpen(false);
                }}
              >
                Scale
              </button>
              <button type="button" onClick={() => setScaleDialogOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {newProjectOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onClick={(e) => e.target === e.currentTarget && setNewProjectOpen(false)}
          onKeyDown={(e) => e.key === "Escape" && setNewProjectOpen(false)}
        >
          <div className="modal">
            <h3>New project</h3>
            <label className="modal-field">
              Grid size (1–{MAX_GRID_SIZE.toLocaleString()})
              <input
                type="number"
                min={1}
                max={MAX_GRID_SIZE}
                step={1}
                value={newGridSize}
                onChange={(e) => setNewGridSize(Number(e.target.value))}
              />
            </label>
            <label className="modal-field">
              Starting shape
              <select
                value={newGridShape}
                onChange={(e) => setNewGridShape(e.target.value as StartShape)}
              >
                <option value="cube">Cube</option>
                <option value="orb">Orb</option>
                <option value="cylinder">Cylinder</option>
                <option value="hollowCube">Hollow cube</option>
                <option value="plane">Plane</option>
                <option value="circle">Circle</option>
                <option value="empty">Empty</option>
              </select>
            </label>
            <div className="modal-buttons">
              <button type="button" onClick={createNewProject}>
                Create
              </button>
              <button type="button" onClick={() => setNewProjectOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {/* ── Start-screen mascot overlays ─────────────────────────────────────
           Invisible click-detection div positioned over the GPU-rendered seagull. */}
      {showStartScreen && mascotsLoaded && (
        <MascotView
          id={0}
          rect={mascotRect}
          visible={showStartScreen}
          onClick={handleMascotClick}
        />
      )}

      {/* ── Speech bubble click-capture overlays ─────────────────────────────
           GPU renders the actual bubble shapes; these divs only capture clicks. */}
      <SpeechBubbleOverlay bubbles={speechBubbles} />

      {/* ── Radial emoji-ping menu (hold Z) ──────────────────────────────── */}
      <RadialPingMenu
        x={radialMenu.x}
        y={radialMenu.y}
        visible={radialMenu.visible}
        onSelect={onRadialSelect}
      />

      {/* ── Off-screen ping arrow indicator ────────────────────────────── */}
      {(() => {
        const p = pingHudRef.current;
        // pingHudTick is read to subscribe to re-renders
        void pingHudTick;
        const isActive = !!p && Date.now() < p.until;
        return (
          <PingArrowIndicator
            wx={p?.wx ?? 0}
            wy={p?.wy ?? 0}
            wz={p?.wz ?? 0}
            active={isActive}
            emoji={p?.emoji}
          />
        );
      })()}
    </div>
  );
}

export default App;
