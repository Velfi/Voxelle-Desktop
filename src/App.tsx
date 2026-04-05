import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useFlyMode } from "./hooks/useFlyMode";
import { useViewportPointer } from "./hooks/useViewportPointer";
import { useWalkMode } from "./hooks/useWalkMode";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useTauriEventListeners, type SelectionCombineModeApi } from "./hooks/useTauriEventListeners";
import {
  useRocksGenerator,
  useGrassGenerator,
  useAshlarGenerator,
  useFloraGenerator,
  useRopeClothGenerator,
  useRoofGenerator,
  useShapeGenerator,
  useBoneGenerator,
} from "./hooks/useGeneratorState";
import { MascotView } from "./MascotView";
import { SpeechBubbleOverlay, type BubbleInfo } from "./SpeechBubbleOverlay";
import { CollabJoinProgressModal } from "./CollabJoinProgressModal";
import { JoinSessionModal } from "./JoinSessionModal";
import { PreferencesModal } from "./PreferencesModal";
import { StampBookModal } from "./StampBookModal";
import type { StampBookEntryTuple } from "./stampBookStorage";
import { loadRecentJoinUrls } from "./joinRecent";
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
import RadialPingMenu from "./RadialPingMenu";
import GamepadRadialMenu from "./GamepadRadialMenu";
import VirtualCursor from "./VirtualCursor";
import { useGamepad } from "./useGamepad";
import { toolSliceToMode, type SubOptionChoice } from "./gamepadRadialMenuData";
import { PingArrowIndicator } from "./PingArrowIndicator";
import { ToolsSidebar, PaletteSwatches } from "./ToolsSidebar";
import { generateIdea } from "./ideaGenerator";
import packageJson from "../package.json";
import type {
  DepthPhaseData,
  MoodState,

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
  SCULPT_BRUSH_MAX_INDEX,
  loadPaintColorDistrib,
  layoutViewportCssSize,
  viewportCursorOverlayPercent,
  playSeagullSpeech,
  basename,
  lastProjectReopenBlurb,
  sculptBrushShapeToRust,
} from "./constants";

/** App semver from `package.json` (status bar when no file is open). */
const VOXELLE_DESKTOP_VERSION = packageJson.version;

// (Multi-color paint distribution types, presets, and utility functions
// are now in ./types.ts, ./constants.ts, and ./generatorPresets.ts)

/** Avoid duplicate `load_start_screen_logo` in React Strict Mode (dev). */
let startScreenLogoInvokeSent = false;

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
  /** Physical-pixel look deltas coalesced per animation frame (shared between useFlyMode, useWalkMode, and useGamepad). */
  const flyPendingLookDxRef = useRef(0);
  const flyPendingLookDyRef = useRef(0);
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
  // Rope/cloth state now in useRopeClothGenerator hook.
  // Generator useStrokePhase + useState blocks now live in per-generator hooks.
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
  const bone = useBoneGenerator({ activeColorRef, activeMaterialRef });
  const {
    bonePhase,
    boneMode, setBoneMode, boneModeRef,
    boneJointCount, setBoneJointCount,
    boneBoneCount, setBoneBoneCount,
    boneDefaultRadius, setBoneDefaultRadius, boneDefaultRadiusRef,
    ikEnabled, setIkEnabled, ikEnabledRef,
  } = bone;
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
  const [logoLightIntensity, setLogoLightIntensity] = useState(3.0);
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
  const colorPaletteSizeRef = useRef(colorPaletteSize);
  colorPaletteSizeRef.current = colorPaletteSize;

  function clampPalettePos(x: number, y: number): { x: number; y: number } {
    const { w, h } = colorPaletteSizeRef.current;
    const maxX = window.innerWidth - w;
    const maxY = window.innerHeight - h;
    return { x: Math.max(0, Math.min(x, maxX)), y: Math.max(0, Math.min(y, maxY)) };
  }

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

  // ── Tauri event listeners hook ──
  useTauriEventListeners({
    viewportRef,
    sendResize,
    refreshSceneObjects,
    viewportPhysRef,
    surfacePhysRef,
    fillOperationPendingRef,
    collabActiveRef,
    chatPanelOpenRef,
    localPeerIdRef,
    chatToastIdRef,
    pendingJoinUrlRef,
    collabActiveMenuRef,
    startHostMenuRef,
    leaveSessionMenuRef,
    viewportCursorDebugScreenRef,
    pingHudRef,
    setLoadError,
    setCollabBanner,
    setStartScreenLogoLoaded,
    setPathLabel,
    setLoading,
    setLoadProgress,
    setLoadPhase,
    setSpeechBubbles,
    setWorkProgress,
    setWorkPhase,
    setWorkBusy,
    setFillOperationPending,
    setLogoLightControlsVisible,
    setMood,
    setFpsDisplayed,
    setPingMs,
    setNewProjectOpen,
    setJoinModalOpen,
    setChatPanelOpen,
    setPreferencesOpen,
    setStampBookOpen,
    setPingHudTick,
    setChatLines,
    setChatToasts,
    setCollabActive,
    setCollabJoinPending,
    setLocalPeerId,
    setRoster,
    setNatPending,
    setNatError,
    setHostWsUrl,
    setHostWanUrl,
    setHostingCopied,
    setChatInput,
    setInteractionMode,
    setMatchMaterialSelectColor,
    setViewportCursorDebugEnabled,
    setViewportCursorDebugJs,
    setViewportCursorDebugRust,
    setViewportCursorDebugScreen,
    setHideUI,
    setSelectionCount,
    setSelectionCombineMode,
    setRotateDialogOpen,
    setScaleDialogOpen,
  });

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
    // Clear bone session when leaving bone mode
    if (prev === "bone" && interactionMode !== "bone") {
      if (bonePhase.active) {
        bonePhase.cancel();
      } else {
        void invoke("bone_session_clear")
          .then(() => {
            setBoneJointCount(0);
            setBoneBoneCount(0);
          })
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

  // -- Generator hooks --------------------------------------------------------
  const _genCtx = { activeColorRef, activeMaterialRef, generatorSphereRadiusRef };
  const rocks = useRocksGenerator(_genCtx);
  const {
    rockRoughness, setRockRoughness, rockRoughnessRef,
    rockCount, setRockCount, rockCountRef,
    rockClusterRadius, setRockClusterRadius, rockClusterRadiusRef,
    rockSinkDirection, setRockSinkDirection, rockSinkDirectionRef,
    rockSinkAmount, setRockSinkAmount, rockSinkAmountRef,
    rockPreviewSeedRef, rocksPhase,
  } = rocks;
  const grass = useGrassGenerator(_genCtx);
  const {
    grassDensity, setGrassDensity, grassDensityRef,
    grassMaxHeight, setGrassMaxHeight, grassMaxHeightRef,
    grassPreviewSeedRef, grassPhase,
  } = grass;
  const ashlar = useAshlarGenerator({ ..._genCtx, rockRoughnessRef });
  const {
    ashlarThickness, setAshlarThickness, ashlarThicknessRef,
    ashlarPreviewSeedRef, ashlarPhase,
  } = ashlar;
  const flora = useFloraGenerator({ activeColorRef, activeMaterialRef });
  const {
    floraPreset, setFloraPreset,
    floraHeight, setFloraHeight,
    floraGirth, setFloraGirth,
    floraWobble, setFloraWobble,
    floraTaper, setFloraTaper,
    floraStemCount, setFloraStemCount,
    floraClusterRadius, setFloraClusterRadius,
    floraBranchCount, setFloraBranchCount,
    floraBranchDepth, setFloraBranchDepth,
    floraBranchStart, setFloraBranchStart,
    floraBranchSpread, setFloraBranchSpread,
    floraBraidStrands, setFloraBraidStrands,
    floraBraidTwist, setFloraBraidTwist,
    floraCanopy, setFloraCanopy,
    floraPreviewSeedRef, floraPhase,
  } = flora;
  const ropeCloth = useRopeClothGenerator({
    activeColorRef,
    activeMaterialRef,
    selectionStrokeSnapToSurfaceRef,
  });
  const {
    clothGravityDirection, setClothGravityDirection, clothGravityDirectionRef,
    ropeBrushRadiusIndex, setRopeBrushRadiusIndex, ropeBrushRadiusIndexRef,
    ropeBrushShapeUi, setRopeBrushShapeUi, ropeBrushShapeUiRef,
    ropeFirstScreen, setRopeFirstScreen, _ropeFirstScreenRef,
    _ropeFirstVoxel, setRopeFirstVoxel, ropeFirstVoxelRef,
    ropeSag,
    ropeTension, setRopeTension, ropeTensionRef,
    ropePhase,
    clothPins, setClothPins, clothPinsRef,
    clothTension, setClothTension, clothTensionRef,
    clothSimGravityPct, setClothSimGravityPct, clothSimGravityPctRef,
    clothSimStiffnessPct, setClothSimStiffnessPct, clothSimStiffnessPctRef,
    clothSimIterations, setClothSimIterations, clothSimIterationsRef,
    clothSimConstraintPasses, setClothSimConstraintPasses, clothSimConstraintPassesRef,
    clothPhase,
    handleClothPinClick,
  } = ropeCloth;
  const roof = useRoofGenerator();
  const {
    roofStyle, setRoofStyle, roofStyleRef,
    roofHeight, setRoofHeight, roofHeightRef,
    roofHollow, setRoofHollow, roofHollowRef,
    roofPins, setRoofPins, roofPinsRef,
    roofAreaShape, setRoofAreaShape, roofAreaShapeRef,
    roofFirstClick, setRoofFirstClick, roofFirstClickRef,
  } = roof;
  const shape = useShapeGenerator({ activeColorRef, activeMaterialRef });
  const {
    shapeKind, setShapeKind, shapeKindRef,
    shapeSize, setShapeSize, shapeSizeRef,
    shapeRotX, setShapeRotX, shapeRotXRef,
    shapeRotY, setShapeRotY, shapeRotYRef,
    shapeRotZ, setShapeRotZ, shapeRotZRef,
    shapeOverwrite, setShapeOverwrite, shapeOverwriteRef,
    shapePhase,
    shapeGizmoPosRef,
  } = shape;

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
  // Generator refs + useEffect syncs now live in per-generator hooks.
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
      shapePhase.ref.current;
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
            generatorRopeFirstVoxel: ropeFirstVoxelRef.current ?? undefined,
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
            // Shape
            generatorShapeKind: shapeKindRef.current,
            generatorShapeSize: shapeSizeRef.current,
            generatorShapeRotX: shapeRotXRef.current,
            generatorShapeRotY: shapeRotYRef.current,
            generatorShapeRotZ: shapeRotZRef.current,
            generatorShapeOverwrite: shapeOverwriteRef.current,
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
  // handleClothPinClick now lives in useRopeClothGenerator hook.

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
      interactionMode === "select" ||
      interactionMode === "selectByColor" ||
      interactionMode === "selectCoplanar" ||
      interactionMode === "selectCoplanarEmpty" ||
      interactionMode === "selectExtrude" ||
      interactionMode === "stamp" ||
      interactionMode === "punch"
    ) {
      setToolsPane("select");
      return;
    }
    if (
      interactionMode === "add" ||
      interactionMode === "remove" ||
      interactionMode === "paint" ||
      interactionMode === "eyedropper"
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
    if (m === "bone") return "bone";
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
    // Clamp into view whenever the palette becomes visible
    if (colorPaletteFloating) {
      setColorPalettePos((p) => clampPalettePos(p.x, p.y));
    }
  }, [colorPaletteFloating]);

  useEffect(() => {
    if (!colorPaletteFloating) return;
    const onResize = () => setColorPalettePos((p) => clampPalettePos(p.x, p.y));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
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

  // ── Keyboard shortcuts hook (undo/redo, save, ping, selection translate/rotate, fill cancel) ──
  const { onRadialSelect } = useKeyboardShortcuts({
    preferencesOpen,
    stampBookOpen,
    joinModalOpen,
    newProjectOpen,
    collabJoinPending,
    loading,
    workBusy,
    fillOperationPending,
    selectionCount,
    lastViewportPickNormRef,
    pendingPingRef,
    radialHoldTimerRef,
    lastCursorScreenRef,
    fillOperationPendingRef,
    workPhaseRef,
    pingHudRef,
    setPingHudTick,
    setRadialMenu,
  });

  const clearPreview = useCallback(() => {
    void invoke("sync_preview_input", {
      args: buildSyncPreviewPayload(-1, 0, "navigate"),
    }).catch(() => {});
    void invoke("squishy_gizmo_pointer_up").catch(() => {});
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
    shapeKind,
    shapeSize,
    shapeRotX,
    shapeRotY,
    shapeRotZ,
    shapeOverwrite,
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

  // ── Gamepad / controller support ──────────────────────────────────────
  const virtualCursorElRef = useRef<HTMLDivElement | null>(null);
  const gamepad = useGamepad({
    flyPendingLookDxRef,
    flyPendingLookDyRef,
    onToolActivate: useCallback(() => {
      void runStrokeAtScreen(0.5, 0.5, {});
    }, []),
    onEyedropper: useCallback(() => {
      void invoke<{ color: number; material: string } | null>(
        "voxel_pick_color_at_screen",
        {
          args: {
            nx: 0.5,
            ny: 0.5,
            tool: "add",
            color: activeColorRef.current,
            material: activeMaterialRef.current,
            brushRadius: 0,
            brushShape: brushShapeRef.current,
          },
        },
      )
        .then((r) => {
          if (r) {
            setActiveColor(r.color);
            setActiveMaterial(r.material);
          }
        })
        .catch(() => {});
    }, []),
    onUndo: useCallback(() => {
      void invoke("voxel_undo").catch(() => {});
    }, []),
    onToolSelect: useCallback(
      (sliceId: string) => {
        const { interactionMode: im, toolsPane: tp } = toolSliceToMode(sliceId);
        setInteractionMode(im);
        setToolsPane(tp);
      },
      [],
    ),
    onSubOptionSelect: useCallback((choice: SubOptionChoice) => {
      if (choice.kind === "selectionMethod") {
        const s = selectionMethodToState(choice.method);
        setDrawStrokeMode(s.drawStrokeMode);
        setStrokeDrawStyle(s.strokeDrawStyle);
        setStrokeFamilyVariant(s.strokeFamilyVariant);
        setSprayDensity(s.sprayDensity);
      } else if (choice.kind === "sculptMode") {
        setSculptStrokeMode(choice.mode);
      }
    }, []),
    onRequestFlyMode: useCallback(() => {
      if (interactionModeRef.current !== "fly" && interactionModeRef.current !== "walk") {
        setInteractionMode("fly");
        setToolsPane("fly");
      }
    }, []),
    onToggleLocomotion: useCallback((direction: "fly" | "walk") => {
      setInteractionMode(direction);
      setToolsPane(direction);
    }, []),
    interactionModeRef,
    cursorElRef: virtualCursorElRef,
  });

  // ── Fly mode hook (WASD, mouse look, RAF tick) ──
  const {
    flySpeed,
    setFlySpeed,
    releaseFlyMouseLook,
    activateFlyMouseLook,
    flyMouseLookActiveRef,
    keysDownRef,
    flyRafRef,
    flyLastClientRef,
    flySkipNextFlyMoveRef,
  } = useFlyMode({
    interactionMode,
    viewportRef,
    pollGamepad: gamepad.pollGamepad,
    flyPendingLookDxRef,
    flyPendingLookDyRef,
  });

  // ── Walk mode hook (first-person with gravity, collision, jumping) ──
  useWalkMode({
    interactionMode,
    viewportRef,
    pollGamepad: gamepad.pollGamepad,
    releaseFlyMouseLook,
    flyMouseLookActiveRef,
    keysDownRef,
    flyRafRef,
    flyLastClientRef,
    flySkipNextFlyMoveRef,
    flyPendingLookDxRef,
    flyPendingLookDyRef,
  });

  // ── Viewport pointer handlers (extracted to useViewportPointer) ──
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const _vpLocalsRef = useRef<any>(null);
  _vpLocalsRef.current = {
    activeColorRef,
    activeMaterialRef,
    activePointerIdRef,
    ashlarPreviewSeedRef,
    brushClipBottomHalfRef,
    brushRadiusRef,
    brushShapeRef,
    capturedPointerIdRef,
    currentStrokeSeedRef,
    dragDidEditRef,
    drawStrokeModeRef,
    extrudeDirectionRefRef,
    extrudeEndCapRef,
    extrudeGizmoRef,
    extrudeProfileRef,
    extrudeRedragRef,
    extrudeStartNormRef,
    extrudeTaperEndRef,
    extrudeTaperRef,
    extrudeTaperStartRef,
    eyedropperReturnModeRef,
    floraPreviewSeedRef,
    flyMouseLookActiveRef,
    generatorKindRef,
    generatorSphereRadiusRef,
    gestureRef,
    gizmoHoverRef,
    gizmoRef,
    grassPreviewSeedRef,
    interactionBlockedRef,
    interactionModeRef,
    lastCursorScreenRef,
    lastRef,
    lastStrokeEditMsRef,
    lastStrokeNormRef,
    lastTerrainHoverMsRef,
    lastViewportPickNormRef,
    lastWallHoverMsRef,
    matchMaterialSelectColorRef,
    maxPointerMoveRef,
    mirrorXRef,
    mirrorYRef,
    mirrorZRef,
    onPointerUpRef,
    paintColorDistribRef,
    pendingPointerUpRef,
    planeAxisRef,
    pointerStartRef,
    probingRef,
    rockPreviewSeedRef,
    roofAreaShapeRef,
    roofFirstClickRef,
    roofPinsRef,
    sculptBrushFalloffRef,
    sculptBrushRadiusRef,
    sculptBrushShapeUiRef,
    sculptBrushStrengthRef,
    sculptSmoothPassesRef,
    sculptSmoothVariantRef,
    sculptStrokeModeRef,
    selectedColorsRef,
    selectionStrokeBegunRef,
    selectionStrokeSnapToSurfaceRef,
    smoothAggressivenessRef,
    smoothLaplacianIterationsRef,
    smoothLaplacianRelaxPctRef,
    smoothNeighborRadiusRef,
    sprayDensityRef,
    sprayDirectionRef,
    squishyModeRef,
    stampOriginXRef,
    stampOriginZRef,
    stampRotXRef,
    stampRotYRef,
    stampRotZRef,
    startScreenLogoLoadedRef,
    strokeClickRef,
    strokeDrawStyleRef,
    strokePolygonLastScreenRef,
    strokePolygonVertsRef,
    strokeShiftKeyRef,
    strokeViewportStartRef,
    surfacePhysRef,
    terrainBaseYRef,
    terrainFlattenUseBaseYRef,
    terrainSculptOpRef,
    terrainSmoothRadiusRef,
    terrainSubVoxelRef,
    viewportCursorDebugRafRef,
    viewportCursorDebugScreenRef,
    viewportPhysRef,
    viewportRef,
    wallAreaShapeRef,
    wallAxisAlignRef,
    wallHeightVoxRef,
    wallLockStartHeightRef,
    wallSculptPolygonVertsRef,
    wallWidthIndexRef,
    cuboidDepthRef,
    cylinderDepthRef,
    setActiveColor,
    setActiveMaterial,
    setCuboidDepthUi,
    setCylinderDepthUi,
    setInteractionMode,
    setRoofFirstClick,
    setRoofPins,
    setRopeFirstScreen,
    setRopeFirstVoxel,
    setSelectionCount,
    setSquishyBallCount,
    setStrokePolygonVerts,
    setTerrainHoverY,
    setViewportCursorDebugJs,
    setViewportCursorDebugRust,
    setViewportCursorDebugScreen,
    setWallSculptPolygonVerts,
    ashlarPhase,
    clothPhase,
    cuboidPhase,
    cylinderPhase,
    extrudePhase,
    floraPhase,
    grassPhase,
    rocksPhase,
    ropePhase,
    shapePhase,
    shapeGizmoPosRef,
    setShapeSize,
    setShapeRotX,
    setShapeRotY,
    setShapeRotZ,
    squishyPhase,
    bonePhase,
    boneModeRef,
    boneDefaultRadiusRef,
    ikEnabledRef,
    setBoneJointCount,
    setBoneBoneCount,
    loading,
    workBusy,
    fillOperationPending,
    viewportCursorDebugEnabled,
    ropeFirstScreen,
    mergedStrokeAux,
    buildSyncPreviewPayload,
    previewModeForSync,
    runStrokeAtScreen,
    clearPreview,
    handleClothPinClick,
    beginFillOperation,
    endFillOperation,
    askFillConfirmation,
    activateFlyMouseLook,
    releaseFlyMouseLook,
  };
  const {
    onPointerDown, onPointerMove, onPointerUp, onPointerLeave,
    onGotPointerCapture, onLostPointerCapture, onWheel,
    commitWallSculptPolygonStroke,
  } = useViewportPointer(_vpLocalsRef);

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
      shapePhase.active ||
      (generatorKind === "cloth" && clothPins.length > 0) ||
      (generatorKind === "roof" && (roofPins.length > 0 || roofFirstClick !== null)) ||
      showPolygonPhaseHud ||
      showWallSculptPolygonHud ||
      bonePhase.active);

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
        <ToolsSidebar
          showEditorChrome={showEditorChrome}
          loading={loading}
          workBusy={workBusy}
          toolsPaneFloating={toolsPaneFloating}
          setToolsPaneFloating={setToolsPaneFloating}
          sidebarExpanded={sidebarExpanded}
          setSidebarExpanded={setSidebarExpanded}
          toolPanePos={toolPanePos}
          onToolPaneDragDown={onToolPaneDragDown}
          toolsPane={toolsPane}
          setToolsPane={setToolsPane}
          interactionMode={interactionMode}
          setInteractionMode={setInteractionMode}
          flySpeed={flySpeed}
          setFlySpeed={setFlySpeed}
          selectionMethod={selectionMethod}
          setDrawStrokeMode={setDrawStrokeMode}
          setStrokeDrawStyle={setStrokeDrawStyle}
          setSprayDensity={setSprayDensity}
          setStrokeFamilyVariant={setStrokeFamilyVariant}
          selectionCount={selectionCount}
          stampBookPatternActive={stampBookPatternActive}
          activeColor={activeColor}
          setActiveColor={setActiveColor}
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
          activeMaterial={activeMaterial}
          setActiveMaterial={setActiveMaterial}
          sculptStrokeMode={sculptStrokeMode}
          setSculptStrokeMode={setSculptStrokeMode}
          generatorKind={generatorKind}
          setGeneratorKind={setGeneratorKind}
          ropePhase={ropePhase}
          clothPhase={clothPhase}
          rocksPhase={rocksPhase}
          grassPhase={grassPhase}
          ashlarPhase={ashlarPhase}
          floraPhase={floraPhase}
          shapePhase={shapePhase}
          ropeFirstScreen={ropeFirstScreen}
          setRopeFirstScreen={setRopeFirstScreen}
          setClothPins={setClothPins}
          clothPinsRef={clothPinsRef}
          colorPaletteFloating={colorPaletteFloating}
          setColorPaletteFloating={setColorPaletteFloating}
          bonePhase={bonePhase}
          boneMode={boneMode}
          setBoneMode={setBoneMode}
          boneJointCount={boneJointCount}
          boneBoneCount={boneBoneCount}
          boneDefaultRadius={boneDefaultRadius}
          setBoneDefaultRadius={setBoneDefaultRadius}
          ikEnabled={ikEnabled}
          setIkEnabled={setIkEnabled}
        />
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
                    setColorPalettePos(clampPalettePos(origX + dx, origY + dy));
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
              <div className="viewport-top-center-hud" onPointerDown={(e) => e.stopPropagation()} onPointerMove={(e) => e.stopPropagation()}>
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
                      min={-1}
                      max={1}
                      step={0.02}
                      value={ropeTension}
                      onChange={(ev) => setRopeTension(Number(ev.target.value))}
                      style={{ width: 60 }}
                      title="-1 = very loose, 0 = loose, 1 = taut"
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
                {shapePhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Shape settings"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Shape</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      Adjust in sidebar — Enter to place
                    </span>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => shapePhase.cancel()}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => shapePhase.commit()}
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
                {bonePhase.active ? (
                  <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Bone session">
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Bone</span>
                    <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
                      {bonePhase.snapshot?.phase === "build"
                        ? "Place joints — click Next to pose"
                        : "Pose joints — Enter to commit, Esc to cancel"}
                    </span>
                    {bonePhase.snapshot?.phase === "build" ? (
                      <>
                        <button onClick={() => bonePhase.cancel()}>Cancel</button>
                        <button onClick={() => { setBoneMode("edit"); bonePhase.advance(); }}>Next</button>
                      </>
                    ) : (
                      <>
                        <button onClick={() => { setBoneMode("add"); bonePhase.retreat(); }}>Back</button>
                        <button onClick={() => bonePhase.commit()}>Done</button>
                      </>
                    )}
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
                toolsPane === "generators" && generatorKind === "flora" ? " is-generator-wide" : ""
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
                  shapeKind={shapeKind}
                  setShapeKind={setShapeKind}
                  shapeSize={shapeSize}
                  setShapeSize={setShapeSize}
                  shapeOverwrite={shapeOverwrite}
                  setShapeOverwrite={setShapeOverwrite}
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
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 60 }}>Intensity</span>
                <input
                  type="range"
                  min={0}
                  max={5}
                  step={0.01}
                  value={logoLightIntensity}
                  style={{ flex: 1 }}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setLogoLightIntensity(val);
                    void invoke("logo_set_light_intensity", { intensity: val });
                  }}
                />
                <span style={{ width: 36, textAlign: "right", fontFamily: "monospace" }}>
                  {logoLightIntensity.toFixed(2)}
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

      {/* ── Gamepad radial menus (LT / RT triggers) ───────────────────── */}
      <GamepadRadialMenu
        visible={gamepad.radialMenu != null}
        slices={gamepad.radialSlices}
        selectedIndex={gamepad.selectedSliceIndex}
        title={gamepad.radialMenu === "tools" ? "Tool" : gamepad.radialMenu === "subOptions" ? "Options" : undefined}
      />

      {/* ── Gamepad speed HUD ─────────────────────────────────────────── */}
      {gamepad.connected && gamepad.speedMultiplier > 1 && (
        <div
          style={{
            position: "fixed",
            bottom: 48,
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(0,0,0,0.5)",
            color: "rgba(255,255,255,0.8)",
            padding: "4px 12px",
            borderRadius: 6,
            fontSize: 13,
            fontWeight: 600,
            pointerEvents: "none",
            zIndex: 99990,
            userSelect: "none",
          }}
        >
          {gamepad.speedMultiplier}x
        </div>
      )}

      {/* ── Gamepad virtual cursor (X to toggle) ─────────────────────── */}
      <VirtualCursor ref={virtualCursorElRef} visible={gamepad.cursorMode} />

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
