import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow, LogicalPosition } from "@tauri-apps/api/window";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import { CollabJoinProgressModal } from "./CollabJoinProgressModal";
import { JoinSessionModal } from "./JoinSessionModal";
import { PreferencesModal } from "./PreferencesModal";
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
import { MATERIAL_BUILTIN_PALETTE_HEX } from "./materialBuiltinPalette";
import { ViewportCameraHud } from "./ViewportCameraHud";
import { useStrokePhase } from "./useStrokePhase";

/** Data carried through the cuboid/cylinder solid depth phase. */
interface DepthPhaseData {
  lineStart: { nx: number; ny: number };
  endNorm: { nx: number; ny: number };
}
import { ViewportSettingsSidebar } from "./ViewportSettingsSidebar";
import packageJson from "../package.json";

/** App semver from `package.json` (status bar when no file is open). */
const VOXELLE_DESKTOP_VERSION = packageJson.version;

/** Desktop viewer: cap new-project grid edge length (web allows larger). */
const MAX_GRID_SIZE = 256;

// ── Mood state ──────────────────────────────────────────────────────
interface MoodState {
  // vignette (desktop-only)
  vignette: number;
  // grain
  grainEnabled: boolean;
  grainStrength: number;
  grainAnimated: boolean;
  grainSpeed: number;
  grainColorful: boolean;
  // atmosphere
  atmEnabled: boolean;
  atmColor: string;
  atmThickness: number;
  atmDensity: number;
  atmAerial: boolean;
  atmPositiveSide: boolean;
  atmPlaneNx: number;
  atmPlaneNy: number;
  atmPlaneNz: number;
  atmPlaneC: number;
  atmHeightBias: number;
  atmHeightFalloff: number;
  atmDriftEnabled: boolean;
  atmDriftAmount: number;
  atmDriftScale: number;
  atmDriftSpeed: number;
  // distance tint
  dtEnabled: boolean;
  dtNearColor: string;
  dtMidColor: string;
  dtFarColor: string;
  dtNearDist: number;
  dtFarDist: number;
  dtStrength: number;
  // sun shafts
  ssEnabled: boolean;
  ssStrength: number;
  ssDecay: number;
  ssDensity: number;
  ssWeight: number;
  ssSamples: number;
}

function defaultMoodState(): MoodState {
  return {
    vignette: 0,
    grainEnabled: false,
    grainStrength: 0.12,
    grainAnimated: true,
    grainSpeed: 1,
    grainColorful: true,
    atmEnabled: false,
    atmColor: "#c8d4e0",
    atmThickness: 28,
    atmDensity: 0.85,
    atmAerial: true,
    atmPositiveSide: false,
    atmPlaneNx: 0,
    atmPlaneNy: 0,
    atmPlaneNz: 0,
    atmPlaneC: 0,
    atmHeightBias: 0,
    atmHeightFalloff: 120,
    atmDriftEnabled: false,
    atmDriftAmount: 0.2,
    atmDriftScale: 0.02,
    atmDriftSpeed: 0.2,
    dtEnabled: false,
    dtNearColor: "#ffffff",
    dtMidColor: "#c8d4e0",
    dtFarColor: "#8fa3bf",
    dtNearDist: 16,
    dtFarDist: 140,
    dtStrength: 0.6,
    ssEnabled: false,
    ssStrength: 0.7,
    ssDecay: 0.92,
    ssDensity: 0.8,
    ssWeight: 0.6,
    ssSamples: 32,
  };
}

/** Helper: update one mood field. */
function moodWith(prev: MoodState, patch: Partial<MoodState>): MoodState {
  return { ...prev, ...patch };
}

const LS_RENDERING_MODE = "voxelleDesktopRenderingMode";
const LS_SIDEBAR_EXPANDED = "voxelleSidebarExpanded";
const LS_RIGHT_SIDEBAR_EXPANDED = "voxelleRightSidebarExpanded";
const LS_TOOLS_FLOATING = "voxelleToolsFloating";
const LS_TOOLS_FLOAT_POS = "voxelleToolsFloatPos";
/** `localStorage` = `"1"`: show JS vs Rust viewport cursor overlay (see `get_viewport_cursor_debug`). */
const LS_VIEWPORT_CURSOR_DEBUG = "voxelleDebugViewportCursor";
/** `localStorage` = `"1"`: GPU uses full swapchain; picking uses client ÷ layout viewport (see `layoutViewportCssSize`). */
const LS_FULL_SURFACE_VIEWPORT = "voxelleFullSurfaceViewport";

/**
 * CSS layout viewport size for mapping `clientX`/`clientY` and layout fractions to the native surface.
 * Prefer `document.documentElement.clientWidth/Height` over `window.inner*` so the denominator matches
 * the pointer coordinate span (inner includes scrollbar gutter; client does not).
 */
function layoutViewportCssSize(): { w: number; h: number } {
  const de = document.documentElement;
  const w = de.clientWidth || window.innerWidth;
  const h = de.clientHeight || window.innerHeight;
  return { w: Math.max(1, w), h: Math.max(1, h) };
}

function readFullSurfaceViewportEnabled(): boolean {
  try {
    return localStorage.getItem(LS_FULL_SURFACE_VIEWPORT) === "1";
  } catch {
    return false;
  }
}

/** Map texture-normalized nx, ny to position inside `.viewport` for debug overlay markers. */
function viewportCursorOverlayPercent(
  fullSurfaceViewport: boolean,
  nx: number,
  ny: number,
  rect: DOMRectReadOnly | null,
): { leftPct: number; topPct: number } {
  if (!fullSurfaceViewport || !rect || rect.width <= 0 || rect.height <= 0) {
    return { leftPct: nx * 100, topPct: ny * 100 };
  }
  const { w: lw, h: lh } = layoutViewportCssSize();
  return {
    leftPct: ((nx * lw - rect.left) / rect.width) * 100,
    topPct: ((ny * lh - rect.top) / rect.height) * 100,
  };
}

/** Payload from `get_viewport_cursor_debug` (camelCase). */
type ViewportCursorDebugPayload = {
  viewportWidth: number;
  viewportHeight: number;
  surfaceWidth: number;
  surfaceHeight: number;
  viewportOriginX: number;
  viewportOriginY: number;
  previewNx: number | null;
  previewNy: number | null;
  texelSx: number | null;
  texelSy: number | null;
  rayOriginX: number | null;
  rayOriginY: number | null;
  rayOriginZ: number | null;
  rayDirX: number | null;
  rayDirY: number | null;
  rayDirZ: number | null;
  projCubeNx: number | null;
  projCubeNy: number | null;
  /** Projected voxel center (same as hover mesh anchor); differs from proj cube on oblique views. */
  projCenterNx: number | null;
  projCenterNy: number | null;
};

/** Browser pointer position for the debug overlay (CSS pixels). */
type ViewportCursorDebugScreen = {
  clientX: number;
  clientY: number;
  /** Offset inside `.viewport` (`client` − `getBoundingClientRect()`). */
  relX: number;
  relY: number;
  innerWidth: number;
  innerHeight: number;
  /** `documentElement.client*` span; matches `sendResize` / `layoutViewportCssSize`. */
  layoutWidth: number;
  layoutHeight: number;
  rectLeft: number;
  rectTop: number;
  rectWidth: number;
  rectHeight: number;
};

/** Avoid duplicate `load_start_screen_logo` in React Strict Mode (dev). */
let startScreenLogoInvokeSent = false;

type RenderingMode = "greedy" | "marchingCubes" | "dualContour" | "ray";

type StartShape =
  | "cube"
  | "orb"
  | "cylinder"
  | "hollowCube"
  | "plane"
  | "circle"
  | "empty";

type RosterEntry = {
  peerId: number;
  displayName: string;
  colorRgb: number;
  isLeader: boolean;
  canEdit: boolean;
};

type LastSessionInfo = {
  lastDocumentPath: string | null;
  documentBasename: string | null;
  autosavePath: string | null;
  documentExists: boolean;
  autosaveExists: boolean;
  autosaveNewerThanDocument: boolean;
};

type SceneObjectRow = {
  id: number;
  parentId: number | null;
  name: string;
  visible: boolean;
  sortOrder: number;
  translation: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
};

type ChatToast = { id: number; text: string };

const CHAT_TOAST_CAP = 5;

const PING_HUD_MS = 2800;
const PING_MP3_URL = `${import.meta.env.BASE_URL}ping.mp3`;

function playPingSound() {
  try {
    const a = new Audio(PING_MP3_URL);
    a.volume = 0.85;
    void a.play().catch(() => {});
  } catch {
    /* ignore */
  }
}

function basename(path: string): string {
  const n = path.replace(/\\/g, "/");
  const i = n.lastIndexOf("/");
  return i >= 0 ? n.slice(i + 1) : n;
}

/** Maps low-level Tauri updater errors to text users can act on. */
function userFacingUpdaterError(err: unknown): string {
  const raw =
    err instanceof Error ? err.message : String(err ?? "unknown error");
  if (
    raw.includes("None of the fallback platforms") &&
    raw.includes("were found in the response")
  ) {
    let platform = "your computer";
    if (raw.includes("darwin-x86_64")) {
      platform = "Intel Macs";
    } else if (raw.includes("darwin-aarch64") || raw.includes("aarch64")) {
      platform = "Apple Silicon Macs";
    } else if (raw.includes("windows")) {
      platform = "Windows";
    } else if (raw.includes("linux")) {
      platform = "Linux";
    }
    return [
      `This release’s update file doesn’t include a build for ${platform}.`,
      "That often happens when a release only ships some platforms, or the update manifest (latest.json) wasn’t merged correctly.",
      "",
      "What you can do: download the installer or archive that matches your system from the releases page and install it manually:",
      "https://github.com/Velfi/Voxelle-Desktop/releases",
    ].join("\n");
  }
  return raw.length > 0 ? raw : "Update failed (unknown error).";
}

/** Must match Rust `ONGOING_UNSAVED_PROJECT_LABEL` (`get_last_session_info`). */
const ONGOING_UNSAVED_PROJECT_LABEL = "An unsaved project";

/** Optional note when reopening (backup vs file). */
function lastProjectReopenBlurb(info: LastSessionInfo): string | null {
  if (!info.lastDocumentPath) return null;
  if (
    info.lastDocumentPath === ONGOING_UNSAVED_PROJECT_LABEL &&
    info.autosaveExists
  ) {
    return "The project is an autosave and will be overwritten by your next autosave.";
  }
  if (!info.documentExists && info.autosaveExists) {
    return "Couldn't find the file — opened your backup instead.";
  }
  if (
    info.documentExists &&
    info.autosaveExists &&
    info.autosaveNewerThanDocument
  ) {
    return "Backup is newer than the saved file.";
  }
  if (
    info.documentExists &&
    info.autosaveExists &&
    !info.autosaveNewerThanDocument
  ) {
    return null;
  }
  return null;
}

type InteractionMode =
  | "navigate"
  | "fly"
  | "add"
  | "remove"
  | "paint"
  | "eyedropper"
  | "select"
  | "selectByColor"
  | "selectCoplanar"
  | "selectCoplanarEmpty"
  | "stamp"
  | "punch"
  | "sculpt"
  | "generator"
  | "squishy";

type ToolsPane =
  | "hand"
  | "draw"
  | "sculpt"
  | "generators"
  | "squishy"
  | "mood"
  | "fly";

/** Matches Rust `SculptStrokeMode` (JSON camelCase). */
type SculptStrokeModeApi =
  | "draw"
  | "smooth"
  | "gouge"
  | "wall"
  | "terrain"
  | "extrude";
/** Matches Rust `TerrainSculptOp`. */
type TerrainSculptOpApi = "raise" | "lower" | "smooth";

type GeneratorKindId =
  | "rocks"
  | "grass"
  | "rope"
  | "cloth"
  | "ashlar"
  | "flora"
  | "roof"
  | "piscina"
  | "insecta"
  | "fauna";
type ClothGravityDirectionId =
  | "down"
  | "up"
  | "left"
  | "right"
  | "forward"
  | "back";

type BrushShape = "sphere" | "cube" | "pyramid" | "square" | "circle";

/** Web `SculptBrushShape`; Rust now accepts all four names directly. */
type SculptBrushShapeUi = "square" | "circle" | "cube" | "sphere";

/** Web `MAX_BRUSH_SIZE - 1` (slider index 0..63 → display 1..64). */
const SCULPT_BRUSH_MAX_INDEX = 63;

function sculptBrushShapeToRust(s: SculptBrushShapeUi): BrushShape {
  return s;
}

/** Web `WallAreaShape` / `SprayDirection` (Rust serde camelCase). */
type WallAreaShapeApi = "brush" | "circle" | "polygon";
/** Web `SculptSmoothVariant`; Rust serde camelCase. */
type SculptSmoothVariantApi = "majority" | "meshLaplacian";
type SprayDirectionApi =
  | "auto"
  | "none"
  | "right"
  | "left"
  | "up"
  | "down"
  | "back"
  | "forward";

const MATERIAL_OPTIONS: { id: string; label: string }[] = [
  { id: "plastic", label: "Plastic" },
  { id: "metal", label: "Metal" },
  { id: "rubber", label: "Rubber" },
  { id: "glass", label: "Glass" },
  { id: "water", label: "Water" },
  { id: "glow", label: "Glow" },
];

function SymmetryColorSidebarSections(props: {
  loading: boolean;
  workBusy: boolean;
  activeColor: number;
  setActiveColor: (n: number) => void;
  interactionMode: InteractionMode;
  setInteractionMode: (m: InteractionMode) => void;
}) {
  const {
    loading,
    workBusy,
    activeColor,
    setActiveColor,
    interactionMode,
    setInteractionMode,
  } = props;
  return (
    <div className="sidebar-symmetry-color-panel">
      <div className="sidebar-section-label">Symmetry</div>
      <div className="sidebar-mode-grid sidebar-mode-grid-3">
        <button type="button" className="sidebar-mode-btn" disabled>
          <span className="sidebar-mode-label">X</span>
        </button>
        <button type="button" className="sidebar-mode-btn" disabled>
          <span className="sidebar-mode-label">Y</span>
        </button>
        <button type="button" className="sidebar-mode-btn" disabled>
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
        <div
          className="sidebar-palette-swatches"
          role="group"
          aria-label="Material color palette"
        >
          {MATERIAL_BUILTIN_PALETTE_HEX.map((hex) => {
            const rgb = Number.parseInt(hex.slice(1), 16);
            const isActive = (activeColor & 0xffffff) === rgb;
            return (
              <button
                key={hex}
                type="button"
                className={
                  isActive
                    ? "sidebar-palette-swatch is-active"
                    : "sidebar-palette-swatch"
                }
                style={{ backgroundColor: hex }}
                title={hex}
                aria-label={`Select color ${hex}`}
                aria-pressed={isActive}
                disabled={loading || workBusy}
                onClick={() => setActiveColor(rgb)}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
  /** Viewport render target in physical pixels (matches projection / picking); from Rust. */
  const viewportPhysRef = useRef({ w: 0, h: 0 });
  /** Swapchain drawable in physical pixels (authoritative native size; may differ from inner×dpr). */
  const surfacePhysRef = useRef({ w: 0, h: 0 });
  /** Last layout viewport CSS size (`layoutViewportCssSize`) — when these change, do not use stale surface for mapping until Rust syncs. */
  const lastLayoutViewportCssRef = useRef({ w: 0, h: 0 });
  /** When true, `viewer_resize` uses the full swapchain; picking uses window-normalized coords. */
  const fullSurfaceViewportRef = useRef(readFullSurfaceViewportEnabled());
  const lastRef = useRef({ x: 0, y: 0 });
  /** Last pointer position over `.viewport` in physical pixels (for Z = ping pick). */
  const lastViewportPickNormRef = useRef<{ nx: number; ny: number } | null>(
    null,
  );
  const pointerStartRef = useRef<{ x: number; y: number } | null>(null);
  const maxPointerMoveRef = useRef(0);
  /** After pick probe: camera orbit/pan/dolly vs voxel click-to-edit (matches web: no hit → camera). */
  const gestureRef = useRef<{
    pointerId: number;
    mode: "camera" | "voxel" | "squishyGizmo";
  } | null>(null);
  const probingRef = useRef(false);
  const activePointerIdRef = useRef<number | null>(null);
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
  const strokeViewportStartRef = useRef<{ nx: number; ny: number } | null>(
    null,
  );
  /** Previous brush sample (normalized viewport). */
  const lastStrokeNormRef = useRef<{ nx: number; ny: number } | null>(null);
  const lastStrokeEditMsRef = useRef(0);
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
  const [interactionMode, setInteractionMode] =
    useState<InteractionMode>("navigate");
  const [mood, setMood] = useState<MoodState>(() => defaultMoodState());
  const [selectionCount, setSelectionCount] = useState(0);
  const [viewportCursorDebugEnabled, setViewportCursorDebugEnabled] = useState(
    () => {
      try {
        return localStorage.getItem(LS_VIEWPORT_CURSOR_DEBUG) === "1";
      } catch {
        return false;
      }
    },
  );
  const [viewportCursorDebugJs, setViewportCursorDebugJs] = useState<{
    nx: number;
    ny: number;
  } | null>(null);
  const [viewportCursorDebugRust, setViewportCursorDebugRust] =
    useState<ViewportCursorDebugPayload | null>(null);
  const [viewportCursorDebugScreen, setViewportCursorDebugScreen] =
    useState<ViewportCursorDebugScreen | null>(null);
  /** Synchronous copy for debug ingest (React state can lag behind rAF). */
  const viewportCursorDebugScreenRef = useRef<ViewportCursorDebugScreen | null>(
    null,
  );
  const viewportCursorDebugRafRef = useRef<number | null>(null);
  const [fullSurfaceViewportEnabled, setFullSurfaceViewportEnabled] = useState(
    readFullSurfaceViewportEnabled,
  );
  const [matchMaterialSelectColor, setMatchMaterialSelectColor] =
    useState(false);
  const matchMaterialSelectColorRef = useRef(false);
  const [activeColor, setActiveColor] = useState(0x8899aa);
  const [activeMaterial, setActiveMaterial] = useState("plastic");
  const [brushRadius, setBrushRadius] = useState(0);
  const [brushShape, setBrushShape] = useState<BrushShape>("sphere");
  /** Brush: clip to half-space along the face outward normal from the pick (see Rust `brush_clip_half_normal_from_screen`). */
  const [brushClipBottomHalf, setBrushClipBottomHalf] = useState(false);
  const [strokeDrawStyle, setStrokeDrawStyle] =
    useState<StrokeDrawStyle>("line");
  const [strokeFamilyVariant, setStrokeFamilyVariant] =
    useState<StrokeFamilyVariant>("stroke");
  const strokeFamilyVariantRef = useRef<StrokeFamilyVariant>("stroke");
  const [drawStrokeMode, setDrawStrokeMode] =
    useState<DrawStrokeModeApi>("line");
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
  const [squishyMode, setSquishyMode] = useState<"add" | "edit" | "delete">(
    "add",
  );
  const squishyModeRef = useRef<"add" | "edit" | "delete">("add");
  const [squishyHollow, setSquishyHollow] = useState(false);
  const [squishyWallThickness, setSquishyWallThickness] = useState(1);
  const [squishySnapToSurface, setSquishySnapToSurface] = useState(true);
  const [selectionStrokeSnapToSurface, setSelectionStrokeSnapToSurface] =
    useState(true);
  const [selectionStrokeAxisAlign, setSelectionStrokeAxisAlign] =
    useState(true);
  const selectionStrokeSnapToSurfaceRef = useRef(true);
  const selectionStrokeAxisAlignRef = useRef(true);
  const [surfacePlaneHollow, setSurfacePlaneHollow] = useState(false);
  const surfacePlaneHollowRef = useRef(false);
  const [sprayConstrainToPlane, setSprayConstrainToPlane] = useState(false);
  const sprayConstrainToPlaneRef = useRef(false);
  const [spraySizeRange, setSpraySizeRange] = useState(false);
  const spraySizeRangeRef = useRef(false);
  const [fillConstrainToPlane, setFillConstrainToPlane] = useState(false);
  const fillConstrainToPlaneRef = useRef(false);
  const [squishyBallCount, setSquishyBallCount] = useState(0);
  const [strokePolygonVerts, setStrokePolygonVerts] = useState<
    [number, number, number][]
  >([]);
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
  const strokePolygonLastScreenRef = useRef<{ nx: number; ny: number } | null>(
    null,
  );
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
      setRopeFirstScreen(null);
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      const { nx1, ny1, nx2, ny2 } = snap.data;
      void invoke("generator_rope_at_screen", {
        args: {
          nx1,
          ny1,
          nx2,
          ny2,
          sag: ropeSagRef.current,
          tension: ropeTensionRef.current,
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
  const [ropeSag, setRopeSag] = useState(2.5);
  /** 0 = loose, 1 = taut (web ropeTension). */
  const [ropeTension, setRopeTension] = useState(0.5);
  const [ropeBrushRadiusIndex, setRopeBrushRadiusIndex] = useState(2);
  const [ropeBrushShapeUi, setRopeBrushShapeUi] = useState<"sphere" | "cube">(
    "sphere",
  );
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
  const [rockRoughness, setRockRoughness] = useState(0.45);
  const [grassDensity, setGrassDensity] = useState(4);
  const [grassMaxHeight, setGrassMaxHeight] = useState(3);
  // Flora params
  const [floraHeight, setFloraHeight] = useState(14);
  const [floraGirth, setFloraGirth] = useState(0);
  const [floraStemCount, setFloraStemCount] = useState(1);
  const [floraBranchCount, setFloraBranchCount] = useState(0);
  const [floraCanopy, setFloraCanopy] = useState(0.18);
  // Piscina params
  const [piscinaSpecies, setPiscinaSpecies] = useState<string>("trout");
  const [piscinaLength, setPiscinaLength] = useState(16);
  // Insecta params
  const [insectaSpecies, setInsectaSpecies] = useState<string>("bee");
  const [insectaTotalLength, setInsectaTotalLength] = useState(24);
  // Fauna params
  const [faunaStance, setFaunaStance] = useState<string>("quadruped");
  const [faunaArchetype, setFaunaArchetype] = useState<string>("ungulate");
  // Roof params
  const [roofStyle, setRoofStyle] = useState<string>("gable");
  const [roofHeight, setRoofHeight] = useState(6);
  const [roofHollow, setRoofHollow] = useState(false);
  const [roofPins, setRoofPins] = useState<[number, number, number][]>([]);
  const roofPinsRef = useRef<[number, number, number][]>([]);
  const [sculptStrokeMode, setSculptStrokeMode] =
    useState<SculptStrokeModeApi>("draw");
  const [terrainSculptOp, setTerrainSculptOp] =
    useState<TerrainSculptOpApi>("raise");
  const [terrainBaseY, setTerrainBaseY] = useState(0);
  const [terrainStrength, setTerrainStrength] = useState(4);
  const [terrainSmoothRadius, setTerrainSmoothRadius] = useState(2);
  const [sculptSmoothPasses, setSculptSmoothPasses] = useState(1);
  /** Web `sculptBrushRadius` index (display = index + 1 voxel span). */
  const [sculptBrushRadius, setSculptBrushRadius] = useState(2);
  const [sculptBrushStrength, setSculptBrushStrength] = useState(100);
  const [sculptBrushFalloff, setSculptBrushFalloff] = useState(0);
  const [sculptBrushShapeUi, setSculptBrushShapeUi] =
    useState<SculptBrushShapeUi>("circle");
  // Extrude-specific params
  const [extrudeDirectionRef, setExtrudeDirectionRef] = useState<
    "camera" | "auto" | "x" | "y" | "z"
  >("camera");
  const extrudeDirectionRefRef = useRef<"camera" | "auto" | "x" | "y" | "z">("camera");
  extrudeDirectionRefRef.current = extrudeDirectionRef;
  const [extrudeProfile, setExtrudeProfile] = useState<"cube" | "cylinder">(
    "cube",
  );
  const [extrudeEndCap, setExtrudeEndCap] = useState<
    "flat" | "rounded" | "pointed"
  >("flat");
  const [extrudeTaper, setExtrudeTaper] = useState(false);
  const [extrudeTaperStart, setExtrudeTaperStart] = useState(3);
  const [extrudeTaperEnd, setExtrudeTaperEnd] = useState(0);
  const [wallAreaShape, setWallAreaShape] = useState<WallAreaShapeApi>("brush");
  const [sprayDirection, setSprayDirection] =
    useState<SprayDirectionApi>("auto");
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
  const [wallSculptPolygonVerts, setWallSculptPolygonVerts] = useState<
    [number, number, number][]
  >([]);
  const [pathLabel, setPathLabel] = useState("");
  /** Cold-start title mesh from `Logo.voxelle`; enables bottom menu layout and viewport orbit. */
  const [startScreenLogoLoaded, setStartScreenLogoLoaded] = useState(false);
  const startScreenLogoLoadedRef = useRef(false);
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
  const [fpsDisplayed, setFpsDisplayed] = useState(0);
  const [showFpsCounter, setShowFpsCounter] = useState(
    () => loadPreferences().showFpsCounter,
  );
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [newGridSize, setNewGridSize] = useState(32);
  const [newGridShape, setNewGridShape] = useState<StartShape>("circle");
  const [sidebarExpanded, setSidebarExpanded] = useState(() => {
    if (typeof localStorage === "undefined") return false;
    return localStorage.getItem(LS_SIDEBAR_EXPANDED) === "1";
  });
  const [rightSidebarExpanded, setRightSidebarExpanded] = useState(() => {
    if (typeof localStorage === "undefined") return false;
    return localStorage.getItem(LS_RIGHT_SIDEBAR_EXPANDED) === "1";
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
  const toolPaneDragRef = useRef<{
    pid: number;
    startX: number;
    startY: number;
    origX: number;
    origY: number;
  } | null>(null);
  const toolPanePosRef = useRef(toolPanePos);
  toolPanePosRef.current = toolPanePos;

  const [joinModalOpen, setJoinModalOpen] = useState(false);
  const [collabJoinPending, setCollabJoinPending] = useState(false);
  const [hostWsUrl, setHostWsUrl] = useState<string | null>(null);
  const [joinUrl, setJoinUrl] = useState(() => {
    const r = loadRecentJoinUrls();
    return r[0] ?? "ws://127.0.0.1:27300";
  });
  const [displayName, setDisplayName] = useState(
    () => loadPreferences().collabDisplayName,
  );
  const [accentColor, setAccentColor] = useState(
    () => loadPreferences().collabAccentColor,
  );
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
  } | null>(null);
  const [pingHudTick, setPingHudTick] = useState(0);
  const [pingLabelCss, setPingLabelCss] = useState<{
    name: string;
    leftPct: number;
    topPct: number;
  } | null>(null);
  const [peerLabels, setPeerLabels] = useState<
    { name: string; colorRgb: number; leftPct: number; topPct: number }[]
  >([]);
  const collabActiveRef = useRef(false);
  const localPeerIdRef = useRef(0);
  const [hostPort, setHostPort] = useState(
    () => loadPreferences().collabHostPort,
  );
  /** From preferences: UPnP when hosting (default off). */
  const [prefsEnableUpnp, setPrefsEnableUpnp] = useState(
    () => loadPreferences().enableUpnp,
  );
  /** Set when UPnP reports a public WebSocket URL (host only). */
  const [hostWanUrl, setHostWanUrl] = useState<string | null>(null);
  const [natPending, setNatPending] = useState(false);
  const [natError, setNatError] = useState<string | null>(null);
  const [lastSessionInfo, setLastSessionInfo] =
    useState<LastSessionInfo | null>(null);
  const [lastSessionReady, setLastSessionReady] = useState(false);
  const [collabActive, setCollabActive] = useState(false);
  /** Set when hosting or after welcome; 0 when solo. */
  const [localPeerId, setLocalPeerId] = useState(0);
  const [hostingCopied, setHostingCopied] = useState(false);
  const [sceneObjects, setSceneObjects] = useState<SceneObjectRow[]>([]);
  const [activeObjectId, setActiveObjectId] = useState(0);
  const [sceneObjectsErr, setSceneObjectsErr] = useState<string | null>(null);

  const refreshSceneObjects = useCallback(() => {
    void invoke<{ objects: SceneObjectRow[]; activeObjectId: number }>(
      "get_scene_objects",
    )
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
    const bootstrapW = Math.max(
      1,
      Math.round(bootstrapH * (layoutW / layoutH)),
    );
    // After a window resize, native size is unknown until the next frame — use bootstrap for configure + origin.
    const useNativeSurface = surf.w > 0 && surf.h > 0 && !innerChanged;
    const surfaceWidth = useNativeSurface ? surf.w : bootstrapW;
    const surfaceHeight = useNativeSurface ? surf.h : bootstrapH;

    const fullVp = fullSurfaceViewportRef.current;
    let viewportWidth: number;
    let viewportHeight: number;
    let viewportX: number;
    let viewportY: number;
    if (fullVp) {
      // Experimental: render 3D to the full drawable; pick using client/inner normalized coords.
      viewportX = 0;
      viewportY = 0;
      viewportWidth = surfaceWidth;
      viewportHeight = surfaceHeight;
    } else {
      // Derive viewport texture size from the same surface×layout fractions as viewportX/Y. Using
      // round(rh*dpr) here while origin uses (rect.top/ih)*surfaceHeight caused vertical drift when
      // surfaceHeight ≠ ih*dpr (native swapchain vs CSS estimate).
      viewportHeight = Math.max(1, Math.round((rh / layoutH) * surfaceHeight));
      viewportWidth = Math.max(1, Math.round(viewportHeight * (rw / rh)));
      // Proportional placement in the same pixel space as the swapchain (not raw rect×dpr alone).
      viewportX = Math.max(0, Math.round((rect.left / layoutW) * surfaceWidth));
      viewportY = Math.max(0, Math.round((rect.top / layoutH) * surfaceHeight));
    }
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
    fullSurfaceViewportRef.current = fullSurfaceViewportEnabled;
  }, [fullSurfaceViewportEnabled]);

  useEffect(() => {
    sendResize();
  }, [fullSurfaceViewportEnabled, sendResize]);

  useEffect(() => {
    try {
      void invoke("view_menu_sync_full_surface_viewport", {
        enabled: readFullSurfaceViewportEnabled(),
      }).catch(() => {});
    } catch {
      /* ignore */
    }
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
      }),
      listen<{ fraction: number; phase: string }>(
        "voxelle-load-progress",
        (e) => {
          const p = e.payload;
          setLoadProgress(p.fraction);
          setLoadPhase(p.phase);
          if (p.fraction >= 1) {
            setLoading(false);
            setLoadPhase("");
          }
        },
      ),
      listen<{ fraction: number; phase: string }>(
        "voxelle-work-progress",
        (e) => {
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
        },
      ),
      listen<unknown>("voxelle-loaded", (e) => {
        setLoadError(null);
        const p = e.payload;
        if (typeof p === "string") {
          setPathLabel(p);
          setStartScreenLogoLoaded(false);
        } else if (p && typeof p === "object" && "path" in p) {
          const o = p as {
            path: string;
            startScreenLogo?: boolean;
            mood?: Partial<MoodState>;
          };
          if (o.startScreenLogo) {
            setPathLabel("");
            setStartScreenLogoLoaded(true);
          } else {
            setPathLabel(o.path);
            setStartScreenLogoLoaded(false);
          }
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
      listen<string>("collab-ping", (e) => {
        try {
          const j = JSON.parse(e.payload) as {
            displayName?: string;
            display_name?: string;
            x?: number;
            y?: number;
            z?: number;
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
            return next.length > CHAT_TOAST_CAP
              ? next.slice(-CHAT_TOAST_CAP)
              : next;
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
            typeof j.displayName === "string" && j.displayName.length > 0
              ? j.displayName
              : "Guest";
          const text =
            j.reason === "left"
              ? `${name} left the session.`
              : `${name} disconnected.`;
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
          setNatError(
            typeof j.error === "string" && j.error.length > 0 ? j.error : null,
          );
          setHostWanUrl(
            typeof j.wanUrl === "string" && j.wanUrl.length > 0
              ? j.wanUrl
              : null,
          );
        } catch {
          setNatPending(false);
        }
      }),
      listen<string>("collab-ended", (e) => {
        const text =
          typeof e.payload === "string" ? e.payload : String(e.payload ?? "");
        if (text.trim().length > 0) {
          setCollabBanner({ text, tone: "info" });
        }
        clearCollabSessionUi();
      }),
      listen<string>("collab-kicked", (e) => {
        const msg =
          typeof e.payload === "string" ? e.payload : String(e.payload ?? "");
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
        if (
          m === "greedy" ||
          m === "marchingCubes" ||
          m === "dualContour" ||
          m === "ray"
        ) {
          localStorage.setItem(LS_RENDERING_MODE, m);
        }
      }),
      listen<string>("voxelle-menu-selection-mode", (e) => {
        const m = e.payload;
        if (
          m === "selectByColor" ||
          m === "selectCoplanar" ||
          m === "selectCoplanarEmpty"
        ) {
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
      listen<boolean>("voxelle-full-surface-viewport", (e) => {
        const enabled = e.payload;
        try {
          localStorage.setItem(LS_FULL_SURFACE_VIEWPORT, enabled ? "1" : "0");
        } catch {
          /* ignore */
        }
        fullSurfaceViewportRef.current = enabled;
        setFullSurfaceViewportEnabled(enabled);
      }),
      listen<number>("voxelle-selection-updated", (e) => {
        setSelectionCount(typeof e.payload === "number" ? e.payload : 0);
      }),
      listen<string>("voxelle-selection-combine-mode", (e) => {
        const p = e.payload;
        if (
          p === "replace" ||
          p === "add" ||
          p === "subtract" ||
          p === "intersect"
        ) {
          setSelectionCombineMode(p);
        }
      }),
      listen<string>("voxelle-menu-not-implemented", (e) => {
        const msg =
          typeof e.payload === "string" ? e.payload : String(e.payload ?? "");
        console.warn(msg);
      }),
      listen("voxelle-open-project-stats", () => {
        window.alert(
          "Project stats: use the menu Debug → Copy performance info, then paste into a note or issue.",
        );
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

  useEffect(() => {
    if (pingHudTick === 0 && !pingHudRef.current) return;
    let cancelled = false;
    let raf = 0;
    const tick = () => {
      if (cancelled) return;
      const p = pingHudRef.current;
      if (!p || Date.now() > p.until) {
        setPingLabelCss(null);
        if (p && Date.now() > p.until) pingHudRef.current = null;
        return;
      }
      void invoke<[number, number] | null>("world_to_viewport_pixels", {
        args: { x: p.wx, y: p.wy, z: p.wz },
      })
        .then((opt) => {
          if (cancelled || opt == null || !pingHudRef.current) return;
          const [sx, sy] = opt;
          const { w, h } = viewportPhysRef.current;
          if (w <= 0 || h <= 0) return;
          const cur = pingHudRef.current;
          if (!cur) return;
          setPingLabelCss({
            name: cur.name,
            leftPct: (sx / w) * 100,
            topPct: (sy / h) * 100,
          });
        })
        .catch(() => {});
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [pingHudTick]);

  // Poll peer label screen positions while collab is active
  useEffect(() => {
    if (!collabActive) {
      setPeerLabels([]);
      return;
    }
    let cancelled = false;
    let raf = 0;
    const tick = () => {
      if (cancelled) return;
      void invoke<
        { name: string; colorRgb: number; leftPct: number; topPct: number }[]
      >("collab_peer_labels")
        .then((labels) => {
          if (!cancelled) setPeerLabels(labels);
        })
        .catch(() => {});
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [collabActive]);

  useEffect(() => {
    interactionModeRef.current = interactionMode;
  }, [interactionMode]);

  const prevInteractionModeForEyedropperRef =
    useRef<InteractionMode>(interactionMode);
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

  useEffect(() => {
    activeColorRef.current = activeColor;
  }, [activeColor]);
  useEffect(() => {
    activeMaterialRef.current = activeMaterial;
  }, [activeMaterial]);
  useEffect(() => {
    brushRadiusRef.current = brushRadius;
  }, [brushRadius]);
  useEffect(() => {
    brushShapeRef.current = brushShape;
  }, [brushShape]);
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
  const grassDensityRef = useRef(grassDensity);
  const grassMaxHeightRef = useRef(grassMaxHeight);
  const rockPreviewSeedRef = useRef((Math.random() * 1e9) | 0);
  const grassPreviewSeedRef = useRef((Math.random() * 1e9) | 0);
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
    grassDensityRef.current = grassDensity;
  }, [grassDensity]);
  useEffect(() => {
    grassMaxHeightRef.current = grassMaxHeight;
  }, [grassMaxHeight]);
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

  function mergedStrokeAux(
    base: Record<string, unknown> = {},
  ): Record<string, unknown> {
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
      strokeFamilyVariant: strokeFamilyVariantRef.current,
      strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
      strokeAxisAlign: selectionStrokeAxisAlignRef.current,
      brushClipBottomHalf: brushClipBottomHalfRef.current,
      ...(polygonVertices != null && polygonVertices.length > 0
        ? { polygonVertices }
        : {}),
    };
    if (sm === "cuboid" && cuboidPhase.ref.current) {
      out.cuboidDepth = cuboidDepthRef.current;
      out.cuboidHollowWallThickness = 1;
    }
    if (sm === "cylinder" && cylinderPhase.ref.current) {
      out.cylinderDepth = cylinderDepthRef.current;
      out.cylinderTaperPct = 0;
      out.cuboidHollowWallThickness = 1;
    }
    return out;
  }

  useEffect(() => {
    cuboidDepthRef.current = cuboidDepthUi;
  }, [cuboidDepthUi]);

  useEffect(() => {
    cylinderDepthRef.current = cylinderDepthUi;
  }, [cylinderDepthUi]);

  // Cuboid depth preview: re-invoke when phase data or depth changes.
  useEffect(() => {
    const snap = cuboidPhase.snapshot;
    if (!snap || loading || workBusy) return;
    const im = interactionModeRef.current;
    if (im !== "add" && im !== "remove" && im !== "paint") return;
    const { lineStart, endNorm } = snap.data;
    const tool = im === "add" ? "add" : im === "remove" ? "remove" : "paint";
    void invoke("voxel_stroke_preview_at_screen", {
      args: {
        nx: endNorm.nx,
        ny: endNorm.ny,
        tool,
        color: activeColorRef.current,
        material: activeMaterialRef.current,
        brushRadius: brushRadiusRef.current,
        brushShape: brushShapeRef.current,
        sprayDensity: sprayDensityRef.current,
        strokeMode: "cuboid",
        planeAxis: planeAxisRef.current,
        strokeAux: mergedStrokeAux({}),
        matchMaterial: matchMaterialSelectColorRef.current,
        strokeLineStartNx: lineStart.nx,
        strokeLineStartNy: lineStart.ny,
      },
    }).catch(() => {});
  }, [cuboidPhase.snapshot, cuboidDepthUi, loading, workBusy, interactionMode]);

  // Cylinder depth preview: re-invoke when phase data or depth changes.
  useEffect(() => {
    const snap = cylinderPhase.snapshot;
    if (!snap || loading || workBusy) return;
    const im = interactionModeRef.current;
    if (im !== "add" && im !== "remove" && im !== "paint") return;
    const { lineStart, endNorm } = snap.data;
    const tool = im === "add" ? "add" : im === "remove" ? "remove" : "paint";
    void invoke("voxel_stroke_preview_at_screen", {
      args: {
        nx: endNorm.nx,
        ny: endNorm.ny,
        tool,
        color: activeColorRef.current,
        material: activeMaterialRef.current,
        brushRadius: brushRadiusRef.current,
        brushShape: brushShapeRef.current,
        sprayDensity: sprayDensityRef.current,
        strokeMode: "cylinder",
        planeAxis: planeAxisRef.current,
        strokeAux: mergedStrokeAux({}),
        matchMaterial: matchMaterialSelectColorRef.current,
        strokeLineStartNx: lineStart.nx,
        strokeLineStartNy: lineStart.ny,
      },
    }).catch(() => {});
  }, [
    cylinderPhase.snapshot,
    cylinderDepthUi,
    loading,
    workBusy,
    interactionMode,
  ]);

  // Cancel extrude phase if interaction mode changes
  useEffect(() => {
    if (extrudePhase.active && interactionMode !== "sculpt") {
      extrudePhase.cancel();
    }
  }, [extrudePhase.active, interactionMode]);

  // Live-update extrude preview when settings change during the phase.
  useEffect(() => {
    if (!extrudePhase.active) return;
    void invoke("extrude_recompute_preview", {
      args: {
        extrudeProfile: extrudeProfileRef.current,
        extrudeEndCap: extrudeEndCapRef.current,
        extrudeTaper: extrudeTaperRef.current,
        extrudeTaperStart: extrudeTaperRef.current
          ? extrudeTaperStartRef.current
          : 0,
        extrudeTaperEnd: extrudeTaperRef.current
          ? extrudeTaperEndRef.current
          : 0,
      },
    }).catch(() => {});
  }, [
    extrudePhase.active,
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
    runStrokeAtScreen(endNorm.nx, endNorm.ny, {}, { lineStart });
    phase.cancel();
  }

  function commitCuboidSolidAtScreen() {
    commitDepthPhaseAtScreen("cuboid");
  }

  function commitCylinderSolidAtScreen() {
    commitDepthPhaseAtScreen("cylinder");
  }

  /** Payload for `sync_preview_input` — must match Rust `SyncPreviewInput` (camelCase). */
  function buildSyncPreviewPayload(_nx: number, _ny: number, modeStr: string) {
    // During rope settings phase, lock the hover position to the stored
    // second endpoint so the rope preview stays fixed while adjusting params.
    const rSnap = ropePhase.ref.current;
    const nx = rSnap ? rSnap.data.nx2 : _nx;
    const ny = rSnap ? rSnap.data.ny2 : _ny;
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
      material: activeMaterialRef.current,
      matchMaterial: matchMaterialSelectColorRef.current,
      useBrushPreview: im !== "squishy" && !isGenerator,
      ...(isGenerator
        ? {
            generatorKind: gk,
            generatorRopeFirstNx: ropeFirstScreenRef.current?.nx,
            generatorRopeFirstNy: ropeFirstScreenRef.current?.ny,
            generatorRopeSag: ropeSagRef.current,
            generatorRopeTension: ropeTensionRef.current,
            generatorClothPins: clothPinsRef.current.map((p) => [
              p[0],
              p[1],
              p[2],
            ]),
            generatorClothTension: clothTensionRef.current,
            generatorClothGravityDirection: clothGravityDirectionRef.current,
            generatorClothGravityScale: clothSimGravityPctRef.current / 100,
            generatorClothStiffnessScale: clothSimStiffnessPctRef.current / 100,
            generatorClothIterations: clothSimIterationsRef.current,
            generatorClothConstraintPasses: clothSimConstraintPassesRef.current,
            generatorRockSize: Math.max(1, generatorSphereRadiusRef.current),
            generatorRockRoughness: rockRoughnessRef.current,
            generatorRockSeed: rockPreviewSeedRef.current,
            generatorGrassDensity: grassDensityRef.current,
            generatorGrassMaxHeight: grassMaxHeightRef.current,
            generatorGrassSeed: grassPreviewSeedRef.current,
          }
        : {}),
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
      return invoke("voxel_edit_at_screen", {
        args: {
          ...sharedArgs,
          tool: dispatch.tool,
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      })
        .catch((e: unknown) => {
          console.error("[voxelle] voxel_edit_at_screen error", e);
        })
        .finally(() => {
          if (isFill) endFillOperation();
        }) as Promise<void>;
    } else {
      return invoke<number>("selection_stroke_at_screen", {
        args: {
          ...sharedArgs,
          interaction: dispatch.interaction,
        },
      })
        .then((n) => {
          if (n > 0) {
            void invoke<number>("selection_get_count").then((c) =>
              setSelectionCount(c),
            );
          }
        })
        .catch(() => {})
        .finally(() => {
          if (isFill) endFillOperation();
        });
    }
  }

  /** Specialized selection single-click commands (selectByColor, selectCoplanar, selectCoplanarEmpty). */
  function invokeSelectionSpecialClick(
    interaction: string,
    nx: number,
    ny: number,
  ) {
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
    void invoke<number>(cmd, { args })
      .then((n) => {
        if (n > 0) {
          void invoke<number>("selection_get_count").then((c) =>
            setSelectionCount(c),
          );
        }
      })
      .catch(() => {});
  }

  async function handleStrokeAnchorClick(nx: number, ny: number) {
    const dispatch = getStrokeDispatch(interactionModeRef.current);
    if (!dispatch) return;
    const tool = dispatch.kind === "edit" ? dispatch.tool : "remove";
    const c = await invoke<[number, number, number] | null>(
      "voxel_stroke_anchor_coord_at_screen",
      {
        args: {
          nx,
          ny,
          tool,
          strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
        },
      },
    );
    if (!c) return;
    const sm = drawStrokeModeRef.current;
    if (sm === "fill") {
      runStrokeAtScreen(nx, ny, {});
      return;
    }
    if (sm === "polygon" || sm === "polygonHull") {
      setStrokePolygonVerts((v) => {
        const idx = v.findIndex(
          (p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2],
        );
        const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
        strokePolygonVertsRef.current = next;
        return next;
      });
      strokePolygonLastScreenRef.current = { nx, ny };
      queueMicrotask(() => {
        void invoke("sync_preview_input", {
          args: buildSyncPreviewPayload(
            nx,
            ny,
            previewModeForSync(interactionModeRef.current),
          ),
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
    const c = await invoke<[number, number, number] | null>(
      "voxel_stroke_anchor_coord_at_screen",
      {
        args: {
          nx,
          ny,
          tool: "add",
          strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
        },
      },
    );
    if (!c) return;
    setWallSculptPolygonVerts((v) => {
      const idx = v.findIndex(
        (p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2],
      );
      const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
      wallSculptPolygonVertsRef.current = next;
      return next;
    });
  }

  async function handleClothPinClick(nx: number, ny: number) {
    const c = await invoke<[number, number, number] | null>(
      "voxel_stroke_anchor_coord_at_screen",
      {
        args: {
          nx,
          ny,
          tool: "add",
          strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
        },
      },
    );
    if (!c) return;
    setClothPins((v) => {
      const idx = v.findIndex(
        (p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2],
      );
      const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
      clothPinsRef.current = next;
      return next;
    });
  }


  function applyPolygonStrokeFill() {
    if (strokePolygonVerts.length < 3) return;
    const scr =
      strokePolygonLastScreenRef.current ?? lastViewportPickNormRef.current;
    const nx = scr?.nx ?? 0;
    const ny = scr?.ny ?? 0;
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
    fillConstrainToPlaneRef.current = fillConstrainToPlane;
  }, [fillConstrainToPlane]);
  useEffect(() => {
    if (interactionMode === "fly") {
      setToolsPane("fly");
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
      (interactionMode === "stamp" || interactionMode === "punch")
    ) {
      setInteractionMode("add");
    }
  }, [selectionCount, interactionMode]);

  const previewModeForSync = (m: InteractionMode): string => {
    if (m === "add") return "add";
    if (m === "remove") return "remove";
    if (m === "paint") return "paint";
    if (m === "sculpt") return "add";
    if (m === "generator") return "add";
    if (m === "fly") return "fly";
    if (m === "squishy") return "squishy";
    if (
      m === "select" ||
      m === "selectByColor" ||
      m === "selectCoplanar" ||
      m === "selectCoplanarEmpty"
    ) {
      return "select";
    }
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
    if (
      strokeDrawStyleRef.current === "brush" &&
      drawStrokeModeRef.current !== "spray"
    ) {
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
      if (
        !fillOperationPendingRef.current &&
        !/fill/i.test(workPhaseRef.current)
      )
        return;
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
    localStorage.setItem(
      LS_RIGHT_SIDEBAR_EXPANDED,
      rightSidebarExpanded ? "1" : "0",
    );
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
    const p = loadPreferences();
    void invoke("set_autosave_settings", autosaveSettingsInvokeArgs(p)).catch(
      () => {},
    );
  }, []);

  useEffect(() => {
    void invoke<LastSessionInfo>("get_last_session_info")
      .then((info) => setLastSessionInfo(info))
      .catch(() => setLastSessionInfo(null))
      .finally(() => setLastSessionReady(true));
  }, []);

  useEffect(() => {
    if (!lastSessionReady || !lastSessionInfo?.lastDocumentPath) return;
    const p = loadPreferences();
    if (!p.reopenLastProject) return;
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
      });
    } else if (info.documentExists) {
      void invoke("load_voxelle_path", { path: doc }).catch((err) => {
        setLoadError(err instanceof Error ? err.message : String(err));
      });
    }
  }, [lastSessionReady, lastSessionInfo]);

  useEffect(() => {
    const p = loadPreferences();
    void invoke("set_tone_mapping", {
      mode: toneMappingToGpuMode(p.toneMapping),
    }).catch(() => {});
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(
      LS_RENDERING_MODE,
    ) as RenderingMode | null;
    const valid =
      saved &&
      ["greedy", "marchingCubes", "dualContour", "ray"].includes(saved);
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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
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
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      ) {
        return;
      }
      if (
        preferencesOpen ||
        joinModalOpen ||
        newProjectOpen ||
        collabJoinPending
      )
        return;
      const p = lastViewportPickNormRef.current;
      if (!p) return;
      e.preventDefault();
      const dn = loadPreferences().collabDisplayName.trim();
      void invoke<{
        ok: boolean;
        x?: number;
        y?: number;
        z?: number;
      }>("ping_cursor_pick", {
        args: { nx: p.nx, ny: p.ny, displayName: dn },
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
          };
          setPingHudTick((n) => n + 1);
          playPingSound();
          void invoke("collab_send_ping", {
            x: r.x,
            y: r.y,
            z: r.z,
          }).catch(() => {});
        })
        .catch(() => {});
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [preferencesOpen, joinModalOpen, newProjectOpen, collabJoinPending]);

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
        joinModalOpen ||
        newProjectOpen ||
        collabJoinPending
      ) {
        return;
      }
      if (loading || workBusy) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.code !== "KeyX" || selectionCount === 0) return;
      e.preventDefault();
      e.stopPropagation();
      void invoke<number>("selection_delete_selected_voxels").catch(() => {});
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    selectionCount,
    preferencesOpen,
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
    const vp = viewportRef.current;
    const pid = flyCapturedPointerIdRef.current;
    flyCapturedPointerIdRef.current = null;
    if (vp != null && pid != null) {
      try {
        vp.releasePointerCapture(pid);
      } catch {
        /* not capturing or pointer already gone */
      }
    }
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

  const activateFlyMouseLook = useCallback(async (pointerId: number) => {
    const el = viewportRef.current;
    if (!el) return;
    flySkipNextFlyMoveRef.current = false;
    flyPendingLookDxRef.current = 0;
    flyPendingLookDyRef.current = 0;
    // Retargets this pointer to the viewport until pointerup (capture does not persist across the full click).
    try {
      el.setPointerCapture(pointerId);
      flyCapturedPointerIdRef.current = pointerId;
    } catch {
      flyCapturedPointerIdRef.current = null;
    }
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
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
    grassDensity,
    grassMaxHeight,
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
      if (flySkipNextFlyMoveRef.current) {
        flySkipNextFlyMoveRef.current = false;
        flyLastClientRef.current = { x: e.clientX, y: e.clientY };
        return;
      }
      // True FPS deltas: prefer movementX/Y. Do NOT use (client - viewportCenter);
      // that behaves like a joystick and keeps turning while the cursor stays off-center.
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
      const speedScale = slow ? 1 / 8 : 1;
      void invoke("sync_fly_input", {
        args: { forward, right, up, speedScale },
      }).catch(() => {});
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
    if (fullSurfaceViewportRef.current) {
      const { w: lw, h: lh } = layoutViewportCssSize();
      return {
        nx: Math.min(1, Math.max(0, e.clientX / lw)),
        ny: Math.min(1, Math.max(0, e.clientY / lh)),
      };
    }
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
    (phase: string, e: React.PointerEvent, extra?: Record<string, unknown>) => {
      if (!planeStrokeDebugEnabledRef.current) return;
      const mode = interactionModeRef.current;
      const sm = drawStrokeModeRef.current;
      if (
        !(
          sm === "plane" &&
          (mode === "add" || mode === "remove" || mode === "paint")
        )
      ) {
        return;
      }
      const g = gestureRef.current;
      console.debug("[voxelle][plane-stroke]", {
        phase,
        eventType: e.type,
        pointerId: e.pointerId,
        button: e.button,
        buttons: e.buttons,
        mode,
        strokeMode: sm,
        gestureMode: g?.mode ?? null,
        gesturePointerId: g?.pointerId ?? null,
        activePointerId: activePointerIdRef.current,
        capturedPointerId: capturedPointerIdRef.current,
        probing: probingRef.current,
        dragDidEdit: dragDidEditRef.current,
        movedPx: maxPointerMoveRef.current,
        ...extra,
      });
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
    },
    [logPlaneStrokeDebug],
  );

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
    if (modeEarly === "fly" && (e.button === 0 || e.button === 2)) {
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
    // Cancel extrude phase on any other click.
    if (extrudePhase.ref.current) {
      extrudePhase.cancel();
    }

    const { nx, ny } = clientToViewportNormalized(e);
    const pointerId = e.pointerId;
    const middleButton = e.button === 1;
    const mode = interactionModeRef.current;
    const navigate = mode === "navigate" || mode === "fly";
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
      (mode === "stamp" && e.button !== 0) ||
      (mode === "punch" && e.button !== 0) ||
      (mode === "sculpt" && e.button !== 0) ||
      (mode === "generator" && e.button !== 0) ||
      (mode === "squishy" && e.button !== 0);

    const logoSplashPointer =
      startScreenLogoLoadedRef.current &&
      !loading &&
      !workBusy &&
      e.button === 0;

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
    // Fill and selection modes skip the async pick probe — they commit on
    // pointer-up via a screen coordinate alone, so the Rust side handles
    // misses gracefully.  Skipping the probe avoids a race where onPointerUp
    // fires while the probe is still in-flight, causing the click to be
    // silently discarded (the "click twice to fill/select" bug).
    const skipProbe =
      isDrawOrSelect &&
      (drawStrokeModeRef.current === "fill" ||
        getStrokeDispatch(mode)?.kind === "selection");
    if (isDrawOrSelect && !skipProbe) {
      try {
        hitSolid = await invoke<boolean>("voxel_pick_probe", {
          args: { nx, ny },
        });
      } catch {
        hitSolid = false;
      }
    }
    if (skipProbe) {
      hitSolid = true;
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
        if (
          drawStrokeModeRef.current === "cylinder" &&
          cylinderPhase.ref.current
        ) {
          cylinderPhase.cancel();
        }
        dragDidEditRef.current = false;
        strokeViewportStartRef.current = { nx, ny };
        lastStrokeNormRef.current = { nx, ny };
        if (dispatch?.kind === "selection") {
          selectionStrokeBegunRef.current = true;
          void invoke("selection_stroke_begin").catch(() => {});
        } else {
          void invoke("voxel_stroke_begin").catch(() => {});
        }
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
      wallAreaShapeRef.current === "circle" &&
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
            wallPolygonVertices: wallSculptPolygonVertsRef.current.map((v) => [
              v[0],
              v[1],
              v[2],
            ]),
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
            terrainStrength: terrainStrengthRef.current,
            terrainSmoothRadius: terrainSmoothRadiusRef.current,
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
      extrudeTaperStart: extrudeTaperRef.current
        ? extrudeTaperStartRef.current
        : 0,
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
              fetch(
                "http://127.0.0.1:7756/ingest/93734617-b27b-4379-bb59-e5971936c3d4",
                {
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
                    message:
                      "pick uses rel/rect; compare inner*surface−origin Y vs relY/rh",
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
                        const ox = Math.max(
                          0,
                          Math.round((rV.left / iw) * surf.w),
                        );
                        const oy = Math.max(
                          0,
                          Math.round((rV.top / ih) * surf.h),
                        );
                        nxWindow = ((scr.clientX / iw) * surf.w - ox) / phys.w;
                        deltaWinVsRel = nxWindow - nxFromRel;
                        nyFromInner =
                          ((scr.clientY / ih) * surf.h - oy) / phys.h;
                        deltaInnerVsRelNy = nyFromInner - nyFromRel;
                      }
                      return {
                        viewportRw: rV?.width,
                        viewportRh: rV?.height,
                        wrapRw: rW?.width,
                        wrapRh: rW?.height,
                        rectDeltaW: rV && rW ? rV.width - rW.width : null,
                        rectDeltaH: rV && rW ? rV.height - rW.height : null,
                        aspectDom:
                          rV && rV.height > 0 ? rV.width / rV.height : null,
                        physW: phys.w,
                        physH: phys.h,
                        aspectPhys: phys.h > 0 ? phys.w / phys.h : null,
                        rustW: d.viewportWidth,
                        rustH: d.viewportHeight,
                        aspectRust:
                          d.viewportHeight > 0
                            ? d.viewportWidth / d.viewportHeight
                            : null,
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
                        deltaPickVsRelNx:
                          pick && nxFromRel != null
                            ? pick.nx - nxFromRel
                            : null,
                        deltaPickVsRelNy:
                          pick && nyFromRel != null
                            ? pick.ny - nyFromRel
                            : null,
                      };
                    })(),
                    timestamp: Date.now(),
                  }),
                },
              ).catch(() => {});
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
        interactionModeRef.current === "generator") &&
      !interactionBlockedRef.current
    ) {
      const m = previewModeForSync(interactionModeRef.current);
      void invoke("sync_preview_input", {
        args: buildSyncPreviewPayload(px, py, m),
      }).catch(() => {});
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
              extrudeTaperStart: extrudeTaperRef.current
                ? extrudeTaperStartRef.current
                : 0,
              extrudeTaperEnd: extrudeTaperRef.current
                ? extrudeTaperEndRef.current
                : 0,
            },
          }).catch((err) => {
            console.error("[extrude_ray_preview re-drag]", err);
          });
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
        maxPointerMoveRef.current = Math.max(
          maxPointerMoveRef.current,
          Math.hypot(dx, dy),
        );
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
          !(
            drawStrokeModeRef.current === "cuboid" && cuboidPhase.ref.current
          ) &&
          !(
            drawStrokeModeRef.current === "cylinder" &&
            cylinderPhase.ref.current
          )
        ) {
          const now = Date.now();
          if (now - lastStrokeEditMsRef.current >= 24) {
            lastStrokeEditMsRef.current = now;
            dragDidEditRef.current = true;
            const lineStart = strokeViewportLineStartNorm();
            const brushPrev =
              strokeDrawStyleRef.current === "brush" &&
              lastStrokeNormRef.current
                ? lastStrokeNormRef.current
                : null;
            if (dispatch.kind === "edit") {
              void invoke("voxel_stroke_preview_at_screen", {
                args: {
                  nx: px,
                  ny: py,
                  tool: dispatch.tool,
                  color: activeColorRef.current,
                  material: activeMaterialRef.current,
                  brushRadius: brushRadiusRef.current,
                  brushShape: brushShapeRef.current,
                  sprayDensity: sprayDensityRef.current,
                  strokeMode: drawStrokeModeRef.current,
                  planeAxis: planeAxisRef.current,
                  strokeAux: mergedStrokeAux({}),
                  matchMaterial: matchMaterialSelectColorRef.current,
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
      if (
        e.buttons &&
        m === "sculpt" &&
        !loading &&
        !workBusy &&
        !fillOperationPending
      ) {
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
            const startNorm =
              extrudeStartNormRef.current ?? strokeViewportStartRef.current;
            if (startNorm) {
              // On first drag, persist the start position for later re-drags.
              if (!extrudeStartNormRef.current && strokeViewportStartRef.current) {
                extrudeStartNormRef.current = { ...strokeViewportStartRef.current };
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
                  extrudeTaperStart: extrudeTaperRef.current
                    ? extrudeTaperStartRef.current
                    : 0,
                  extrudeTaperEnd: extrudeTaperRef.current
                    ? extrudeTaperEndRef.current
                    : 0,
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
      return;
    }
    if (pointerStartRef.current) {
      const dx = e.clientX - pointerStartRef.current.x;
      const dy = e.clientY - pointerStartRef.current.y;
      maxPointerMoveRef.current = Math.max(
        maxPointerMoveRef.current,
        Math.hypot(dx, dy),
      );
    }
    const dpr = window.devicePixelRatio || 1;
    const dx = (e.clientX - lastRef.current.x) * dpr;
    const dy = (e.clientY - lastRef.current.y) * dpr;
    lastRef.current = { x: e.clientX, y: e.clientY };
    if (e.buttons === 0) {
      logPlaneStrokeDebug("move:buttons-zero", e);
      if (
        startScreenLogoLoadedRef.current &&
        interactionModeRef.current !== "fly"
      ) {
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
    if (interactionModeRef.current !== "fly") {
      void invoke("viewport_pointer", {
        ev: {
          kind: "move",
          nx,
          ny,
          dx,
          dy,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey,
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
      probingRef.current = false;
      resetPointerGesture("probe-up", e);
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

    const g = gestureRef.current;
    const start = pointerStartRef.current;
    const moved = maxPointerMoveRef.current;
    const isThisPointer = g?.pointerId === e.pointerId;
    let hasPointerCaptureForUp = capturedPointerIdRef.current === e.pointerId;
    if (!hasPointerCaptureForUp && viewportRef.current) {
      try {
        hasPointerCaptureForUp = viewportRef.current.hasPointerCapture(
          e.pointerId,
        );
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
              color: activeColorRef.current,
              material: activeMaterialRef.current,
            },
          }).catch(() => {});
        } else if (m === "punch") {
          void invoke("clipboard_punch_at_screen", { args: { nx, ny } }).catch(
            () => {},
          );
        } else if (m === "generator") {
          const gk = generatorKindRef.current;
          if (gk === "rocks") {
            void invoke("generator_rocks_at_screen", {
              args: {
                nx,
                ny,
                seed: (Math.random() * 1e9) | 0,
                size: Math.max(1, generatorSphereRadiusRef.current),
                roughness: rockRoughness,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => {});
          } else if (gk === "grass") {
            void invoke("generator_grass_at_screen", {
              args: {
                nx,
                ny,
                seed: (Math.random() * 1e9) | 0,
                density: grassDensity,
                maxHeight: grassMaxHeight,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => {});
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
              ropePhase.enter("settings", {
                nx1: ropeFirstScreen.nx,
                ny1: ropeFirstScreen.ny,
                nx2: nx,
                ny2: ny,
              });
            }
          } else if (gk === "ashlar") {
            void invoke("generator_ashlar_at_screen", {
              args: {
                nx,
                ny,
                seed: (Math.random() * 1e9) | 0,
                size: Math.max(1, generatorSphereRadiusRef.current),
                roughness: rockRoughness,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => {});
          } else if (gk === "flora") {
            void invoke("generator_flora_at_screen", {
              args: {
                nx,
                ny,
                seed: (Math.random() * 1e9) | 0,
                height: floraHeight,
                girth: floraGirth,
                wobble: 0.12,
                taper: 0.12,
                stemCount: floraStemCount,
                clusterRadius: 0,
                branchCount: floraBranchCount,
                branchDepth: 1,
                branchStart: 0.5,
                branchSpread: 1.0,
                braidStrands: 1,
                braidTwist: 0.35,
                canopy: floraCanopy,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => {});
          } else if (gk === "roof") {
            // Roof uses pin-based input like cloth
            void invoke<[number, number, number] | null>(
              "voxel_stroke_anchor_coord_at_screen",
              {
                args: {
                  nx,
                  ny,
                  tool: "add",
                  strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
                },
              },
            )
              .then((c) => {
                if (!c) return;
                setRoofPins((v) => {
                  const next = [...v, c];
                  roofPinsRef.current = next;
                  return next;
                });
              })
              .catch(() => {});
          } else if (gk === "piscina") {
            void invoke("generator_piscina_at_screen", {
              args: {
                nx,
                ny,
                seed: (Math.random() * 1e9) | 0,
                species: piscinaSpecies,
                length: piscinaLength,
                widthParam: 4,
                thickness: 3,
                spineBend: 0,
                spineSCurve: 0,
                finDorsal: 3,
                finAnal: 3,
                finCaudal: 3,
                finPectoral: 3,
                finPelvic: 3,
                finAdipose: 3,
                showFinDorsal: true,
                showFinAnal: true,
                showFinCaudal: true,
                showFinPectoral: true,
                showFinPelvic: true,
                showFinAdipose: true,
                anchorOffsetU: 0,
                anchorOffsetV: 0,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => {});
          } else if (gk === "insecta") {
            void invoke("generator_insecta_at_screen", {
              args: {
                nx,
                ny,
                species: insectaSpecies,
                totalLength: insectaTotalLength,
                headRatio: 1.0,
                thoraxRatio: 1.2,
                abdomenRatio: 2.0,
                bodyHalfWidth: 3,
                bodyHalfHeight: 3,
                abdomenTaper: 0.6,
                headShape: 60,
                anchorOffsetU: 0,
                anchorOffsetV: 0,
                bodyYaw: 0,
                bodyArch: 0,
                antennaLength: 6,
                antennaSpread: 20,
                antennaPitch: 30,
                antennaRoot: 0,
                mandibleLength: 0,
                mandibleSpread: 0,
                mandibleForward: 0,
                wingShape: 85,
                showWingFore: true,
                wingForeLength: 12,
                wingForeWidth: 3,
                wingForeSpread: 15,
                wingForePitch: 0,
                wingForeOffset: 0,
                wingForeForwardCant: 0,
                showWingHind: false,
                wingHindLength: 8,
                wingHindWidth: 2,
                wingHindSpread: 15,
                wingHindPitch: 0,
                wingHindOffset: 0,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => {});
          } else if (gk === "fauna") {
            void invoke("generator_fauna_at_screen", {
              args: {
                nx,
                ny,
                stance: faunaStance,
                archetype: faunaArchetype,
                anchorOffsetU: 0,
                anchorOffsetV: 0,
                bodyYaw: 0,
                bodyArch: 0.02,
                spineSegments: 7,
                bodyLength: 17,
                bodyHalfWidth: 2,
                bodyHalfHeight: 3,
                neckLength: 8,
                neckHalfWidth: 2,
                neckHalfHeight: 3,
                headLength: 6,
                headHalfWidth: 2,
                headHalfHeight: 3,
                tailLength: 1,
                shoulderOffsetForward: 3,
                hipOffsetForward: -3,
                frontUpperLength: 7,
                frontLowerLength: 7,
                hindUpperLength: 8,
                hindLowerLength: 8,
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
                autoFootPlacement: false,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => {});
          }
        } else if (m === "squishy") {
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
            .then(() =>
              invoke<{ balls: { id: number }[] }>("squishy_session_get"),
            )
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
          const setDepthUi =
            sm === "cuboid" ? setCuboidDepthUi : setCylinderDepthUi;
          depthRef.current = 1;
          setDepthUi(1);
          phase.enter("depth", {
            lineStart: { ...strokeViewportStartRef.current },
            endNorm: { nx, ny },
          });
        }
        // Single-click handling (no drag)
        // For selection strokes, we must wait for the stroke invoke to
        // complete before calling selection_stroke_end, otherwise the
        // end command can race ahead and clear the accumulator.
        let clickStrokePromise: Promise<void> | null = null;
        if (!dragDidEditRef.current && moved < 5) {
          if (
            (sm === "cuboid" || sm === "cylinder") &&
            strokeViewportStartRef.current
          ) {
            // Single-click cuboid/cylinder: enter depth phase
            const phase = sm === "cuboid" ? cuboidPhase : cylinderPhase;
            const depthRef =
              sm === "cuboid" ? cuboidDepthRef : cylinderDepthRef;
            const setDepthUi =
              sm === "cuboid" ? setCuboidDepthUi : setCylinderDepthUi;
            depthRef.current = 1;
            setDepthUi(1);
            phase.enter("depth", {
              lineStart: { ...strokeViewportStartRef.current },
              endNorm: { nx, ny },
            });
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
      }
    } else if (
      isThisPointer &&
      g?.mode === "voxel" &&
      e.button === 0 &&
      !hasPointerCaptureForUp
    ) {
      logPlaneStrokeDebug("up:ignored-no-capture", e, {
        moved,
      });
    }

    if (
      isThisPointer &&
      g?.mode === "camera" &&
      interactionModeRef.current !== "fly"
    ) {
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

  const onPointerLeave = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("leave", e);
    // Keep the select preview visible when the pointer moves to the sidebar / tool panel.
    const im = interactionModeRef.current;
    if (
      im !== "select" &&
      im !== "selectByColor" &&
      im !== "selectCoplanar" &&
      im !== "selectCoplanarEmpty"
    ) {
      clearPreview();
    }
    if (viewportCursorDebugEnabled) {
      setViewportCursorDebugJs(null);
      setViewportCursorDebugRust(null);
      viewportCursorDebugScreenRef.current = null;
      setViewportCursorDebugScreen(null);
    }
    if (
      startScreenLogoLoadedRef.current &&
      interactionModeRef.current !== "fly"
    ) {
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
    if (interactionModeRef.current === "fly") return;
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
  const showEmptyOpenFile = showStartScreen;
  /** Cold start: logo mesh still decoding (ignore `showStartScreen` while `workBusy` toggles editor chrome). */
  const showStartScreenLogoSpinner =
    !startScreenLogoLoaded && !pathLabel && !collabActive;

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

  const lastProjectBlurb =
    lastSessionInfo != null ? lastProjectReopenBlurb(lastSessionInfo) : null;

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
    void invoke("collab_set_can_edit", { targetPeer: peerId, canEdit }).catch(
      () => {},
    );
  };

  const isSelectionInteractionMode =
    interactionMode === "select" ||
    interactionMode === "selectByColor" ||
    interactionMode === "selectCoplanar" ||
    interactionMode === "selectCoplanarEmpty";

  const isDrawVoxelEditMode =
    interactionMode === "add" ||
    interactionMode === "remove" ||
    interactionMode === "paint";

  const showDrawPaneToolMatrix =
    toolsPane === "draw" && (isDrawVoxelEditMode || isSelectionInteractionMode);

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
      (generatorKind === "cloth" && clothPins.length > 0) ||
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
          isSelectionInteractionMode)));

  return (
    <div className={`app${loading && !loadError ? " app-loading-cursor" : ""}`}>
      <div className="app-main">
        {toolsPaneFloating && showEditorChrome ? (
          <div className="app-sidebar-spacer" aria-hidden />
        ) : null}
        {showEditorChrome ? (
          <aside
            className={`${
              sidebarExpanded
                ? "app-sidebar is-expanded"
                : "app-sidebar is-collapsed"
            }${toolsPaneFloating ? " is-floating" : ""}`}
            style={
              toolsPaneFloating
                ? { left: toolPanePos.x, top: toolPanePos.y }
                : undefined
            }
            aria-label="Tools"
          >
            <div
              className={
                toolsPaneFloating
                  ? "sidebar-header sidebar-header-floating"
                  : "sidebar-header"
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
                    {sidebarExpanded ? (
                      <span className="floating-tools-title">Tools</span>
                    ) : null}
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
                      title={
                        sidebarExpanded ? "Collapse tools" : "Expand tools"
                      }
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
                  <div
                    className="sidebar-toolpane-tabs"
                    role="tablist"
                    aria-label="Tool panes"
                  >
                    <button
                      type="button"
                      role="tab"
                      className={
                        toolsPane === "hand"
                          ? "sidebar-pane-tab is-active"
                          : "sidebar-pane-tab"
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
                        toolsPane === "draw"
                          ? "sidebar-pane-tab is-active"
                          : "sidebar-pane-tab"
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
                        toolsPane === "sculpt"
                          ? "sidebar-pane-tab is-active"
                          : "sidebar-pane-tab"
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
                        toolsPane === "squishy"
                          ? "sidebar-pane-tab is-active"
                          : "sidebar-pane-tab"
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
                        toolsPane === "mood"
                          ? "sidebar-pane-tab is-active"
                          : "sidebar-pane-tab"
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
                        toolsPane === "fly"
                          ? "sidebar-pane-tab is-active"
                          : "sidebar-pane-tab"
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
                  </div>
                  <div
                    className="sidebar-expanded-slot"
                    aria-label="Tool pane options"
                  >
                    {toolsPane === "hand" ? (
                      <p className="sidebar-pane-hint">
                        Drag in viewport to orbit/pan.
                      </p>
                    ) : null}

                    {toolsPane === "fly" ? (
                      <p className="sidebar-pane-hint">
                        Click viewport to capture pointer. WASD move, E/Q
                        up/down, Shift slow. Mouse looks. Esc releases pointer.
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
                              {(["add", "remove", "paint"] as const).map(
                                (m) => (
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
                                ),
                              )}
                            </div>
                          </div>
                          <div className="sidebar-tool-selection-col">
                            <div className="sidebar-section-label">
                              Selection
                            </div>
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
                                <span className="sidebar-mode-label">
                                  Select
                                </span>
                              </button>
                              <button
                                type="button"
                                className={
                                  interactionMode === "stamp"
                                    ? "sidebar-mode-btn is-active"
                                    : "sidebar-mode-btn"
                                }
                                disabled={
                                  loading || workBusy || selectionCount === 0
                                }
                                onClick={() => setInteractionMode("stamp")}
                              >
                                <span className="sidebar-mode-label">
                                  Stamp
                                </span>
                              </button>
                              <button
                                type="button"
                                className={
                                  interactionMode === "punch"
                                    ? "sidebar-mode-btn is-active"
                                    : "sidebar-mode-btn"
                                }
                                disabled={
                                  loading || workBusy || selectionCount === 0
                                }
                                onClick={() => setInteractionMode("punch")}
                              >
                                <span className="sidebar-mode-label">
                                  Punch
                                </span>
                              </button>
                            </div>
                          </div>
                        </div>

                        <div className="sidebar-section-label">
                          Selection method
                        </div>
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
                              disabled={
                                loading ||
                                workBusy ||
                                interactionMode !== "sculpt"
                              }
                              onClick={() => setSculptStrokeMode(id)}
                            >
                              <span className="sidebar-mode-label">
                                {label}
                              </span>
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
                        />
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                          Brush and terrain options are in the tool options
                          panel.
                        </p>
                      </>
                    ) : null}

                    {toolsPane === "generators" ? (
                      <>
                        <div className="sidebar-section-label">Generators</div>
                        <button
                          type="button"
                          className={
                            interactionMode === "generator"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setInteractionMode("generator")}
                        >
                          <span className="sidebar-mode-label">Place</span>
                        </button>
                        <div className="sidebar-section-label">Tool</div>
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
                              }}
                            >
                              <span className="sidebar-mode-label">
                                {label}
                              </span>
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
                        />
                        {generatorKind === "rope" &&
                        ropeFirstScreen &&
                        !ropePhase.active ? (
                          <p
                            className="sidebar-pane-hint sidebar-toolpanel-hint"
                            role="status"
                          >
                            Click second point for rope.
                          </p>
                        ) : null}
                        {generatorKind === "rope" && ropePhase.active ? (
                          <p
                            className="sidebar-pane-hint sidebar-toolpanel-hint"
                            role="status"
                          >
                            Adjust tension and sag, then Done.
                          </p>
                        ) : null}
                        {generatorKind === "cloth" && !clothPhase.active ? (
                          <p
                            className="sidebar-pane-hint sidebar-toolpanel-hint"
                            role="status"
                          >
                            Cloth: click surface to add pins (3+ corners), then
                            Done.
                          </p>
                        ) : null}
                        {generatorKind === "cloth" && clothPhase.active ? (
                          <p
                            className="sidebar-pane-hint sidebar-toolpanel-hint"
                            role="status"
                          >
                            Adjust settings, then Done.
                          </p>
                        ) : null}
                        {generatorKind === "roof" ? (
                          <p
                            className="sidebar-pane-hint sidebar-toolpanel-hint"
                            role="status"
                          >
                            Roof: click the surface to add pins (3+ corners),
                            then Apply in tool options.
                          </p>
                        ) : null}
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                          Size and shape in the tool options panel. Rope: two
                          clicks. Cloth: multi-pin + Apply.
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
                        />
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                          Add / pick / delete blobs in the viewport; commit in
                          tool options.
                        </p>
                      </>
                    ) : null}

                    {toolsPane === "mood" ? (
                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Mood sliders are in the tool options panel.
                      </p>
                    ) : null}

                    <ViewportSettingsSidebar
                      loading={loading}
                      workBusy={workBusy}
                    />
                  </div>
                </>
              ) : (
                <div
                  className="sidebar-mode-group"
                  role="group"
                  aria-label="Interaction mode"
                >
                  <button
                    type="button"
                    className={
                      interactionMode === "navigate"
                        ? "sidebar-mode-btn is-active"
                        : "sidebar-mode-btn"
                    }
                    disabled={loading || workBusy}
                    onClick={() => setInteractionMode("navigate")}
                    title="Navigate"
                  >
                    <span className="sidebar-mode-icon" aria-hidden>
                      ✋
                    </span>
                  </button>
                  <button
                    type="button"
                    className={
                      interactionMode === "add"
                        ? "sidebar-mode-btn is-active"
                        : "sidebar-mode-btn"
                    }
                    disabled={loading || workBusy}
                    onClick={() => setInteractionMode("add")}
                    title="Add"
                  >
                    <span className="sidebar-mode-icon" aria-hidden>
                      👇
                    </span>
                  </button>
                  <button
                    type="button"
                    className={
                      interactionMode === "sculpt"
                        ? "sidebar-mode-btn is-active"
                        : "sidebar-mode-btn"
                    }
                    disabled={loading || workBusy}
                    onClick={() => setInteractionMode("sculpt")}
                    title="Sculpt"
                  >
                    <span className="sidebar-mode-icon" aria-hidden>
                      ∧
                    </span>
                  </button>
                  <button
                    type="button"
                    className={
                      interactionMode === "fly"
                        ? "sidebar-mode-btn is-active"
                        : "sidebar-mode-btn"
                    }
                    disabled={loading || workBusy}
                    onClick={() => setInteractionMode("fly")}
                    title="Fly"
                  >
                    <span className="sidebar-mode-icon" aria-hidden>
                      ✈
                    </span>
                  </button>
                </div>
              )}
            </div>
          </aside>
        ) : null}
        <div
          className={`viewport-wrap${showStartScreen ? " is-start-screen" : ""}${
            showEditorChrome && !rightSidebarExpanded
              ? " is-right-sidebar-collapsed"
              : ""
          }`}
        >
          {loading || workBusy ? (
            <div className="load-bar" aria-hidden>
              <div
                className="load-bar-fill"
                style={{
                  width: `${Math.round(
                    Math.min(
                      1,
                      Math.max(0, loading ? loadProgress : workProgress),
                    ) * 100,
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
                : interactionMode === "fly"
                  ? "viewport viewport-mode-fly"
                  : "viewport viewport-mode-edit"
            }
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerLeave}
            onGotPointerCapture={onGotPointerCapture}
            onLostPointerCapture={onLostPointerCapture}
            onContextMenu={(ev) => ev.preventDefault()}
            onWheel={onWheel}
            role="application"
            aria-label="3D viewport"
          >
            {showEditorChrome && viewportCursorDebugEnabled ? (
              <div className="viewport-cursor-debug-overlay" aria-hidden>
                {(() => {
                  const overlayRect =
                    viewportRef.current?.getBoundingClientRect() ?? null;
                  const jsPct = viewportCursorDebugJs
                    ? viewportCursorOverlayPercent(
                        fullSurfaceViewportEnabled,
                        viewportCursorDebugJs.nx,
                        viewportCursorDebugJs.ny,
                        overlayRect,
                      )
                    : null;
                  const rustPct =
                    viewportCursorDebugRust?.previewNx != null &&
                    viewportCursorDebugRust.previewNy != null
                      ? viewportCursorOverlayPercent(
                          fullSurfaceViewportEnabled,
                          viewportCursorDebugRust.previewNx,
                          viewportCursorDebugRust.previewNy,
                          overlayRect,
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
                    Full-window GPU (experimental){" "}
                    {fullSurfaceViewportEnabled ? "on" : "off"}
                  </div>
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
                          const expX = Math.round(
                            (s.rectLeft / s.layoutWidth) * r.surfaceWidth,
                          );
                          const expY = Math.round(
                            (s.rectTop / s.layoutHeight) * r.surfaceHeight,
                          );
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
              <div
                className="viewport-top-center-hud"
                onPointerDown={(e) => e.stopPropagation()}
              >
                {viewportTopCenterHud ? (
                  <div
                    className="viewport-work-phase-chip"
                    role="status"
                    aria-live="polite"
                  >
                    <span className="viewport-work-phase-text">
                      {viewportTopCenterHud.label}
                    </span>
                    <span className="viewport-work-phase-pct">
                      {viewportTopCenterHud.pct}%
                    </span>
                    {viewportTopCenterHud.showFillCancel ? (
                      <button
                        type="button"
                        className="viewport-work-phase-cancel"
                        onClick={() =>
                          void invoke("voxel_fill_cancel").catch(() => {})
                        }
                      >
                        Cancel
                      </button>
                    ) : null}
                  </div>
                ) : null}
                {cuboidPhase.active || cylinderPhase.active ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label={
                      cuboidPhase.active
                        ? "Cuboid extrusion depth"
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
                          if (extrusionDepthEditing)
                            setExtrusionDepthDraft(String(n));
                        } else {
                          const n = Math.max(-256, cylinderDepthUi - 1);
                          cylinderDepthRef.current = n;
                          setCylinderDepthUi(n);
                          if (extrusionDepthEditing)
                            setExtrusionDepthDraft(String(n));
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
                          : cylinderDepthUi;
                        setExtrusionDepthEditing(true);
                        setExtrusionDepthDraft(String(cur));
                        e.target.select();
                      }}
                      onBlur={() => {
                        const current = cuboidPhase.active
                          ? cuboidDepthUi
                          : cylinderDepthUi;
                        let n = parseInt(extrusionDepthDraft, 10);
                        if (Number.isNaN(n)) n = current;
                        n = Math.max(-256, Math.min(256, n));
                        if (cuboidPhase.active) {
                          setCuboidDepthUi(n);
                          cuboidDepthRef.current = n;
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
                          if (extrusionDepthEditing)
                            setExtrusionDepthDraft(String(n));
                        } else {
                          const n = Math.min(256, cylinderDepthUi + 1);
                          cylinderDepthRef.current = n;
                          setCylinderDepthUi(n);
                          if (extrusionDepthEditing)
                            setExtrusionDepthDraft(String(n));
                        }
                      }}
                    >
                      +
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      onClick={() => {
                        if (cuboidPhase.active)
                          void commitCuboidSolidAtScreen();
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
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                      Extrude
                    </span>
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
                {/* Cloth placing phase: pin count + Done/Cancel */}
                {generatorKind === "cloth" &&
                !clothPhase.active &&
                clothPins.length > 0 ? (
                  <div
                    className="viewport-cuboid-depth-bar"
                    role="dialog"
                    aria-label="Cloth pin placement"
                  >
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                      Cloth
                    </span>
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
                        void invoke("voxel_stroke_preview_reset").catch(
                          () => {},
                        );
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
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                      Cloth
                    </span>
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
                      onChange={(ev) =>
                        setClothTension(Number(ev.target.value))
                      }
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
                    <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                      Rope
                    </span>
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
                      onChange={(ev) =>
                        setRopeTension(Number(ev.target.value))
                      }
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
                    <span
                      style={{
                        fontSize: "0.78rem",
                        color: "var(--app-text-muted)",
                      }}
                    >
                      Sag
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={8}
                      step={0.1}
                      value={ropeSag}
                      onChange={(ev) => setRopeSag(Number(ev.target.value))}
                      style={{ width: 60 }}
                    />
                    <span
                      style={{
                        fontSize: "0.78rem",
                        minWidth: "1.6em",
                        textAlign: "right",
                      }}
                    >
                      {ropeSag.toFixed(1)}
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
                {showWallSculptPolygonHud ? (
                  <div
                    className="viewport-polygon-phase-bar"
                    role="dialog"
                    aria-label="Wall polygon outline"
                  >
                    <p className="viewport-polygon-phase-hint">
                      Wall outline: {wallSculptPolygonVerts.length} corners.
                      Click the surface to add; Done applies (min 2 corners).
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
                        disabled={
                          loading ||
                          workBusy ||
                          wallSculptPolygonVerts.length < 2
                        }
                        onClick={() => commitWallSculptPolygonStroke()}
                      >
                        Done
                      </button>
                    </div>
                  </div>
                ) : null}
                {showPolygonPhaseHud ? (
                  <div
                    className="viewport-polygon-phase-bar"
                    role="dialog"
                    aria-label="Polygon area"
                  >
                    <p className="viewport-polygon-phase-hint">
                      Vertices: {strokePolygonVerts.length}. Click to add
                      corners; Apply with three or more.
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
                        disabled={
                          loading || workBusy || strokePolygonVerts.length < 3
                        }
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
          </div>
          {showEditorChrome ? (
            <div className="viewport-ping-overlay" aria-hidden>
              {pingLabelCss ? (
                <div
                  className="viewport-ping-label"
                  style={{
                    left: `${pingLabelCss.leftPct}%`,
                    top: `${pingLabelCss.topPct}%`,
                  }}
                >
                  {pingLabelCss.name}
                </div>
              ) : null}
            </div>
          ) : null}
          {peerLabels.length > 0 ? (
            <div className="viewport-peer-labels-overlay" aria-hidden>
              {peerLabels.map((pl, i) => (
                <div
                  key={i}
                  className="viewport-peer-label"
                  style={
                    {
                      left: `${pl.leftPct}%`,
                      top: `${pl.topPct}%`,
                      "--peer-color": `#${pl.colorRgb.toString(16).padStart(6, "0")}`,
                    } as React.CSSProperties
                  }
                >
                  {pl.name}
                </div>
              ))}
            </div>
          ) : null}
          {showEditorChrome ? (
            <ViewportCameraHud
              flyMode={interactionMode === "fly"}
              loadingOrBusy={loading || workBusy}
            />
          ) : null}
          {showToolOptionsPanel ? (
            <div
              className={`tool-options-panel${
                toolsPaneFloating ? " is-tools-floating" : ""
              }${
                !toolsPaneFloating && sidebarExpanded
                  ? " is-sidebar-expanded"
                  : ""
              }${
                !toolsPaneFloating && !sidebarExpanded
                  ? " is-sidebar-collapsed"
                  : ""
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
                    <span
                      className="tool-panel-selection-count"
                      role="status"
                      aria-live="polite"
                    >
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
              {toolsPane === "draw" &&
              drawStrokeMode === "fill" &&
              (interactionMode === "add" ||
                interactionMode === "remove" ||
                interactionMode === "paint" ||
                isSelectionInteractionMode) ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Fill</div>
                  <p className="tool-options-hint">
                    Click a solid voxel. The connected region is filled,
                    recolored, or added to the selection per your current tool
                    and the options below.
                  </p>
                </div>
              ) : null}
              {toolsPane === "draw" && isSelectionInteractionMode ? (
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
                    setSelectionStrokeSnapToSurface={
                      setSelectionStrokeSnapToSurface
                    }
                    selectionStrokeAxisAlign={selectionStrokeAxisAlign}
                    setSelectionStrokeAxisAlign={setSelectionStrokeAxisAlign}
                    surfacePlaneHollow={surfacePlaneHollow}
                    setSurfacePlaneHollow={setSurfacePlaneHollow}
                    sprayConstrainToPlane={sprayConstrainToPlane}
                    setSprayConstrainToPlane={setSprayConstrainToPlane}
                    spraySizeRange={spraySizeRange}
                    setSpraySizeRange={setSpraySizeRange}
                    fillConstrainToPlane={fillConstrainToPlane}
                    setFillConstrainToPlane={setFillConstrainToPlane}
                  />
                </>
              ) : null}
              {toolsPane === "draw" && isSelectionInteractionMode ? (
                <>
                  {interactionMode === "selectByColor" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">By color</div>
                      <p className="tool-options-hint">
                        Click a voxel to select all connected voxels of the same
                        color.
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
                          onChange={(ev) =>
                            setMatchMaterialSelectColor(ev.target.checked)
                          }
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
                        Click a solid voxel to extend the selection along the
                        same face plane.
                      </p>
                    </div>
                  ) : null}
                  {interactionMode === "selectCoplanarEmpty" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">Coplanar void</div>
                      <p className="tool-options-hint">
                        Click empty space on a plane to select the coplanar
                        empty region.
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
                          onChange={(ev) =>
                            setSculptBrushRadius(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="Brush size (1–64 voxels)"
                        />
                        <span className="tool-options-range-value">
                          {sculptBrushRadius + 1}
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Strength</span>
                        <input
                          type="range"
                          min={1}
                          max={100}
                          value={sculptBrushStrength}
                          onChange={(ev) =>
                            setSculptBrushStrength(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="How strongly the brush applies (with falloff)"
                        />
                        <span className="tool-options-range-value">
                          {sculptBrushStrength}
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Falloff</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={sculptBrushFalloff}
                          onChange={(ev) =>
                            setSculptBrushFalloff(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="0 = hard edge; higher = softer falloff toward brush radius"
                        />
                        <span className="tool-options-range-value">
                          {sculptBrushFalloff}
                        </span>
                      </label>
                      {sculptStrokeMode === "draw" ||
                      sculptStrokeMode === "smooth" ||
                      sculptStrokeMode === "gouge" ||
                      sculptStrokeMode === "extrude" ||
                      sculptStrokeMode === "terrain" ? (
                        <label
                          className="tool-options-checkbox-row"
                          style={{ marginTop: "0.35rem" }}
                        >
                          <input
                            type="checkbox"
                            checked={brushClipBottomHalf}
                            onChange={(ev) =>
                              setBrushClipBottomHalf(ev.target.checked)
                            }
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
                          <div
                            className="tool-options-heading"
                            style={{ marginTop: "0.35rem" }}
                          >
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
                          <div
                            className="tool-options-heading"
                            style={{ marginTop: "0.35rem" }}
                          >
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
                          <div
                            className="tool-options-heading"
                            style={{ marginTop: "0.35rem" }}
                          >
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
                              onChange={(ev) =>
                                setExtrudeTaper(ev.target.checked)
                              }
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
                                  onChange={(ev) =>
                                    setExtrudeTaperStart(
                                      Number(ev.target.value),
                                    )
                                  }
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
                                  onChange={(ev) =>
                                    setExtrudeTaperEnd(Number(ev.target.value))
                                  }
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
                          <div
                            className="tool-options-heading"
                            style={{ marginTop: "0.35rem" }}
                          >
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
                                sculptBrushShapeUi === "circle" ||
                                sculptBrushShapeUi === "sphere"
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
                                sculptBrushShapeUi === "square" ||
                                sculptBrushShapeUi === "cube"
                                  ? "tool-options-shape-btn is-active"
                                  : "tool-options-shape-btn"
                              }
                              disabled={loading || workBusy}
                              onClick={() => setSculptBrushShapeUi("square")}
                              title="Square footprint in XZ"
                            >
                              Circle
                            </button>
                          </div>
                          <div
                            className="tool-options-heading"
                            style={{ marginTop: "0.35rem" }}
                          >
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
                                setTerrainBaseY(
                                  Math.max(-512, Math.min(512, n)),
                                );
                              }}
                              disabled={loading || workBusy}
                            />
                          </label>
                          {terrainSculptOp === "raise" ||
                          terrainSculptOp === "lower" ? (
                            <label className="tool-options-range-label tool-options-range-with-value">
                              <span>Strength</span>
                              <input
                                type="range"
                                min={1}
                                max={32}
                                value={terrainStrength}
                                onChange={(ev) =>
                                  setTerrainStrength(Number(ev.target.value))
                                }
                                disabled={loading || workBusy}
                              />
                              <span className="tool-options-range-value">
                                {terrainStrength}
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
                                onChange={(ev) =>
                                  setTerrainSmoothRadius(
                                    Number(ev.target.value),
                                  )
                                }
                                disabled={loading || workBusy}
                              />
                              <span className="tool-options-range-value">
                                {terrainSmoothRadius}
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
                              onChange={(ev) =>
                                setSculptSmoothPasses(Number(ev.target.value))
                              }
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">
                              {sculptSmoothPasses}
                            </span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Neighbor radius</span>
                            <input
                              type="range"
                              min={0}
                              max={6}
                              value={smoothNeighborRadius}
                              onChange={(ev) =>
                                setSmoothNeighborRadius(Number(ev.target.value))
                              }
                              disabled={loading || workBusy}
                              title="0 = six face neighbors only"
                            />
                            <span className="tool-options-range-value">
                              {smoothNeighborRadius}
                            </span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Aggressiveness</span>
                            <input
                              type="range"
                              min={0}
                              max={100}
                              value={smoothAggressiveness}
                              onChange={(ev) =>
                                setSmoothAggressiveness(Number(ev.target.value))
                              }
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">
                              {smoothAggressiveness}
                            </span>
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
                                setSmoothLaplacianIterations(
                                  Number(ev.target.value),
                                )
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
                              onChange={(ev) =>
                                setSmoothLaplacianRelaxPct(
                                  Number(ev.target.value),
                                )
                              }
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
                              onChange={(ev) =>
                                setSmoothNeighborRadius(Number(ev.target.value))
                              }
                              disabled={loading || workBusy}
                              title="Neighborhood margin + mesh fallback"
                            />
                            <span className="tool-options-range-value">
                              {smoothNeighborRadius}
                            </span>
                          </label>
                          <label className="tool-options-range-label tool-options-range-with-value">
                            <span>Fallback aggressiveness</span>
                            <input
                              type="range"
                              min={0}
                              max={100}
                              value={smoothAggressiveness}
                              onChange={(ev) =>
                                setSmoothAggressiveness(Number(ev.target.value))
                              }
                              disabled={loading || workBusy}
                            />
                            <span className="tool-options-range-value">
                              {smoothAggressiveness}
                            </span>
                          </label>
                        </>
                      )}
                    </div>
                  ) : null}
                  {sculptStrokeMode === "wall" ? (
                    <div
                      className="tool-options-section"
                      aria-label="Sculpt wall"
                    >
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
                          onChange={(ev) =>
                            setSprayDirection(
                              ev.target.value as SprayDirectionApi,
                            )
                          }
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
                          onChange={(ev) =>
                            setWallWidthIndex(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="Path thickness (1–64 voxels)"
                        />
                        <span className="tool-options-range-value">
                          {wallWidthIndex + 1}
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Height</span>
                        <input
                          type="range"
                          min={2}
                          max={20}
                          value={wallHeightVox}
                          onChange={(ev) =>
                            setWallHeightVox(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="Voxels to extend along direction (min 2)"
                        />
                        <span className="tool-options-range-value">
                          {wallHeightVox}
                        </span>
                      </label>
                      <label
                        className="tool-options-checkbox-row"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <input
                          type="checkbox"
                          checked={wallLockStartHeight}
                          onChange={(ev) =>
                            setWallLockStartHeight(ev.target.checked)
                          }
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
                  <p className="tool-options-hint">
                    Select Sculpt mode in the sidebar.
                  </p>
                </div>
              ) : null}
              {toolsPane === "generators" ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Generator</div>
                  {generatorKind === "rocks" ? (
                    <>
                      <label className="tool-options-range-label">
                        <span>Size</span>
                        <input
                          type="range"
                          min={1}
                          max={12}
                          value={Math.min(12, generatorSphereRadius)}
                          onChange={(ev) =>
                            setGeneratorSphereRadius(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Roughness</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={rockRoughness}
                          onChange={(ev) =>
                            setRockRoughness(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  ) : null}
                  {generatorKind === "grass" ? (
                    <>
                      <label className="tool-options-range-label">
                        <span>Density</span>
                        <input
                          type="range"
                          min={1}
                          max={8}
                          value={grassDensity}
                          onChange={(ev) =>
                            setGrassDensity(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Max height</span>
                        <input
                          type="range"
                          min={1}
                          max={8}
                          value={grassMaxHeight}
                          onChange={(ev) =>
                            setGrassMaxHeight(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  ) : null}
                  {generatorKind === "rope" ? (
                    <>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Sag</span>
                        <input
                          type="range"
                          min={0}
                          max={8}
                          step={0.1}
                          value={ropeSag}
                          onChange={(ev) => setRopeSag(Number(ev.target.value))}
                          disabled={loading || workBusy}
                        />
                        <span className="tool-options-range-value">
                          {ropeSag.toFixed(1)}
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Tension</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={ropeTension}
                          onChange={(ev) =>
                            setRopeTension(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="0 = loose, 1 = taut"
                        />
                        <span className="tool-options-range-value">
                          {ropeTension.toFixed(2)}
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Brush</span>
                        <input
                          type="range"
                          min={0}
                          max={SCULPT_BRUSH_MAX_INDEX}
                          value={ropeBrushRadiusIndex}
                          onChange={(ev) =>
                            setRopeBrushRadiusIndex(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                        <span className="tool-options-range-value">
                          {ropeBrushRadiusIndex + 1}
                        </span>
                      </label>
                      <div
                        className="tool-options-shape-row"
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr",
                          gap: "0.25rem",
                          marginTop: "0.35rem",
                        }}
                        role="group"
                        aria-label="Rope brush shape"
                      >
                        <button
                          type="button"
                          className={
                            ropeBrushShapeUi === "sphere"
                              ? "tool-options-shape-btn is-active"
                              : "tool-options-shape-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setRopeBrushShapeUi("sphere")}
                        >
                          Sphere
                        </button>
                        <button
                          type="button"
                          className={
                            ropeBrushShapeUi === "cube"
                              ? "tool-options-shape-btn is-active"
                              : "tool-options-shape-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setRopeBrushShapeUi("cube")}
                        >
                          Cube
                        </button>
                      </div>
                    </>
                  ) : null}
                  {generatorKind === "cloth" ? (
                    <>
                      <label
                        className="tool-options-range-label"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <span>Gravity</span>
                        <select
                          aria-label="Cloth gravity direction"
                          value={clothGravityDirection}
                          onChange={(ev) =>
                            setClothGravityDirection(
                              ev.target.value as ClothGravityDirectionId,
                            )
                          }
                          disabled={loading || workBusy}
                        >
                          <option value="down">Down (−Y)</option>
                          <option value="up">Up (+Y)</option>
                          <option value="left">Left (−X)</option>
                          <option value="right">Right (+X)</option>
                          <option value="forward">Forward (−Z)</option>
                          <option value="back">Back (+Z)</option>
                        </select>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Tension</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={clothTension}
                          onChange={(ev) =>
                            setClothTension(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="0 = loose drape, 1 = stiff"
                        />
                        <span className="tool-options-range-value">
                          {clothTension.toFixed(2)}
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Brush</span>
                        <input
                          type="range"
                          min={0}
                          max={SCULPT_BRUSH_MAX_INDEX}
                          value={ropeBrushRadiusIndex}
                          onChange={(ev) =>
                            setRopeBrushRadiusIndex(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                        <span className="tool-options-range-value">
                          {ropeBrushRadiusIndex + 1}
                        </span>
                      </label>
                      <div
                        className="tool-options-shape-row"
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr",
                          gap: "0.25rem",
                          marginTop: "0.35rem",
                        }}
                        role="group"
                        aria-label="Cloth brush shape"
                      >
                        <button
                          type="button"
                          className={
                            ropeBrushShapeUi === "sphere"
                              ? "tool-options-shape-btn is-active"
                              : "tool-options-shape-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setRopeBrushShapeUi("sphere")}
                        >
                          Sphere
                        </button>
                        <button
                          type="button"
                          className={
                            ropeBrushShapeUi === "cube"
                              ? "tool-options-shape-btn is-active"
                              : "tool-options-shape-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setRopeBrushShapeUi("cube")}
                        >
                          Cube
                        </button>
                      </div>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Sim gravity</span>
                        <input
                          type="range"
                          min={50}
                          max={200}
                          step={5}
                          value={clothSimGravityPct}
                          onChange={(ev) =>
                            setClothSimGravityPct(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="PBD gravity step scale"
                        />
                        <span className="tool-options-range-value">
                          {clothSimGravityPct}%
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Stiffness</span>
                        <input
                          type="range"
                          min={50}
                          max={150}
                          step={5}
                          value={clothSimStiffnessPct}
                          onChange={(ev) =>
                            setClothSimStiffnessPct(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                        <span className="tool-options-range-value">
                          {clothSimStiffnessPct}%
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Iterations</span>
                        <input
                          type="range"
                          min={0}
                          max={64}
                          step={1}
                          value={clothSimIterations}
                          onChange={(ev) =>
                            setClothSimIterations(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                          title="0 = automatic from tension"
                        />
                        <span className="tool-options-range-value">
                          {clothSimIterations === 0
                            ? "Auto"
                            : String(clothSimIterations)}
                        </span>
                      </label>
                      <label className="tool-options-range-label tool-options-range-with-value">
                        <span>Passes</span>
                        <input
                          type="range"
                          min={1}
                          max={6}
                          step={1}
                          value={clothSimConstraintPasses}
                          onChange={(ev) =>
                            setClothSimConstraintPasses(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                        <span className="tool-options-range-value">
                          {clothSimConstraintPasses}
                        </span>
                      </label>
                    </>
                  ) : null}
                  {generatorKind === "ashlar" ? (
                    <>
                      <label className="tool-options-range-label">
                        <span>Size</span>
                        <input
                          type="range"
                          min={1}
                          max={12}
                          value={Math.min(12, generatorSphereRadius)}
                          onChange={(ev) =>
                            setGeneratorSphereRadius(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Roughness</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={rockRoughness}
                          onChange={(ev) =>
                            setRockRoughness(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  ) : null}
                  {generatorKind === "flora" ? (
                    <>
                      <label className="tool-options-range-label">
                        <span>Height</span>
                        <input
                          type="range"
                          min={1}
                          max={96}
                          value={floraHeight}
                          onChange={(ev) =>
                            setFloraHeight(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Girth</span>
                        <input
                          type="range"
                          min={0}
                          max={20}
                          value={floraGirth}
                          onChange={(ev) =>
                            setFloraGirth(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Stems</span>
                        <input
                          type="range"
                          min={1}
                          max={8}
                          value={floraStemCount}
                          onChange={(ev) =>
                            setFloraStemCount(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Branches</span>
                        <input
                          type="range"
                          min={0}
                          max={6}
                          value={floraBranchCount}
                          onChange={(ev) =>
                            setFloraBranchCount(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Canopy</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={floraCanopy}
                          onChange={(ev) =>
                            setFloraCanopy(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  ) : null}
                  {generatorKind === "roof" ? (
                    <>
                      <label
                        className="tool-options-range-label"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <span>Style</span>
                        <select
                          value={roofStyle}
                          onChange={(ev) => setRoofStyle(ev.target.value)}
                          disabled={loading || workBusy}
                        >
                          <option value="flat">Flat</option>
                          <option value="flat_parapet">Flat Parapet</option>
                          <option value="pyramid">Pyramid</option>
                          <option value="cone">Cone</option>
                          <option value="shed">Shed</option>
                          <option value="gable">Gable</option>
                          <option value="saltbox">Saltbox</option>
                          <option value="hip">Hip</option>
                          <option value="barrel">Barrel</option>
                          <option value="mansard">Mansard</option>
                          <option value="gambrel">Gambrel</option>
                          <option value="pavilion">Pavilion</option>
                          <option value="dutch_gable">Dutch Gable</option>
                        </select>
                      </label>
                      <label className="tool-options-range-label">
                        <span>Height</span>
                        <input
                          type="range"
                          min={1}
                          max={32}
                          value={roofHeight}
                          onChange={(ev) =>
                            setRoofHeight(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-checkbox-row">
                        <input
                          type="checkbox"
                          checked={roofHollow}
                          onChange={(ev) => setRoofHollow(ev.target.checked)}
                          disabled={loading || workBusy}
                        />
                        <span>Hollow</span>
                      </label>
                      <p
                        className="sidebar-pane-hint"
                        style={{ marginTop: "0.25rem" }}
                      >
                        Click surface to add pins ({roofPins.length} placed).
                      </p>
                      <button
                        type="button"
                        className="tool-options-shape-btn"
                        style={{ marginTop: "0.5rem", width: "100%" }}
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
                            })
                            .catch(() => {});
                        }}
                      >
                        Apply roof
                      </button>
                    </>
                  ) : null}
                  {generatorKind === "piscina" ? (
                    <>
                      <label
                        className="tool-options-range-label"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <span>Species</span>
                        <select
                          value={piscinaSpecies}
                          onChange={(ev) => setPiscinaSpecies(ev.target.value)}
                          disabled={loading || workBusy}
                        >
                          <option value="trout">Trout</option>
                          <option value="bass">Bass</option>
                          <option value="goldfish">Goldfish</option>
                          <option value="tuna">Tuna</option>
                          <option value="eel">Eel</option>
                        </select>
                      </label>
                      <label className="tool-options-range-label">
                        <span>Length</span>
                        <input
                          type="range"
                          min={4}
                          max={72}
                          value={piscinaLength}
                          onChange={(ev) =>
                            setPiscinaLength(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  ) : null}
                  {generatorKind === "insecta" ? (
                    <>
                      <label
                        className="tool-options-range-label"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <span>Species</span>
                        <select
                          value={insectaSpecies}
                          onChange={(ev) => setInsectaSpecies(ev.target.value)}
                          disabled={loading || workBusy}
                        >
                          <option value="bee">Bee</option>
                          <option value="dragonfly">Dragonfly</option>
                          <option value="grasshopper">Grasshopper</option>
                          <option value="fly">Fly</option>
                          <option value="junebug">June Bug</option>
                        </select>
                      </label>
                      <label className="tool-options-range-label">
                        <span>Length</span>
                        <input
                          type="range"
                          min={12}
                          max={72}
                          value={insectaTotalLength}
                          onChange={(ev) =>
                            setInsectaTotalLength(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    </>
                  ) : null}
                  {generatorKind === "fauna" ? (
                    <>
                      <label
                        className="tool-options-range-label"
                        style={{ marginTop: "0.35rem" }}
                      >
                        <span>Stance</span>
                        <select
                          value={faunaStance}
                          onChange={(ev) => setFaunaStance(ev.target.value)}
                          disabled={loading || workBusy}
                        >
                          <option value="quadruped">Quadruped</option>
                          <option value="biped">Biped</option>
                        </select>
                      </label>
                      <label
                        className="tool-options-range-label"
                        style={{ marginTop: "0.25rem" }}
                      >
                        <span>Archetype</span>
                        <select
                          value={faunaArchetype}
                          onChange={(ev) => setFaunaArchetype(ev.target.value)}
                          disabled={loading || workBusy}
                        >
                          <option value="plantigrade">Plantigrade</option>
                          <option value="digitigrade">Digitigrade</option>
                          <option value="ungulate">Ungulate</option>
                        </select>
                      </label>
                    </>
                  ) : null}
                </div>
              ) : null}
              {toolsPane === "squishy" && interactionMode === "squishy" ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Squishy session</div>
                  <div
                    className="tool-options-shape-row"
                    role="group"
                    aria-label="Squishy mode"
                  >
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
                    Metaballs: {squishyBallCount}. Click viewport to
                    add/pick/delete; Commit voxelizes the combined field.
                  </p>
                  <label className="tool-options-range-label">
                    <span>Blob radius (add)</span>
                    <input
                      type="range"
                      min={2}
                      max={10}
                      value={Math.min(10, Math.max(2, generatorSphereRadius))}
                      onChange={(ev) =>
                        setGeneratorSphereRadius(Number(ev.target.value))
                      }
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
                        onChange={(ev) =>
                          setSquishyWallThickness(Number(ev.target.value))
                        }
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
                      onChange={(ev) =>
                        setSquishySnapToSurface(ev.target.checked)
                      }
                      disabled={loading || workBusy}
                    />
                    <span>Snap add to surface</span>
                  </label>
                  <div
                    className="tool-options-shape-row"
                    style={{ marginTop: "0.35rem" }}
                  >
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      disabled={loading || workBusy}
                      onClick={() => {
                        void invoke("squishy_session_commit", {
                          args: {
                            color: activeColorRef.current,
                            material: activeMaterialRef.current,
                          },
                        })
                          .then(() =>
                            invoke<{ balls: { id: number }[] }>(
                              "squishy_session_get",
                            ),
                          )
                          .then((s) =>
                            setSquishyBallCount(s.balls?.length ?? 0),
                          )
                          .catch(() => {});
                      }}
                    >
                      Commit to voxels
                    </button>
                    <button
                      type="button"
                      className="tool-options-shape-btn"
                      disabled={loading || workBusy}
                      onClick={() => {
                        void invoke("squishy_session_clear")
                          .then(() => setSquishyBallCount(0))
                          .catch(() => {});
                      }}
                    >
                      Clear session
                    </button>
                  </div>
                </div>
              ) : toolsPane === "squishy" ? (
                <div className="tool-options-section">
                  <p className="tool-options-hint">
                    Select Squishy mode in the sidebar.
                  </p>
                </div>
              ) : null}
              {toolsPane === "mood" ? (
                <div className="tool-options-section">
                  {/* ── Grain ────────────────────────────── */}
                  <div className="tool-options-heading">Grain</div>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <input type="checkbox" checked={mood.grainEnabled}
                      onChange={(ev) => setMood((p) => moodWith(p, { grainEnabled: ev.target.checked }))}
                      disabled={loading || workBusy} />
                    <span>Enable grain</span>
                  </label>
                  {mood.grainEnabled && <>
                    <label className="tool-options-range-label">
                      <span>Strength</span>
                      <input type="range" min={0} max={0.5} step={0.01} value={mood.grainStrength}
                        onChange={(ev) => setMood((p) => moodWith(p, { grainStrength: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <input type="checkbox" checked={mood.grainAnimated}
                        onChange={(ev) => setMood((p) => moodWith(p, { grainAnimated: ev.target.checked }))}
                        disabled={loading || workBusy} />
                      <span>Animated</span>
                    </label>
                    {mood.grainAnimated && (
                      <label className="tool-options-range-label">
                        <span>Speed</span>
                        <input type="range" min={0} max={4} step={0.1} value={mood.grainSpeed}
                          onChange={(ev) => setMood((p) => moodWith(p, { grainSpeed: Number(ev.target.value) }))}
                          disabled={loading || workBusy} />
                      </label>
                    )}
                    <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <input type="checkbox" checked={mood.grainColorful}
                        onChange={(ev) => setMood((p) => moodWith(p, { grainColorful: ev.target.checked }))}
                        disabled={loading || workBusy} />
                      <span>Colorful</span>
                    </label>
                  </>}

                  {/* ── Vignette ─────────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>Vignette</div>
                  <label className="tool-options-range-label">
                    <span>Strength</span>
                    <input type="range" min={0} max={1} step={0.02} value={mood.vignette}
                      onChange={(ev) => setMood((p) => moodWith(p, { vignette: Number(ev.target.value) }))}
                      disabled={loading || workBusy} />
                  </label>

                  {/* ── Atmosphere ────────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>Atmosphere</div>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <input type="checkbox" checked={mood.atmEnabled}
                      onChange={(ev) => setMood((p) => moodWith(p, { atmEnabled: ev.target.checked }))}
                      disabled={loading || workBusy} />
                    <span>Enable atmosphere</span>
                  </label>
                  {mood.atmEnabled && <>
                    <label className="tool-options-range-label">
                      <span>Color</span>
                      <input type="color" value={mood.atmColor}
                        onChange={(ev) => setMood((p) => moodWith(p, { atmColor: ev.target.value }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Thickness</span>
                      <input type="range" min={1} max={200} step={1} value={mood.atmThickness}
                        onChange={(ev) => setMood((p) => moodWith(p, { atmThickness: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Density</span>
                      <input type="range" min={0} max={1} step={0.02} value={mood.atmDensity}
                        onChange={(ev) => setMood((p) => moodWith(p, { atmDensity: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.25rem" }}>
                      <label style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
                        <input type="radio" name="atm-spatial" checked={mood.atmAerial}
                          onChange={() => setMood((p) => moodWith(p, { atmAerial: true }))}
                          disabled={loading || workBusy} />
                        <span>Aerial</span>
                      </label>
                      <label style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
                        <input type="radio" name="atm-spatial" checked={!mood.atmAerial}
                          onChange={() => setMood((p) => moodWith(p, { atmAerial: false }))}
                          disabled={loading || workBusy} />
                        <span>Plane</span>
                      </label>
                    </div>
                    {!mood.atmAerial && (
                      <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.25rem" }}>
                        <label style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
                          <input type="radio" name="atm-mode" checked={!mood.atmPositiveSide}
                            onChange={() => setMood((p) => moodWith(p, { atmPositiveSide: false }))}
                            disabled={loading || workBusy} />
                          <span>Layer (slab)</span>
                        </label>
                        <label style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
                          <input type="radio" name="atm-mode" checked={mood.atmPositiveSide}
                            onChange={() => setMood((p) => moodWith(p, { atmPositiveSide: true }))}
                            disabled={loading || workBusy} />
                          <span>Above face</span>
                        </label>
                      </div>
                    )}
                    <label className="tool-options-range-label">
                      <span>Height bias</span>
                      <input type="range" min={-200} max={200} step={1} value={mood.atmHeightBias}
                        onChange={(ev) => setMood((p) => moodWith(p, { atmHeightBias: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Height falloff</span>
                      <input type="range" min={10} max={400} step={1} value={mood.atmHeightFalloff}
                        onChange={(ev) => setMood((p) => moodWith(p, { atmHeightFalloff: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.25rem" }}>
                      <input type="checkbox" checked={mood.atmDriftEnabled}
                        onChange={(ev) => setMood((p) => moodWith(p, { atmDriftEnabled: ev.target.checked }))}
                        disabled={loading || workBusy} />
                      <span>Drift</span>
                    </label>
                    {mood.atmDriftEnabled && <>
                      <label className="tool-options-range-label">
                        <span>Amount</span>
                        <input type="range" min={0} max={1} step={0.02} value={mood.atmDriftAmount}
                          onChange={(ev) => setMood((p) => moodWith(p, { atmDriftAmount: Number(ev.target.value) }))}
                          disabled={loading || workBusy} />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Scale</span>
                        <input type="range" min={0.001} max={0.1} step={0.001} value={mood.atmDriftScale}
                          onChange={(ev) => setMood((p) => moodWith(p, { atmDriftScale: Number(ev.target.value) }))}
                          disabled={loading || workBusy} />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Speed</span>
                        <input type="range" min={0} max={2} step={0.02} value={mood.atmDriftSpeed}
                          onChange={(ev) => setMood((p) => moodWith(p, { atmDriftSpeed: Number(ev.target.value) }))}
                          disabled={loading || workBusy} />
                      </label>
                    </>}
                  </>}

                  {/* ── Distance tint ────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>Distance tint</div>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <input type="checkbox" checked={mood.dtEnabled}
                      onChange={(ev) => setMood((p) => moodWith(p, { dtEnabled: ev.target.checked }))}
                      disabled={loading || workBusy} />
                    <span>Enable distance tint</span>
                  </label>
                  {mood.dtEnabled && <>
                    <label className="tool-options-range-label">
                      <span>Near color</span>
                      <input type="color" value={mood.dtNearColor}
                        onChange={(ev) => setMood((p) => moodWith(p, { dtNearColor: ev.target.value }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Mid color</span>
                      <input type="color" value={mood.dtMidColor}
                        onChange={(ev) => setMood((p) => moodWith(p, { dtMidColor: ev.target.value }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Far color</span>
                      <input type="color" value={mood.dtFarColor}
                        onChange={(ev) => setMood((p) => moodWith(p, { dtFarColor: ev.target.value }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Near distance</span>
                      <input type="range" min={1} max={200} step={1} value={mood.dtNearDist}
                        onChange={(ev) => setMood((p) => moodWith(p, { dtNearDist: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Far distance</span>
                      <input type="range" min={1} max={400} step={1} value={mood.dtFarDist}
                        onChange={(ev) => setMood((p) => moodWith(p, { dtFarDist: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Strength</span>
                      <input type="range" min={0} max={1} step={0.02} value={mood.dtStrength}
                        onChange={(ev) => setMood((p) => moodWith(p, { dtStrength: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                  </>}

                  {/* ── Sun shafts ────────────────────────── */}
                  <div className="tool-options-heading" style={{ marginTop: "0.75rem" }}>Sun shafts</div>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <input type="checkbox" checked={mood.ssEnabled}
                      onChange={(ev) => setMood((p) => moodWith(p, { ssEnabled: ev.target.checked }))}
                      disabled={loading || workBusy} />
                    <span>Enable sun shafts</span>
                  </label>
                  {mood.ssEnabled && <>
                    <label className="tool-options-range-label">
                      <span>Strength</span>
                      <input type="range" min={0} max={10} step={0.1} value={mood.ssStrength}
                        onChange={(ev) => setMood((p) => moodWith(p, { ssStrength: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Decay</span>
                      <input type="range" min={0.5} max={0.99} step={0.01} value={mood.ssDecay}
                        onChange={(ev) => setMood((p) => moodWith(p, { ssDecay: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Density</span>
                      <input type="range" min={0.1} max={1.5} step={0.02} value={mood.ssDensity}
                        onChange={(ev) => setMood((p) => moodWith(p, { ssDensity: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Weight</span>
                      <input type="range" min={0} max={1.5} step={0.02} value={mood.ssWeight}
                        onChange={(ev) => setMood((p) => moodWith(p, { ssWeight: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                    <label className="tool-options-range-label">
                      <span>Samples</span>
                      <input type="range" min={20} max={56} step={1} value={mood.ssSamples}
                        onChange={(ev) => setMood((p) => moodWith(p, { ssSamples: Number(ev.target.value) }))}
                        disabled={loading || workBusy} />
                    </label>
                  </>}
                </div>
              ) : null}
            </div>
          ) : null}
          {showEmptyOpenFile ? (
            <div
              className="viewport-empty-open"
              role="region"
              aria-label="No file open"
            >
              <div className="viewport-empty-open-stack">
                {lastSessionReady &&
                lastSessionInfo?.lastDocumentPath &&
                (lastSessionInfo.documentExists ||
                  lastSessionInfo.autosaveExists) ? (
                  <div
                    className="viewport-empty-last"
                    role="group"
                    aria-label="Continue last project"
                  >
                    <div className="viewport-empty-last-title">
                      Continue where you left off
                    </div>
                    {lastSessionInfo.documentBasename ? (
                      <div
                        className="viewport-empty-last-filename"
                        title={lastSessionInfo.lastDocumentPath ?? undefined}
                      >
                        {lastSessionInfo.documentBasename}
                      </div>
                    ) : null}
                    {lastProjectBlurb ? (
                      <p
                        id="viewport-empty-last-desc"
                        className="viewport-empty-last-blurb"
                      >
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
                        aria-describedby={
                          lastProjectBlurb
                            ? "viewport-empty-last-desc"
                            : undefined
                        }
                      >
                        Reopen last project
                      </button>
                    </div>
                  </div>
                ) : null}
                <button
                  type="button"
                  className="viewport-empty-open-btn is-secondary"
                  onClick={() =>
                    void invoke("open_voxelle_dialog").catch(() => {})
                  }
                >
                  Open file…
                </button>
                <div className="viewport-empty-session-row">
                  <button
                    type="button"
                    className="viewport-empty-open-btn is-secondary"
                    onClick={() => setJoinModalOpen(true)}
                    disabled={collabActive}
                    title={
                      collabActive
                        ? "Leave your session first"
                        : "Paste a host link"
                    }
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
                    {hostWsUrl
                      ? "Stop hosting"
                      : collabGuest
                        ? "Leave"
                        : "Start Session"}
                  </button>
                </div>
              </div>
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
                collabBanner.tone === "alert"
                  ? "viewport-notice is-alert"
                  : "viewport-notice"
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
            <div
              className="chat-toast-stack"
              aria-live="polite"
              aria-label="New chat messages"
            >
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
                      setChatToasts((prev) =>
                        prev.filter((x) => x.id !== t.id),
                      );
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          {chatPanelOpen ? (
            <div
              className="chat-float-panel"
              role="dialog"
              aria-label="Collaboration chat"
            >
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
                  placeholder={
                    collabActive ? "Message…" : "Join or host to chat"
                  }
                  disabled={!collabActive}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) =>
                    collabActive && e.key === "Enter" && sendChat()
                  }
                />
                <button
                  type="button"
                  onClick={sendChat}
                  disabled={!collabActive}
                >
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
                title={
                  rightSidebarExpanded
                    ? "Collapse inspector"
                    : "Expand inspector"
                }
              >
                {rightSidebarExpanded ? (
                  <>
                    <span className="sidebar-expand-toggle-label">
                      Inspector
                    </span>
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
                    {sceneObjectsErr ? (
                      <p className="inspector-hint">{sceneObjectsErr}</p>
                    ) : null}
                    <ul className="inspector-object-list">
                      {sceneObjects
                        .slice()
                        .sort(
                          (a, b) => a.sortOrder - b.sortOrder || a.id - b.id,
                        )
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
                              <span className="inspector-object-name">
                                {o.name}
                              </span>
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
                              {natError} You can forward port {hostPort} in your
                              router settings. Some networks won&apos;t allow
                              guests over the internet.
                            </p>
                          ) : null}
                        </>
                      ) : null}
                      <h4 className="inspector-heading inspector-roster-heading">
                        Roster
                      </h4>
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
                                    onChange={(e) =>
                                      setCanEdit(r.peerId, e.target.checked)
                                    }
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
      <footer
        className={`app-status-bar${showStartScreen ? " is-start-screen" : ""}`}
        role="contentinfo"
      >
        <div className="status-bar-main">
          <div
            className="status-bar-message"
            role="status"
            aria-live="polite"
            title={pathLabel || statusBarMessage}
          >
            {statusBarMessage}
          </div>
          {hostWsUrl ? (
            <button
              type="button"
              className="status-bar-hosting-btn"
              onClick={copyHostingJoinAddress}
              title={hostingCopied ? "Copied" : "Copy invite link"}
            >
              {hostingCopied
                ? "Copied invite link"
                : `Hosting · ${roster.length} ${
                    roster.length === 1 ? "person" : "people"
                  }`}
            </button>
          ) : null}
        </div>
        {showFpsCounter && showEditorChrome ? (
          <div className="fps-counter" role="status" aria-live="polite">
            {fpsDisplayed} FPS
          </div>
        ) : null}
      </footer>
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
      <PreferencesModal
        open={preferencesOpen}
        onClose={() => setPreferencesOpen(false)}
        onFpsCounterChange={setShowFpsCounter}
        onEnableUpnpChange={setPrefsEnableUpnp}
        onCollabDisplayNameChange={setDisplayName}
        onCollabAccentColorChange={setAccentColor}
        onCollabHostPortChange={setHostPort}
        collabHosting={hostWsUrl != null}
      />
      {newProjectOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onClick={(e) =>
            e.target === e.currentTarget && setNewProjectOpen(false)
          }
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
    </div>
  );
}

export default App;
