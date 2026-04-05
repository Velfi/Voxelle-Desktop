// ── Shared types extracted from App.tsx ───────────────────────────────

/** Frozen world-space geometry from `query_cuboid_plane_geometry`. */
export interface CuboidPlaneGeo {
  a: [number, number, number];
  b: [number, number, number];
  planeAx: number;
  hit: [number, number, number];
  prev: [number, number, number];
}

/** Data carried through the cuboid/cylinder solid depth phase. */
export interface DepthPhaseData {
  lineStart: { nx: number; ny: number };
  endNorm: { nx: number; ny: number };
  /** Frozen world-space geometry resolved when entering depth phase.
   *  Passed back to Rust via strokeAux so camera movement cannot change the extrusion direction. */
  frozenGeo: CuboidPlaneGeo | null;
}

// ── Mood state ──────────────────────────────────────────────────────
export interface MoodState {
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
  // screen-space reflections
  ssrEnabled: boolean;
  ssrStrength: number;
  // bloom
  bloomStrength: number;
}

export function defaultMoodState(): MoodState {
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
    ssrEnabled: false,
    ssrStrength: 0.8,
    bloomStrength: 0.1,
  };
}

/** Helper: update one mood field. */
export function moodWith(prev: MoodState, patch: Partial<MoodState>): MoodState {
  return { ...prev, ...patch };
}

// ── Multi-color paint distribution ─────────────────────────────────────────

export type PaintColorMode = "whiteNoise" | "randomSingle" | "fbmNoise" | "gradient" | "dither";

export interface FbmParams {
  octaves: number;
  lacunarity: number;
  persistence: number;
  frequency: number;
  noiseSeed: number;
  quantized: boolean;
}

export interface GradientParams {
  kind: "linear" | "radial";
  linearAxis: 0 | 1 | 2;
  scale: number;
  phase: number;
  radialCenter: [number, number, number];
  quantized: boolean;
}

export interface DitherParams {
  orderedSize: 2 | 4 | 8;
  orderedStrength: number;
  errorDiffusion: "none" | "floydSteinberg";
}

export interface PaintColorDistrib {
  mode: PaintColorMode;
  fbm: FbmParams;
  gradient: GradientParams;
  dither: DitherParams;
}

/** Payload from `get_viewport_cursor_debug` (camelCase). */
export type ViewportCursorDebugPayload = {
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
export type ViewportCursorDebugScreen = {
  clientX: number;
  clientY: number;
  /** Offset inside `.viewport` (`client` - `getBoundingClientRect()`). */
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

export type RenderingMode = "greedy" | "marchingCubes" | "dualContour" | "ray";

export type RosterEntry = {
  peerId: number;
  displayName: string;
  colorRgb: number;
  isLeader: boolean;
  canEdit: boolean;
};

export type LastSessionInfo = {
  lastDocumentPath: string | null;
  documentBasename: string | null;
  autosavePath: string | null;
  documentExists: boolean;
  autosaveExists: boolean;
  autosaveNewerThanDocument: boolean;
};

export type SceneObjectRow = {
  id: number;
  parentId: number | null;
  name: string;
  visible: boolean;
  sortOrder: number;
  translation: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
};

export type ChatToast = { id: number; text: string };

export type InteractionMode =
  | "navigate"
  | "fly"
  | "walk"
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
  | "selectExtrude"
  | "sculpt"
  | "generator"
  | "squishy"
  | "bone";

export type ToolsPane =
  | "hand"
  | "draw"
  | "select"
  | "sculpt"
  | "generators"
  | "squishy"
  | "bone"
  | "mood"
  | "fly"
  | "walk";

/** Matches Rust `SculptStrokeMode` (JSON camelCase). */
export type SculptStrokeModeApi = "draw" | "smooth" | "gouge" | "wall" | "terrain" | "extrude";
/** Matches Rust `TerrainSculptOp`. */
export type TerrainSculptOpApi = "raise" | "lower" | "smooth" | "flatten" | "erode";

export type GeneratorKindId =
  | "rocks"
  | "grass"
  | "rope"
  | "cloth"
  | "ashlar"
  | "flora"
  | "roof"
  | "shape";
export type ClothGravityDirectionId = "down" | "up" | "left" | "right" | "forward" | "back";

export type StartShapeId = "cube" | "orb" | "cylinder" | "hollowCube" | "plane" | "circle";

export type BrushShape = "sphere" | "cube" | "pyramid" | "square" | "circle";

/** Web `SculptBrushShape`; Rust now accepts all four names directly. */
export type SculptBrushShapeUi = "square" | "circle" | "cube" | "sphere";

/** Web `WallAreaShape` / `SprayDirection` (Rust serde camelCase). */
export type WallAreaShapeApi = "brush" | "circle" | "polygon";
/** Web `SculptSmoothVariant`; Rust serde camelCase. */
export type SculptSmoothVariantApi = "majority" | "meshLaplacian";
export type SprayDirectionApi = "auto" | "none" | "right" | "left" | "up" | "down" | "back" | "forward";
