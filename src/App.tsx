import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import { CollabJoinProgressModal } from "./CollabJoinProgressModal";
import { JoinSessionModal } from "./JoinSessionModal";
import { PreferencesModal } from "./PreferencesModal";
import { loadRecentJoinUrls, rememberJoinedUrl } from "./joinRecent";
import {
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

/** Desktop viewer: cap new-project grid edge length (web allows larger). */
const MAX_GRID_SIZE = 256;

const LS_RENDERING_MODE = "voxelleDesktopRenderingMode";
const LS_SIDEBAR_EXPANDED = "voxelleSidebarExpanded";
const LS_RIGHT_SIDEBAR_EXPANDED = "voxelleRightSidebarExpanded";
const LS_TOOLS_FLOATING = "voxelleToolsFloating";
const LS_TOOLS_FLOAT_POS = "voxelleToolsFloatPos";

type RenderingMode =
  | "greedy"
  | "marchingCubes"
  | "dualContour"
  | "ray";

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
    void a.play().catch(() => { });
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

/** Optional note when reopening (backup vs file). */
function lastProjectReopenBlurb(info: LastSessionInfo): string | null {
  if (!info.lastDocumentPath) return null;
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
  | "fill"
  | "eyedropper"
  | "select"
  | "selectByColor"
  | "selectCoplanar"
  | "selectCoplanarEmpty"
  | "stamp"
  | "punch"
  | "sculpt"
  | "generator";

/** `line` = anchor-to-cursor line (web Stroke / Line). `brush` = follow ray + connect samples (web Spray path). */
type StrokeDrawStyle = "line" | "brush";

/** Matches Rust `stroke_modes::DrawStrokeMode` (JSON camelCase). */
type DrawStrokeModeApi =
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
/** Plane constraint for `plane` stroke (Rust `PlaneAxis`). */
type PlaneAxisApi = "auto" | "x" | "y" | "z";

function strokeModeSkipsDrag(mode: DrawStrokeModeApi): boolean {
  return (
    mode === "polygon" ||
    mode === "polygonHull" ||
    mode === "circle" ||
    mode === "cuboid" ||
    mode === "cylinder"
  );
}

type ToolsPane = "hand" | "draw" | "sculpt" | "generators" | "mood" | "fly";

/** Matches Rust `SculptStrokeMode` (JSON camelCase). */
type SculptStrokeModeApi =
  | "draw"
  | "smooth"
  | "gouge"
  | "wall"
  | "terrain"
  | "branch";
/** Matches Rust `TerrainSculptOp`. */
type TerrainSculptOpApi = "raise" | "lower" | "smooth";

type GeneratorKindId = "sphere" | "rocks" | "grass" | "rope" | "squishy";

type BrushShape = "sphere" | "cube" | "pyramid";

/** Distinguishes Stroke vs Solid when both use line stroke + no spray (web parity). */
type StrokeFamilyVariant = "stroke" | "solid";

const MATERIAL_OPTIONS: { id: string; label: string }[] = [
  { id: "plastic", label: "Plastic" },
  { id: "metal", label: "Metal" },
  { id: "rubber", label: "Rubber" },
  { id: "glass", label: "Glass" },
  { id: "water", label: "Water" },
  { id: "glow", label: "Glow" },
];

function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
  /** Physical pixel size of the GPU surface; kept in sync with Rust (may differ slightly from CSS×dpr). */
  const viewportPhysRef = useRef({ w: 0, h: 0 });
  const lastRef = useRef({ x: 0, y: 0 });
  /** Last pointer position over `.viewport` in physical pixels (for Z = ping pick). */
  const lastViewportPickPhysRef = useRef<{ x: number; y: number } | null>(null);
  const pointerStartRef = useRef<{ x: number; y: number } | null>(null);
  const maxPointerMoveRef = useRef(0);
  /** After pick probe: camera orbit/pan/dolly vs voxel click-to-edit (matches web: no hit → camera). */
  const gestureRef = useRef<{
    pointerId: number;
    mode: "camera" | "voxel";
  } | null>(null);
  const probingRef = useRef(false);
  const activePointerIdRef = useRef<number | null>(null);
  const interactionModeRef = useRef<InteractionMode>("navigate");
  const activeColorRef = useRef(0x8899aa);
  const activeMaterialRef = useRef("plastic");
  const brushRadiusRef = useRef(0);
  const brushShapeRef = useRef<BrushShape>("sphere");
  const strokeDrawStyleRef = useRef<StrokeDrawStyle>("line");
  const drawStrokeModeRef = useRef<DrawStrokeModeApi>("line");
  const planeAxisRef = useRef<PlaneAxisApi>("auto");
  const sprayDensityRef = useRef(0);
  /** Viewport-physical start of stroke (for line stroke). */
  const strokeViewportStartRef = useRef<{ x: number; y: number } | null>(null);
  /** Previous sample for brush-mode segment chaining (viewport physical px). */
  const lastStrokePhysRef = useRef<{ x: number; y: number } | null>(null);
  const lastStrokeEditMsRef = useRef(0);
  const dragDidEditRef = useRef(false);
  const loadingRef = useRef(false);
  const interactionBlockedRef = useRef(false);
  const pendingJoinUrlRef = useRef<string | null>(null);
  const collabActiveMenuRef = useRef(false);
  const startHostMenuRef = useRef<() => void>(() => { });
  const leaveSessionMenuRef = useRef<() => void>(() => { });
  const keysDownRef = useRef<Set<string>>(new Set());
  const flyRafRef = useRef<number>(0);
  const flyLastTRef = useRef<number | null>(null);
  const [interactionMode, setInteractionMode] =
    useState<InteractionMode>("navigate");
  const [moodGrain, setMoodGrain] = useState(0);
  const [moodVignette, setMoodVignette] = useState(0);
  const [moodDistanceTint, setMoodDistanceTint] = useState(0);
  const [moodAtmosphere, setMoodAtmosphere] = useState(0);
  const [moodSunShafts, setMoodSunShafts] = useState(0);
  const [selectionCount, setSelectionCount] = useState(0);
  const [matchMaterialSelectColor, setMatchMaterialSelectColor] =
    useState(false);
  const matchMaterialSelectColorRef = useRef(false);
  const [activeColor, setActiveColor] = useState(0x8899aa);
  const [activeMaterial, setActiveMaterial] = useState("plastic");
  const [brushRadius, setBrushRadius] = useState(0);
  const [brushShape, setBrushShape] = useState<BrushShape>("sphere");
  const [strokeDrawStyle, setStrokeDrawStyle] =
    useState<StrokeDrawStyle>("line");
  const [strokeFamilyVariant, setStrokeFamilyVariant] =
    useState<StrokeFamilyVariant>("stroke");
  const [drawStrokeMode, setDrawStrokeMode] =
    useState<DrawStrokeModeApi>("line");
  const [planeAxis, setPlaneAxis] = useState<PlaneAxisApi>("auto");
  const [sprayDensity, setSprayDensity] = useState(0);
  const [toolsPane, setToolsPane] = useState<ToolsPane>("draw");
  const [generatorSphereRadius, setGeneratorSphereRadius] = useState(4);
  const [generatorKind, setGeneratorKind] = useState<GeneratorKindId>("sphere");
  const [squishyMode, setSquishyMode] = useState<"add" | "edit" | "delete">(
    "add",
  );
  const squishyModeRef = useRef<"add" | "edit" | "delete">("add");
  const [squishyHollow, setSquishyHollow] = useState(false);
  const [squishyWallThickness, setSquishyWallThickness] = useState(1);
  const [squishySnapToSurface, setSquishySnapToSurface] = useState(true);
  const [squishyBallCount, setSquishyBallCount] = useState(0);
  const [strokePolygonVerts, setStrokePolygonVerts] = useState<
    [number, number, number][]
  >([]);
  const strokeClickRef = useRef<{
    circleCenter: [number, number, number] | null;
    cuboidMin: [number, number, number] | null;
    cylinderA: [number, number, number] | null;
  }>({
    circleCenter: null,
    cuboidMin: null,
    cylinderA: null,
  });
  const strokePolygonLastScreenRef = useRef<{ x: number; y: number } | null>(
    null,
  );
  const [ropeFirstScreen, setRopeFirstScreen] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [ropeSag, setRopeSag] = useState(2.5);
  const [rockRoughness, setRockRoughness] = useState(0.45);
  const [grassDensity, setGrassDensity] = useState(4);
  const [grassMaxHeight, setGrassMaxHeight] = useState(3);
  const [sculptStrokeMode, setSculptStrokeMode] =
    useState<SculptStrokeModeApi>("draw");
  const [terrainSculptOp, setTerrainSculptOp] =
    useState<TerrainSculptOpApi>("raise");
  const [terrainBaseY, setTerrainBaseY] = useState(0);
  const [terrainStrength, setTerrainStrength] = useState(4);
  const [terrainSmoothRadius, setTerrainSmoothRadius] = useState(2);
  const [sculptSmoothPasses, setSculptSmoothPasses] = useState(1);
  const [pathLabel, setPathLabel] = useState("");
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
  const [lastSessionInfo, setLastSessionInfo] = useState<LastSessionInfo | null>(
    null,
  );
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
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return n & 0xffffff;
  };

  const sendResize = useCallback(() => {
    const el = viewportRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    // Swapchain must match the webview layout size (innerWidth/innerHeight), not only Rust
    // inner_size(), or macOS leaves extra drawable rows below the document → transparent band reads as black.
    const iw = window.innerWidth;
    const ih = window.innerHeight;
    if (iw <= 0 || ih <= 0) return;
    const surfaceWidth = Math.max(1, Math.round(iw * dpr));
    const surfaceHeight = Math.max(1, Math.round(surfaceWidth * (ih / iw)));

    const rect = el.getBoundingClientRect();
    const rw = rect.width;
    const rh = rect.height;
    if (rw <= 0 || rh <= 0) return;
    // Height-first rounding keeps the blit flush with the bottom of `.viewport` (footer sits below).
    const viewportHeight = Math.max(1, Math.round(rh * dpr));
    const viewportWidth = Math.max(1, Math.round(viewportHeight * (rw / rh)));
    const viewportX = Math.max(0, Math.round(rect.left * dpr));
    const viewportY = Math.max(0, Math.round(rect.top * dpr));
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
        invoke<{ width: number; height: number }>("get_viewport_pixel_size"),
      )
      .then((sz) => {
        viewportPhysRef.current = { w: sz.width, h: sz.height };
      })
      .catch(() => { });
  }, []);

  useEffect(() => {
    chatPanelOpenRef.current = chatPanelOpen;
    collabActiveRef.current = collabActive;
    localPeerIdRef.current = localPeerId;
  }, [chatPanelOpen, collabActive, localPeerId]);

  useEffect(() => {
    if (chatPanelOpen) setChatToasts([]);
  }, [chatPanelOpen]);

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
        setPathLabel(e.payload);
        setLoading(true);
        setLoadProgress(0);
        setLoadPhase("");
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
        } else {
          setWorkBusy(true);
        }
      }),
      listen<unknown>("voxelle-loaded", (e) => {
        setLoadError(null);
        const p = e.payload;
        if (typeof p === "string") {
          setPathLabel(p);
        } else if (p && typeof p === "object" && "path" in p) {
          const o = p as {
            path: string;
            mood?: {
              grain: number;
              vignette: number;
              distanceTint: number;
              atmosphere: number;
              sunShafts: number;
            };
          };
          setPathLabel(o.path);
          if (o.mood) {
            setMoodGrain(o.mood.grain);
            setMoodVignette(o.mood.vignette);
            setMoodDistanceTint(o.mood.distanceTint);
            setMoodAtmosphere(o.mood.atmosphere);
            setMoodSunShafts(o.mood.sunShafts);
          } else {
            setMoodGrain(0);
            setMoodVignette(0);
            setMoodDistanceTint(0);
            setMoodAtmosphere(0);
            setMoodSunShafts(0);
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
      listen<{ width: number; height: number }>("viewport-pixel-size", (e) => {
        const p = e.payload;
        viewportPhysRef.current = { w: p.width, h: p.height };
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
          (fromPeerId === undefined ||
            fromPeerId !== localPeerIdRef.current);
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
      listen<number>("voxelle-selection-updated", (e) => {
        setSelectionCount(typeof e.payload === "number" ? e.payload : 0);
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
  }, [sendResize, sidebarExpanded, rightSidebarExpanded, toolsPaneFloating]);

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

  const onToolPaneDragEnd = useCallback((e: PointerEvent) => {
    const d = toolPaneDragRef.current;
    if (!d || e.pointerId !== d.pid) return;
    toolPaneDragRef.current = null;
    window.removeEventListener("pointermove", onToolPaneDragMove);
    window.removeEventListener("pointerup", onToolPaneDragEnd);
    window.removeEventListener("pointercancel", onToolPaneDragEnd);
  }, [onToolPaneDragMove]);

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
        .catch(() => { });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [pingHudTick]);

  useEffect(() => {
    interactionModeRef.current = interactionMode;
  }, [interactionMode]);

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
  const generatorSphereRadiusRef = useRef(4);
  const generatorKindRef = useRef<GeneratorKindId>("sphere");
  const sculptStrokeModeRef = useRef<SculptStrokeModeApi>("draw");
  const terrainSculptOpRef = useRef<TerrainSculptOpApi>("raise");
  const terrainBaseYRef = useRef(0);
  const terrainStrengthRef = useRef(4);
  const terrainSmoothRadiusRef = useRef(2);
  const sculptSmoothPassesRef = useRef(1);
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
    strokeDrawStyleRef.current = strokeDrawStyle;
  }, [strokeDrawStyle]);
  useEffect(() => {
    drawStrokeModeRef.current = drawStrokeMode;
  }, [drawStrokeMode]);
  useEffect(() => {
    planeAxisRef.current = planeAxis;
  }, [planeAxis]);
  useEffect(() => {
    strokeClickRef.current = {
      circleCenter: null,
      cuboidMin: null,
      cylinderA: null,
    };
    setStrokePolygonVerts([]);
    strokePolygonLastScreenRef.current = null;
  }, [drawStrokeMode]);
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
    }).catch(() => { });
  }, [squishyHollow, squishySnapToSurface, squishyWallThickness]);

  function runVoxelEditAtScreen(
    physX: number,
    physY: number,
    strokeAux: {
      polygonVertices?: [number, number, number][];
      circleCenter?: [number, number, number];
      circleEdge?: [number, number, number];
      cuboidMin?: [number, number, number];
      cuboidMax?: [number, number, number];
      cylinderA?: [number, number, number];
      cylinderB?: [number, number, number];
    },
  ) {
    const im = interactionModeRef.current;
    if (im !== "add" && im !== "remove" && im !== "paint") return;
    const tool = im === "add" ? "add" : im === "remove" ? "remove" : "paint";
    void invoke("voxel_edit_at_screen", {
      args: {
        x: physX,
        y: physY,
        tool,
        color: activeColorRef.current,
        material: activeMaterialRef.current,
        brushRadius: brushRadiusRef.current,
        brushShape: brushShapeRef.current,
        sprayDensity: sprayDensityRef.current,
        strokeMode: drawStrokeModeRef.current,
        planeAxis: planeAxisRef.current,
        strokeAux,
        matchMaterial: matchMaterialSelectColorRef.current,
      },
    }).catch(() => { });
  }

  async function handleStrokeAnchorClick(physX: number, physY: number) {
    const im = interactionModeRef.current;
    if (im !== "add" && im !== "remove" && im !== "paint") return;
    const tool = im === "add" ? "add" : im === "remove" ? "remove" : "paint";
    const c = await invoke<[number, number, number] | null>(
      "voxel_stroke_anchor_coord_at_screen",
      { args: { x: physX, y: physY, tool } },
    );
    if (!c) return;
    const sm = drawStrokeModeRef.current;
    if (sm === "polygon" || sm === "polygonHull") {
      setStrokePolygonVerts((v) => [...v, c]);
      strokePolygonLastScreenRef.current = { x: physX, y: physY };
      return;
    }
    if (sm === "circle") {
      const r = strokeClickRef.current;
      if (!r.circleCenter) {
        r.circleCenter = c;
      } else {
        runVoxelEditAtScreen(physX, physY, {
          circleCenter: r.circleCenter,
          circleEdge: c,
        });
        r.circleCenter = null;
      }
      return;
    }
    if (sm === "cuboid") {
      const r = strokeClickRef.current;
      if (!r.cuboidMin) {
        r.cuboidMin = c;
      } else {
        runVoxelEditAtScreen(physX, physY, {
          cuboidMin: r.cuboidMin,
          cuboidMax: c,
        });
        r.cuboidMin = null;
      }
      return;
    }
    if (sm === "cylinder") {
      const r = strokeClickRef.current;
      if (!r.cylinderA) {
        r.cylinderA = c;
      } else {
        runVoxelEditAtScreen(physX, physY, {
          cylinderA: r.cylinderA,
          cylinderB: c,
        });
        r.cylinderA = null;
      }
    }
  }

  function applyPolygonStrokeFill() {
    if (strokePolygonVerts.length < 3) return;
    const scr =
      strokePolygonLastScreenRef.current ?? lastViewportPickPhysRef.current;
    const x = scr?.x ?? 0;
    const y = scr?.y ?? 0;
    runVoxelEditAtScreen(x, y, {
      polygonVertices: strokePolygonVerts.map((v) => [v[0], v[1], v[2]]),
    });
  }

  useEffect(() => {
    sprayDensityRef.current = sprayDensity;
  }, [sprayDensity]);
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
    if (
      interactionMode === "add" ||
      interactionMode === "remove" ||
      interactionMode === "paint" ||
      interactionMode === "fill" ||
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

  const previewModeForSync = (m: InteractionMode): string => {
    if (m === "add") return "add";
    if (m === "remove") return "remove";
    if (m === "paint") return "paint";
    if (m === "sculpt") return "add";
    if (m === "fly") return "fly";
    return "navigate";
  };

  useEffect(() => {
    loadingRef.current = loading;
    interactionBlockedRef.current = loading || workBusy;
  }, [loading, workBusy]);

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
    }).catch(() => { });
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
      () => { },
    );
  }, []);

  useEffect(() => {
    void invoke<LastSessionInfo>("get_last_session_info")
      .then((info) => setLastSessionInfo(info))
      .catch(() => setLastSessionInfo(null))
      .finally(() => setLastSessionReady(true));
  }, []);

  useEffect(() => {
    const p = loadPreferences();
    void invoke("set_tone_mapping", {
      mode: toneMappingToGpuMode(p.toneMapping),
    }).catch(() => { });
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(LS_RENDERING_MODE) as RenderingMode | null;
    const valid =
      saved &&
      ["greedy", "marchingCubes", "dualContour", "ray"].includes(saved);
    void invoke<RenderingMode>("get_rendering_mode")
      .then((m) => {
        if (valid && saved !== m) {
          void invoke("set_rendering_mode", { mode: saved }).catch(() => { });
        }
      })
      .catch(() => { });
  }, []);

  useEffect(() => {
    if (!collabActive) return;
    const id = window.setInterval(() => {
      void invoke("collab_push_camera").catch(() => { });
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
          void invoke("voxel_redo").catch(() => { });
        } else {
          void invoke("voxel_undo").catch(() => { });
        }
        return;
      }
      if (meta && e.key === "s") {
        e.preventDefault();
        void invoke("save_voxelle").catch(() => {
          void invoke("save_voxelle_as").catch(() => { });
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
      const p = lastViewportPickPhysRef.current;
      if (!p) return;
      e.preventDefault();
      const dn = loadPreferences().collabDisplayName.trim();
      void invoke<{
        ok: boolean;
        x?: number;
        y?: number;
        z?: number;
      }>("ping_cursor_pick", {
        args: { x: p.x, y: p.y, displayName: dn },
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
        .catch(() => { });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [preferencesOpen, joinModalOpen, newProjectOpen, collabJoinPending]);

  const clearPreview = useCallback(() => {
    void invoke("sync_preview_input", {
      args: { x: -1, y: 0, mode: "navigate" },
    }).catch(() => { });
  }, []);

  useEffect(() => {
    void invoke("sync_preview_input", {
      args: { x: -1, y: 0, mode: previewModeForSync(interactionMode) },
    }).catch(() => { });
  }, [interactionMode]);

  useEffect(() => {
    matchMaterialSelectColorRef.current = matchMaterialSelectColor;
  }, [matchMaterialSelectColor]);

  useEffect(() => {
    void invoke("selection_menu_sync_match_material", {
      checked: matchMaterialSelectColor,
    }).catch(() => {});
  }, [matchMaterialSelectColor]);

  useEffect(() => {
    void invoke("set_mood_params", {
      args: {
        grain: moodGrain,
        vignette: moodVignette,
        distanceTint: moodDistanceTint,
        atmosphere: moodAtmosphere,
        sunShafts: moodSunShafts,
      },
    }).catch(() => { });
  }, [moodGrain, moodVignette, moodDistanceTint, moodAtmosphere, moodSunShafts]);

  useEffect(() => {
    if (interactionMode !== "fly") {
      void invoke("set_fly_mode", { enabled: false }).catch(() => { });
      flyLastTRef.current = null;
      keysDownRef.current.clear();
      return;
    }
    void invoke("set_fly_mode", { enabled: true }).catch(() => { });
    const onKeyDown = (e: KeyboardEvent) => {
      keysDownRef.current.add(e.code);
    };
    const onKeyUp = (e: KeyboardEvent) => {
      keysDownRef.current.delete(e.code);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    const tick = (t: number) => {
      const last = flyLastTRef.current ?? t;
      flyLastTRef.current = t;
      const dt = Math.min(0.05, (t - last) / 1000);
      const k = keysDownRef.current;
      let forward = 0;
      let right = 0;
      let up = 0;
      if (k.has("KeyW")) forward += 1;
      if (k.has("KeyS")) forward -= 1;
      if (k.has("KeyD")) right += 1;
      if (k.has("KeyA")) right -= 1;
      if (k.has("Space")) up += 1;
      if (k.has("ShiftLeft") || k.has("ShiftRight")) up -= 1;
      void invoke("camera_fly_tick", {
        args: { forward, right, up, dtSecs: dt },
      }).catch(() => { });
      flyRafRef.current = requestAnimationFrame(tick);
    };
    flyRafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(flyRafRef.current);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      void invoke("set_fly_mode", { enabled: false }).catch(() => { });
    };
  }, [interactionMode]);

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
      .catch(() => { });
  }, [interactionMode]);

  useEffect(() => {
    if (loading || workBusy) {
      clearPreview();
    }
  }, [loading, workBusy, clearPreview]);

  const clientToViewportPhysical = useCallback((e: React.PointerEvent) => {
    const el = viewportRef.current;
    if (!el) {
      const dpr = window.devicePixelRatio || 1;
      return { x: e.clientX * dpr, y: e.clientY * dpr };
    }
    const { w: pw, h: ph } = viewportPhysRef.current;
    const rect = el.getBoundingClientRect();
    const rw = rect.width;
    const rh = rect.height;
    if (pw > 0 && ph > 0 && rw > 0 && rh > 0) {
      // Same normalization as sendResize (fractional rect), not offsetX/clientWidth
      // (integer sizes can disagree with rect aspect and cause edge-only ray error).
      return {
        x: ((e.clientX - rect.left) / rw) * pw,
        y: ((e.clientY - rect.top) / rh) * ph,
      };
    }
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rw * dpr));
    const h = Math.max(1, Math.round(w * (rh / rw)));
    return {
      x: ((e.clientX - rect.left) / rw) * w,
      y: ((e.clientY - rect.top) / rh) * h,
    };
  }, []);

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
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    activePointerIdRef.current = e.pointerId;
    pointerStartRef.current = { x: e.clientX, y: e.clientY };
    maxPointerMoveRef.current = 0;
    probingRef.current = true;
    gestureRef.current = null;

    const { x, y } = clientToViewportPhysical(e);
    const pointerId = e.pointerId;
    const middleButton = e.button === 1;
    const mode = interactionModeRef.current;
    const navigate = mode === "navigate" || mode === "fly";
    const forceCamera =
      middleButton ||
      e.shiftKey ||
      (mode === "add" && e.button !== 0) ||
      (mode === "remove" && e.button !== 0) ||
      (mode === "paint" && e.button !== 0) ||
      (mode === "fill" && e.button !== 0) ||
      (mode === "eyedropper" && e.button !== 0) ||
      (mode === "select" && e.button !== 0) ||
      (mode === "selectByColor" && e.button !== 0) ||
      (mode === "selectCoplanar" && e.button !== 0) ||
      (mode === "selectCoplanarEmpty" && e.button !== 0) ||
      (mode === "stamp" && e.button !== 0) ||
      (mode === "punch" && e.button !== 0) ||
      (mode === "sculpt" && e.button !== 0) ||
      (mode === "generator" && e.button !== 0);

    let hitSolid = false;
    if (
      !loading &&
      !workBusy &&
      !forceCamera &&
      !navigate &&
      (mode === "add" ||
        mode === "remove" ||
        mode === "paint" ||
        mode === "fill" ||
        mode === "eyedropper" ||
        mode === "select" ||
        mode === "selectByColor" ||
        mode === "selectCoplanar" ||
        mode === "selectCoplanarEmpty" ||
        mode === "stamp" ||
        mode === "punch" ||
        mode === "sculpt" ||
        mode === "generator") &&
      e.button === 0
    ) {
      try {
        hitSolid = await invoke<boolean>("voxel_pick_probe", {
          args: { x, y },
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
    lastRef.current = { x: e.clientX, y: e.clientY };

    if (
      gestureRef.current.mode === "voxel" &&
      (mode === "add" ||
        mode === "remove" ||
        mode === "paint" ||
        mode === "sculpt")
    ) {
      dragDidEditRef.current = false;
      strokeViewportStartRef.current = { x, y };
      lastStrokePhysRef.current = { x, y };
      void invoke("voxel_stroke_begin").catch(() => { });
    }

    if (gestureRef.current.mode === "camera" && mode !== "fly") {
      void invoke("viewport_pointer", {
        ev: {
          kind: "down",
          x,
          y,
          dx: 0,
          dy: 0,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey,
        },
      });
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const { x: px, y: py } = clientToViewportPhysical(e);
    lastViewportPickPhysRef.current = { x: px, y: py };
    if (
      !probingRef.current &&
      (interactionModeRef.current === "add" ||
        interactionModeRef.current === "remove" ||
        interactionModeRef.current === "paint" ||
        interactionModeRef.current === "sculpt") &&
      !interactionBlockedRef.current
    ) {
      const m = previewModeForSync(interactionModeRef.current);
      void invoke("sync_preview_input", {
        args: { x: px, y: py, mode: m },
      }).catch(() => { });
    }

    if (probingRef.current && activePointerIdRef.current === e.pointerId) {
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
      if (
        e.buttons &&
        (m === "add" || m === "remove" || m === "paint") &&
        !loading &&
        !workBusy &&
        !strokeModeSkipsDrag(drawStrokeModeRef.current)
      ) {
        const now = Date.now();
        if (now - lastStrokeEditMsRef.current >= 24) {
          lastStrokeEditMsRef.current = now;
          dragDidEditRef.current = true;
          const tool = m === "add" ? "add" : m === "remove" ? "remove" : "paint";
          const lineStart =
            strokeDrawStyleRef.current === "line" && strokeViewportStartRef.current
              ? strokeViewportStartRef.current
              : null;
          const brushPrev =
            strokeDrawStyleRef.current === "brush" && lastStrokePhysRef.current
              ? lastStrokePhysRef.current
              : null;
          void invoke("voxel_edit_at_screen", {
            args: {
              x: px,
              y: py,
              tool,
              color: activeColorRef.current,
              material: activeMaterialRef.current,
              brushRadius: brushRadiusRef.current,
              brushShape: brushShapeRef.current,
              sprayDensity: sprayDensityRef.current,
              strokeMode: drawStrokeModeRef.current,
              planeAxis: planeAxisRef.current,
              strokeAux: {},
              matchMaterial: matchMaterialSelectColorRef.current,
              ...(lineStart
                ? {
                  strokeLineStartX: lineStart.x,
                  strokeLineStartY: lineStart.y,
                }
                : {}),
              ...(!lineStart && brushPrev
                ? {
                  strokeSegmentPrevX: brushPrev.x,
                  strokeSegmentPrevY: brushPrev.y,
                }
                : {}),
            },
          })
            .finally(() => {
              if (strokeDrawStyleRef.current === "brush") {
                lastStrokePhysRef.current = { x: px, y: py };
              }
            })
            .catch(() => { });
        }
      }
      if (
        e.buttons &&
        m === "sculpt" &&
        !loading &&
        !workBusy
      ) {
        const now = Date.now();
        if (now - lastStrokeEditMsRef.current >= 24) {
          lastStrokeEditMsRef.current = now;
          dragDidEditRef.current = true;
          const lineStart =
            strokeDrawStyleRef.current === "line" && strokeViewportStartRef.current
              ? strokeViewportStartRef.current
              : null;
          const brushPrev =
            strokeDrawStyleRef.current === "brush" && lastStrokePhysRef.current
              ? lastStrokePhysRef.current
              : null;
          const sm = sculptStrokeModeRef.current;
          void invoke("voxel_sculpt_stroke_at_screen", {
            args: {
              x: px,
              y: py,
              sculptMode: sm,
              color: activeColorRef.current,
              material: activeMaterialRef.current,
              brushRadius: brushRadiusRef.current,
              brushShape: brushShapeRef.current,
              sprayDensity: sprayDensityRef.current,
              ...(lineStart
                ? {
                  strokeLineStartX: lineStart.x,
                  strokeLineStartY: lineStart.y,
                }
                : {}),
              ...(!lineStart && brushPrev
                ? {
                  strokeSegmentPrevX: brushPrev.x,
                  strokeSegmentPrevY: brushPrev.y,
                }
                : {}),
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
            },
          })
            .finally(() => {
              if (strokeDrawStyleRef.current === "brush") {
                lastStrokePhysRef.current = { x: px, y: py };
              }
            })
            .catch(() => { });
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
      return;
    }
    const { x, y } = clientToViewportPhysical(e);
    if (interactionModeRef.current !== "fly") {
      void invoke("viewport_pointer", {
        ev: {
          kind: "move",
          x,
          y,
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
    if (probingRef.current && activePointerIdRef.current === e.pointerId) {
      probingRef.current = false;
      gestureRef.current = null;
      activePointerIdRef.current = null;
      pointerStartRef.current = null;
      return;
    }

    const g = gestureRef.current;
    const start = pointerStartRef.current;
    const moved = maxPointerMoveRef.current;
    const isThisPointer = g?.pointerId === e.pointerId;

    if (
      isThisPointer &&
      g?.mode === "voxel" &&
      !loading &&
      !workBusy &&
      start &&
      !e.shiftKey &&
      e.button === 0
    ) {
      const { x, y } = clientToViewportPhysical(e);
      const m = interactionModeRef.current;
      if (moved < 5) {
        if (m === "select") {
          void invoke("selection_toggle_at_screen", { args: { x, y } })
            .then(() =>
              invoke<number>("selection_get_count").then((n) =>
                setSelectionCount(n),
              ),
            )
            .catch(() => { });
        } else if (m === "stamp") {
          void invoke("clipboard_stamp_at_screen", {
            args: {
              x,
              y,
              color: activeColorRef.current,
              material: activeMaterialRef.current,
            },
          }).catch(() => { });
        } else if (m === "punch") {
          void invoke("clipboard_punch_at_screen", { args: { x, y } }).catch(
            () => { },
          );
        } else if (m === "generator") {
          const gk = generatorKindRef.current;
          if (gk === "sphere") {
            void invoke("generator_sphere_at_screen", {
              args: {
                x,
                y,
                radius: generatorSphereRadiusRef.current,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => { });
          } else if (gk === "rocks") {
            void invoke("generator_rocks_at_screen", {
              args: {
                x,
                y,
                seed: (Math.random() * 1e9) | 0,
                size: Math.max(1, generatorSphereRadiusRef.current),
                roughness: rockRoughness,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => { });
          } else if (gk === "grass") {
            void invoke("generator_grass_at_screen", {
              args: {
                x,
                y,
                seed: (Math.random() * 1e9) | 0,
                density: grassDensity,
                maxHeight: grassMaxHeight,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch(() => { });
          } else if (gk === "rope") {
            if (!ropeFirstScreen) {
              setRopeFirstScreen({ x, y });
            } else {
              void invoke("generator_rope_at_screen", {
                args: {
                  x1: ropeFirstScreen.x,
                  y1: ropeFirstScreen.y,
                  x2: x,
                  y2: y,
                  sag: ropeSag,
                  color: activeColorRef.current,
                  material: activeMaterialRef.current,
                },
              }).catch(() => { });
              setRopeFirstScreen(null);
            }
          } else if (gk === "squishy") {
            const mode = squishyModeRef.current;
            void invoke("squishy_session_set_mode", { args: { mode } })
              .then(() => {
                if (mode === "add") {
                  return invoke("squishy_metaball_add_at_screen", {
                    args: {
                      x,
                      y,
                      radius: Math.max(2, generatorSphereRadiusRef.current),
                    },
                  });
                }
                return invoke<number | null>("squishy_pick_at_screen", {
                  args: { x, y },
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
              .catch(() => { });
          }
        } else if (m === "selectByColor") {
          void invoke<number>("selection_add_by_color_at_screen", {
            args: {
              x,
              y,
              matchMaterial: matchMaterialSelectColorRef.current,
            },
          })
            .then((n) => {
              if (n > 0) {
                void invoke<number>("selection_get_count").then((c) =>
                  setSelectionCount(c),
                );
              }
            })
            .catch(() => { });
        } else if (m === "selectCoplanar") {
          void invoke<number>("selection_add_coplanar_at_screen", {
            args: { x, y },
          })
            .then((n) => {
              if (n > 0) {
                void invoke<number>("selection_get_count").then((c) =>
                  setSelectionCount(c),
                );
              }
            })
            .catch(() => { });
        } else if (m === "selectCoplanarEmpty") {
          void invoke<number>("selection_add_coplanar_empty_at_screen", {
            args: { x, y },
          })
            .then((n) => {
              if (n > 0) {
                void invoke<number>("selection_get_count").then((c) =>
                  setSelectionCount(c),
                );
              }
            })
            .catch(() => { });
        } else if (m === "fill") {
          void invoke<boolean>("voxel_fill_at_screen", {
            args: {
              x,
              y,
              color: activeColorRef.current,
              material: activeMaterialRef.current,
              matchMaterial: matchMaterialSelectColorRef.current,
            },
          }).catch(() => { });
        }
      }
      if (m === "eyedropper") {
        if (moved < 5) {
          void invoke<{
            color: number;
            material: string;
          } | null>("voxel_pick_color_at_screen", {
            args: {
              x,
              y,
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
              }
            })
            .catch(() => { });
        }
      } else if (m === "add" || m === "remove" || m === "paint") {
        const sm = drawStrokeModeRef.current;
        if (!dragDidEditRef.current && moved < 5) {
          if (strokeModeSkipsDrag(sm)) {
            void handleStrokeAnchorClick(x, y);
          } else {
            const tool = m === "add" ? "add" : m === "remove" ? "remove" : "paint";
            const lineStart =
              strokeDrawStyleRef.current === "line" && strokeViewportStartRef.current
                ? strokeViewportStartRef.current
                : null;
            void invoke("voxel_edit_at_screen", {
              args: {
                x,
                y,
                tool,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
                brushRadius: brushRadiusRef.current,
                brushShape: brushShapeRef.current,
                sprayDensity: sprayDensityRef.current,
                strokeMode: drawStrokeModeRef.current,
                planeAxis: planeAxisRef.current,
                strokeAux: {},
                matchMaterial: matchMaterialSelectColorRef.current,
                ...(lineStart
                  ? {
                    strokeLineStartX: lineStart.x,
                    strokeLineStartY: lineStart.y,
                  }
                  : {}),
              },
            }).catch(() => { });
          }
        }
        void invoke("voxel_stroke_end").catch(() => { });
        lastStrokePhysRef.current = null;
      } else if (m === "sculpt") {
        if (!dragDidEditRef.current && moved < 5) {
          const lineStart =
            strokeDrawStyleRef.current === "line" && strokeViewportStartRef.current
              ? strokeViewportStartRef.current
              : null;
          const sm = sculptStrokeModeRef.current;
          void invoke("voxel_sculpt_stroke_at_screen", {
            args: {
              x,
              y,
              sculptMode: sm,
              color: activeColorRef.current,
              material: activeMaterialRef.current,
              brushRadius: brushRadiusRef.current,
              brushShape: brushShapeRef.current,
              sprayDensity: sprayDensityRef.current,
              ...(lineStart
                ? {
                  strokeLineStartX: lineStart.x,
                  strokeLineStartY: lineStart.y,
                }
                : {}),
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
            },
          }).catch(() => { });
        }
        void invoke("voxel_stroke_end").catch(() => { });
        lastStrokePhysRef.current = null;
      }
    }

    if (isThisPointer && g?.mode === "camera" && interactionModeRef.current !== "fly") {
      const { x, y } = clientToViewportPhysical(e);
      void invoke("viewport_pointer", {
        ev: {
          kind: "up",
          x,
          y,
          dx: 0,
          dy: 0,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey,
        },
      });
    }

    if (isThisPointer) {
      gestureRef.current = null;
      activePointerIdRef.current = null;
    }
    pointerStartRef.current = null;
  };

  const onPointerLeave = (e: React.PointerEvent) => {
    clearPreview();
    onPointerUp(e);
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

  const leaveSession = () => {
    void invoke("collab_leave").catch(() => { });
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
      () => { },
    );
  };

  const amLeader = roster.some(
    (r) => r.peerId === localPeerId && r.isLeader,
  );

  /** Solo or host: can open files. Guests (session without hosting) cannot. */
  const collabGuest = collabActive && !hostWsUrl;
  const showEmptyOpenFile =
    !pathLabel && !loading && !workBusy && !collabGuest;

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
    lastSessionInfo != null
      ? lastProjectReopenBlurb(lastSessionInfo)
      : null;

  const statusBarMessage = (() => {
    if (loading && pathLabel) {
      const pct = Math.round(Math.min(1, Math.max(0, loadProgress)) * 100);
      const phase = loadPhase.trim();
      return phase
        ? `Loading ${basename(pathLabel)}… ${pct}% — ${phase}`
        : `Loading ${basename(pathLabel)}… ${pct}%`;
    }
    if (loading) return "Loading…";
    if (workBusy) {
      const pct = Math.round(Math.min(1, Math.max(0, workProgress)) * 100);
      const phase = workPhase.trim();
      return phase ? `${phase} ${pct}%` : `Working… ${pct}%`;
    }
    if (pathLabel) {
      const base = basename(pathLabel);
      if (collabActive) return `${base} · Live`;
      return base;
    }
    return "No file open";
  })();

  const sendChat = () => {
    const t = chatInput.trim();
    if (!t) return;
    void invoke("collab_send_chat", { text: t }).catch(() => { });
    setChatInput("");
  };

  const onRosterSnapCamera = (peerId: number) => {
    void invoke("collab_snap_camera", { peerId }).catch(() => { });
  };

  const setCanEdit = (peerId: number, canEdit: boolean) => {
    void invoke("collab_set_can_edit", { targetPeer: peerId, canEdit }).catch(
      () => { },
    );
  };

  const showToolOptionsPanel =
    !loading &&
    !workBusy &&
    (toolsPane === "sculpt" ||
      toolsPane === "generators" ||
      toolsPane === "mood" ||
      (toolsPane === "draw" &&
        (interactionMode === "add" ||
          interactionMode === "remove" ||
          interactionMode === "paint" ||
          interactionMode === "fill")));

  return (
    <div className="app">
      <div className="app-main">
        {toolsPaneFloating ? (
          <div className="app-sidebar-spacer" aria-hidden />
        ) : null}
        <aside
          className={`${sidebarExpanded
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
                  title={
                    sidebarExpanded ? "Collapse tools" : "Expand tools"
                  }
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
                    className={toolsPane === "hand" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"}
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
                    className={toolsPane === "draw" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"}
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
                    className={toolsPane === "sculpt" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"}
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
                    className={toolsPane === "generators" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"}
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
                    className={toolsPane === "mood" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"}
                    aria-selected={toolsPane === "mood"}
                    disabled={loading || workBusy}
                    onClick={() => setToolsPane("mood")}
                  >
                    Mood
                  </button>
                  <button
                    type="button"
                    role="tab"
                    className={toolsPane === "fly" ? "sidebar-pane-tab is-active" : "sidebar-pane-tab"}
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
                <div className="sidebar-expanded-slot" aria-label="Tool pane options">
                  {toolsPane === "hand" ? (
                    <p className="sidebar-pane-hint">Drag in viewport to orbit/pan.</p>
                  ) : null}

                  {toolsPane === "fly" ? (
                    <p className="sidebar-pane-hint">Click viewport, then WASD + Space/Shift to fly.</p>
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
                                className={interactionMode === m ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                                disabled={loading || workBusy}
                                onClick={() => setInteractionMode(m)}
                              >
                                <span className="sidebar-mode-label">{m[0].toUpperCase() + m.slice(1)}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="sidebar-tool-selection-col">
                          <div className="sidebar-section-label">Selection</div>
                          <div className="sidebar-selection-count">{selectionCount} selected</div>
                          <div className="sidebar-mode-grid sidebar-mode-grid-stacked">
                            <button
                              type="button"
                              className={interactionMode === "select" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                              disabled={loading || workBusy}
                              onClick={() => setInteractionMode("select")}
                            >
                              <span className="sidebar-mode-label">Select</span>
                            </button>
                            <button
                              type="button"
                              className={interactionMode === "stamp" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                              disabled={loading || workBusy}
                              onClick={() => setInteractionMode("stamp")}
                            >
                              <span className="sidebar-mode-label">Stamp</span>
                            </button>
                            <button
                              type="button"
                              className={interactionMode === "punch" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                              disabled={loading || workBusy}
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
                            interactionMode !== "fill" &&
                              strokeDrawStyle === "line" &&
                              sprayDensity === 0 &&
                              strokeFamilyVariant === "stroke"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            if (interactionMode === "fill") setInteractionMode("add");
                            setStrokeDrawStyle("line");
                            setSprayDensity(0);
                            setStrokeFamilyVariant("stroke");
                          }}
                          title="Line from pointer down to cursor (web Stroke)"
                        >
                          <span className="sidebar-mode-label">Stroke</span>
                        </button>
                        <button
                          type="button"
                          className={
                            interactionMode !== "fill" &&
                              strokeDrawStyle === "brush" &&
                              sprayDensity === 0
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            if (interactionMode === "fill") setInteractionMode("add");
                            setStrokeDrawStyle("brush");
                            setSprayDensity(0);
                          }}
                          title="Brush along the drag (web Surface)"
                        >
                          <span className="sidebar-mode-label">Surface</span>
                        </button>
                        <button
                          type="button"
                          className={
                            interactionMode !== "fill" &&
                              strokeDrawStyle === "line" &&
                              sprayDensity === 0 &&
                              strokeFamilyVariant === "solid"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            if (interactionMode === "fill") setInteractionMode("add");
                            setStrokeDrawStyle("line");
                            setSprayDensity(0);
                            setStrokeFamilyVariant("solid");
                          }}
                          title="Solid volume stroke (line; placeholder for future volume behavior)"
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
                            interactionMode !== "fill" &&
                              strokeDrawStyle === "brush" &&
                              sprayDensity > 0
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => {
                            if (interactionMode === "fill") setInteractionMode("add");
                            setStrokeDrawStyle("brush");
                            setSprayDensity(0.45);
                          }}
                          title="Spray density along brush path"
                        >
                          <span className="sidebar-mode-label">Spray</span>
                        </button>
                        <button
                          type="button"
                          className={
                            interactionMode === "fill"
                              ? "sidebar-mode-btn is-active"
                              : "sidebar-mode-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setInteractionMode("fill")}
                          title="Fill connected region"
                        >
                          <span className="sidebar-mode-label">Fill</span>
                        </button>
                      </div>

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
                          className={interactionMode === "eyedropper" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                          disabled={loading || workBusy}
                          onClick={() => setInteractionMode("eyedropper")}
                        >
                          <span className="sidebar-mode-label">Eyedropper</span>
                        </button>
                      </div>

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

                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Brush size and shape are in the tool options panel.
                      </p>
                    </>
                  ) : null}

                  {toolsPane === "sculpt" ? (
                    <>
                      <div className="sidebar-section-label">Sculpt</div>
                      <button
                        type="button"
                        className={interactionMode === "sculpt" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                        disabled={loading || workBusy}
                        onClick={() => setInteractionMode("sculpt")}
                      >
                        <span className="sidebar-mode-label">Sculpt</span>
                      </button>
                      <div className="sidebar-section-label">Mode</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-3">
                        {(
                          [
                            ["draw", "Draw"],
                            ["smooth", "Smooth"],
                            ["gouge", "Gouge"],
                            ["wall", "Wall"],
                            ["terrain", "Terrain"],
                            ["branch", "Branch"],
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
                      {sculptStrokeMode === "terrain" ? (
                        <>
                          <div className="sidebar-section-label">Terrain op</div>
                          <div className="sidebar-mode-grid sidebar-mode-grid-3">
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
                                    ? "sidebar-mode-btn is-active"
                                    : "sidebar-mode-btn"
                                }
                                disabled={loading || workBusy || interactionMode !== "sculpt"}
                                onClick={() => setTerrainSculptOp(id)}
                              >
                                <span className="sidebar-mode-label">{label}</span>
                              </button>
                            ))}
                          </div>
                        </>
                      ) : null}
                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Brush, stroke, and color are in the tool options panel.
                      </p>
                    </>
                  ) : null}

                  {toolsPane === "generators" ? (
                    <>
                      <div className="sidebar-section-label">Generators</div>
                      <button
                        type="button"
                        className={interactionMode === "generator" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                        disabled={loading || workBusy}
                        onClick={() => setInteractionMode("generator")}
                      >
                        <span className="sidebar-mode-label">Place</span>
                      </button>
                      <div className="sidebar-section-label">Tool</div>
                      <div className="sidebar-mode-grid sidebar-mode-grid-2">
                        {(
                          [
                            ["sphere", "Sphere"],
                            ["rocks", "Rocks"],
                            ["grass", "Grass"],
                            ["rope", "Rope"],
                            ["squishy", "Squishy"],
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
                              setRopeFirstScreen(null);
                            }}
                          >
                            <span className="sidebar-mode-label">{label}</span>
                          </button>
                        ))}
                      </div>
                      {generatorKind === "rope" && ropeFirstScreen ? (
                        <p className="sidebar-pane-hint sidebar-toolpanel-hint" role="status">
                          Click second point to finish rope.
                        </p>
                      ) : null}
                      <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                        Options in the tool options panel. Rope: two clicks.
                      </p>
                    </>
                  ) : null}

                  {toolsPane === "mood" ? (
                    <p className="sidebar-pane-hint sidebar-toolpanel-hint">
                      Mood sliders are in the tool options panel.
                    </p>
                  ) : null}
                </div>
              </>
            ) : (
              <div className="sidebar-mode-group" role="group" aria-label="Interaction mode">
                <button
                  type="button"
                  className={interactionMode === "navigate" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                  disabled={loading || workBusy}
                  onClick={() => setInteractionMode("navigate")}
                  title="Navigate"
                >
                  <span className="sidebar-mode-icon" aria-hidden>✋</span>
                </button>
                <button
                  type="button"
                  className={interactionMode === "add" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                  disabled={loading || workBusy}
                  onClick={() => setInteractionMode("add")}
                  title="Add"
                >
                  <span className="sidebar-mode-icon" aria-hidden>👇</span>
                </button>
                <button
                  type="button"
                  className={interactionMode === "sculpt" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                  disabled={loading || workBusy}
                  onClick={() => setInteractionMode("sculpt")}
                  title="Sculpt"
                >
                  <span className="sidebar-mode-icon" aria-hidden>∧</span>
                </button>
                <button
                  type="button"
                  className={interactionMode === "fly" ? "sidebar-mode-btn is-active" : "sidebar-mode-btn"}
                  disabled={loading || workBusy}
                  onClick={() => setInteractionMode("fly")}
                  title="Fly"
                >
                  <span className="sidebar-mode-icon" aria-hidden>✈</span>
                </button>
              </div>
            )}
          </div>
        </aside>
        <div className="viewport-wrap">
          {loading || workBusy ? (
            <div className="load-bar" aria-hidden>
              <div
                className="load-bar-fill"
                style={{
                  width: `${Math.round(
                    Math.min(1, Math.max(0, loading ? loadProgress : workProgress)) *
                    100,
                  )}%`,
                }}
              />
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
            onContextMenu={(ev) => ev.preventDefault()}
            onWheel={onWheel}
            role="application"
            aria-label="3D viewport"
          />
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
          {showToolOptionsPanel ? (
            <div
              className={`tool-options-panel${toolsPaneFloating ? " is-tools-floating" : ""
                }${!toolsPaneFloating && sidebarExpanded
                  ? " is-sidebar-expanded"
                  : ""
                }${!toolsPaneFloating && !sidebarExpanded
                  ? " is-sidebar-collapsed"
                  : ""
                }`}
              role="dialog"
              aria-label="Tool options"
              onPointerDown={(e) => e.stopPropagation()}
              onPointerUp={(e) => e.stopPropagation()}
            >
              {toolsPane === "draw" &&
                (interactionMode === "add" ||
                  interactionMode === "remove" ||
                  interactionMode === "paint") ? (
                <>
                  <div className="tool-options-section">
                    <div className="tool-options-heading">Brush</div>
                    <div className="tool-options-shape-row" role="group" aria-label="Brush shape">
                      {(
                        [
                          ["sphere", "Sphere"],
                          ["cube", "Cube"],
                          ["pyramid", "Pyr"],
                        ] as const
                      ).map(([id, label]) => (
                        <button
                          key={id}
                          type="button"
                          className={
                            brushShape === id
                              ? "tool-options-shape-btn is-active"
                              : "tool-options-shape-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setBrushShape(id)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                    <label className="tool-options-range-label">
                      <span>Size</span>
                      <input
                        type="range"
                        min={0}
                        max={32}
                        value={brushRadius}
                        onChange={(ev) =>
                          setBrushRadius(Number(ev.target.value))
                        }
                        disabled={loading || workBusy}
                      />
                    </label>
                    {sprayDensity > 0 ? (
                      <label className="tool-options-range-label">
                        <span>Spray density</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={sprayDensity}
                          onChange={(ev) =>
                            setSprayDensity(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    ) : null}
                  </div>
                </>
              ) : null}
              {toolsPane === "draw" && interactionMode === "fill" ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Fill</div>
                  <p className="tool-options-hint">
                    Click a solid voxel. Connected region matching the fill
                    options in the sidebar is recolored.
                  </p>
                </div>
              ) : null}
              {toolsPane === "sculpt" && interactionMode === "sculpt" ? (
                <>
                  <div className="tool-options-section">
                    <div className="tool-options-heading">Brush</div>
                    <div className="tool-options-shape-row" role="group" aria-label="Brush shape">
                      {(
                        [
                          ["sphere", "Sphere"],
                          ["cube", "Cube"],
                          ["pyramid", "Pyr"],
                        ] as const
                      ).map(([id, label]) => (
                        <button
                          key={id}
                          type="button"
                          className={
                            brushShape === id
                              ? "tool-options-shape-btn is-active"
                              : "tool-options-shape-btn"
                          }
                          disabled={loading || workBusy}
                          onClick={() => setBrushShape(id)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                    <label className="tool-options-range-label">
                      <span>Size</span>
                      <input
                        type="range"
                        min={0}
                        max={32}
                        value={brushRadius}
                        onChange={(ev) =>
                          setBrushRadius(Number(ev.target.value))
                        }
                        disabled={loading || workBusy}
                      />
                    </label>
                    <div className="tool-options-heading" style={{ marginTop: "0.5rem" }}>
                      Stroke
                    </div>
                    <div className="tool-options-shape-row" role="group" aria-label="Stroke style">
                      <button
                        type="button"
                        className={
                          strokeDrawStyle === "line"
                            ? "tool-options-shape-btn is-active"
                            : "tool-options-shape-btn"
                        }
                        disabled={loading || workBusy}
                        onClick={() => setStrokeDrawStyle("line")}
                      >
                        Line
                      </button>
                      <button
                        type="button"
                        className={
                          strokeDrawStyle === "brush"
                            ? "tool-options-shape-btn is-active"
                            : "tool-options-shape-btn"
                        }
                        disabled={loading || workBusy}
                        onClick={() => setStrokeDrawStyle("brush")}
                      >
                        Brush
                      </button>
                    </div>
                    <label
                      className="tool-options-range-label"
                      style={{ marginTop: "0.35rem" }}
                    >
                      <span>Stroke mode</span>
                      <select
                        value={drawStrokeMode}
                        onChange={(ev) =>
                          setDrawStrokeMode(ev.target.value as DrawStrokeModeApi)
                        }
                        disabled={loading || workBusy}
                      >
                        <option value="line">Line</option>
                        <option value="spray">Spray</option>
                        <option value="plane">Plane</option>
                        <option value="precise">Precise</option>
                        <option value="circle">Circle</option>
                        <option value="cuboid">Cuboid</option>
                        <option value="cylinder">Cylinder</option>
                        <option value="polygonHull">Polygon hull</option>
                        <option value="polygon">Polygon</option>
                        <option value="fill">Fill (paint)</option>
                      </select>
                    </label>
                    {drawStrokeMode === "plane" ? (
                      <label
                        className="tool-options-range-label"
                        style={{ marginTop: "0.25rem" }}
                      >
                        <span>Plane axis</span>
                        <select
                          value={planeAxis}
                          onChange={(ev) =>
                            setPlaneAxis(ev.target.value as PlaneAxisApi)
                          }
                          disabled={loading || workBusy}
                        >
                          <option value="auto">Auto (face)</option>
                          <option value="x">X</option>
                          <option value="y">Y</option>
                          <option value="z">Z</option>
                        </select>
                      </label>
                    ) : null}
                    {(drawStrokeMode === "polygon" ||
                      drawStrokeMode === "polygonHull") ? (
                      <div style={{ marginTop: "0.35rem" }}>
                        <p style={{ margin: "0 0 0.35rem", fontSize: "0.85rem", opacity: 0.9 }}>
                          Vertices: {strokePolygonVerts.length}. Click to add corners; Apply with three or more.
                        </p>
                        <div className="tool-options-shape-row" style={{ flexWrap: "wrap" }}>
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
                            onClick={() => applyPolygonStrokeFill()}
                          >
                            Apply
                          </button>
                        </div>
                      </div>
                    ) : null}
                    {drawStrokeMode === "circle" ? (
                      <p style={{ margin: "0.35rem 0 0", fontSize: "0.85rem", opacity: 0.9 }}>
                        Circle: first click center, second click edge.
                      </p>
                    ) : null}
                    {drawStrokeMode === "cuboid" ? (
                      <p style={{ margin: "0.35rem 0 0", fontSize: "0.85rem", opacity: 0.9 }}>
                        Cuboid: two opposite corners.
                      </p>
                    ) : null}
                    {drawStrokeMode === "cylinder" ? (
                      <p style={{ margin: "0.35rem 0 0", fontSize: "0.85rem", opacity: 0.9 }}>
                        Cylinder: axis start then end (uses brush radius).
                      </p>
                    ) : null}
                    {sprayDensity > 0 ? (
                      <label className="tool-options-range-label">
                        <span>Spray density</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.02}
                          value={sprayDensity}
                          onChange={(ev) =>
                            setSprayDensity(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                    ) : null}
                  </div>
                  <div className="tool-options-section">
                    <div className="tool-options-heading">Color</div>
                    <label className="tool-options-color-label">
                      <span>Color</span>
                      <input
                        type="color"
                        value={`#${activeColor.toString(16).padStart(6, "0")}`}
                        onChange={(ev) => {
                          const h = ev.target.value.slice(1);
                          const n = Number.parseInt(h, 16);
                          if (!Number.isNaN(n)) setActiveColor(n);
                        }}
                        disabled={loading || workBusy}
                      />
                    </label>
                  </div>
                  {sculptStrokeMode === "terrain" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">Terrain</div>
                      <label className="tool-options-range-label">
                        <span>Base Y</span>
                        <input
                          type="range"
                          min={-64}
                          max={64}
                          value={terrainBaseY}
                          onChange={(ev) =>
                            setTerrainBaseY(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      <label className="tool-options-range-label">
                        <span>Strength</span>
                        <input
                          type="range"
                          min={0}
                          max={32}
                          value={terrainStrength}
                          onChange={(ev) =>
                            setTerrainStrength(Number(ev.target.value))
                          }
                          disabled={loading || workBusy}
                        />
                      </label>
                      {terrainSculptOp === "smooth" ? (
                        <label className="tool-options-range-label">
                          <span>Smooth radius</span>
                          <input
                            type="range"
                            min={0}
                            max={8}
                            value={terrainSmoothRadius}
                            onChange={(ev) =>
                              setTerrainSmoothRadius(Number(ev.target.value))
                            }
                            disabled={loading || workBusy}
                          />
                        </label>
                      ) : null}
                    </div>
                  ) : null}
                  {sculptStrokeMode === "smooth" ? (
                    <div className="tool-options-section">
                      <div className="tool-options-heading">Smooth</div>
                      <label className="tool-options-range-label">
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
                <div className="tool-options-section">
                  <div className="tool-options-heading">Generator</div>
                  {generatorKind === "sphere" ? (
                    <label className="tool-options-range-label">
                      <span>Radius</span>
                      <input
                        type="range"
                        min={1}
                        max={32}
                        value={generatorSphereRadius}
                        onChange={(ev) =>
                          setGeneratorSphereRadius(Number(ev.target.value))
                        }
                        disabled={loading || workBusy}
                      />
                    </label>
                  ) : null}
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
                    <label className="tool-options-range-label">
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
                    </label>
                  ) : null}
                  {generatorKind === "squishy" ? (
                    <>
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
                      <p style={{ fontSize: "0.85rem", opacity: 0.85, margin: "0.25rem 0" }}>
                        Metaballs: {squishyBallCount}. Click viewport to add/pick/delete; Commit voxelizes the combined field.
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
                      <label className="tool-options-range-label" style={{ flexDirection: "row", alignItems: "center", gap: "0.5rem" }}>
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
                      <label className="tool-options-range-label" style={{ flexDirection: "row", alignItems: "center", gap: "0.5rem" }}>
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
                              .catch(() => { });
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
                              .catch(() => { });
                          }}
                        >
                          Clear session
                        </button>
                      </div>
                    </>
                  ) : null}
                </div>
              ) : null}
              {toolsPane === "mood" ? (
                <div className="tool-options-section">
                  <div className="tool-options-heading">Mood</div>
                  <label className="tool-options-range-label">
                    <span>Grain</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={moodGrain}
                      onChange={(ev) =>
                        setMoodGrain(Number(ev.target.value))
                      }
                      disabled={loading || workBusy}
                    />
                  </label>
                  <label className="tool-options-range-label">
                    <span>Vignette</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={moodVignette}
                      onChange={(ev) =>
                        setMoodVignette(Number(ev.target.value))
                      }
                      disabled={loading || workBusy}
                    />
                  </label>
                  <label className="tool-options-range-label">
                    <span>Distance tint</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={moodDistanceTint}
                      onChange={(ev) =>
                        setMoodDistanceTint(Number(ev.target.value))
                      }
                      disabled={loading || workBusy}
                    />
                  </label>
                  <label className="tool-options-range-label">
                    <span>Atmosphere</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={moodAtmosphere}
                      onChange={(ev) =>
                        setMoodAtmosphere(Number(ev.target.value))
                      }
                      disabled={loading || workBusy}
                    />
                  </label>
                  <label className="tool-options-range-label">
                    <span>Sun shafts</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={moodSunShafts}
                      onChange={(ev) =>
                        setMoodSunShafts(Number(ev.target.value))
                      }
                      disabled={loading || workBusy}
                    />
                  </label>
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
                    <div className="viewport-empty-last-title">Continue</div>
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
                        onClick={reopenLastProject}
                        aria-describedby={
                          lastProjectBlurb ? "viewport-empty-last-desc" : undefined
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
                    void invoke("open_voxelle_dialog").catch(() => { })
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
                <button type="button" onClick={sendChat} disabled={!collabActive}>
                  Send
                </button>
              </div>
            </div>
          ) : null}
        </div>
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
                    <p
                      className={`inspector-session-status${hostWsUrl ? " is-live" : ""}`}
                      role="status"
                      aria-live="polite"
                    >
                      {hostWsUrl
                        ? "Guests can use the link in the status bar."
                        : "You're a guest."}
                    </p>
                    {hostWsUrl ? (
                      <>
                        <p className="collab-hint inspector-collab-hint">
                          Nearby: <code>{hostWsUrl}</code>
                        </p>
                        {prefsEnableUpnp && natPending ? (
                          <p
                            className="collab-hint collab-hint-muted inspector-collab-hint"
                            role="status"
                          >
                            Checking your router…
                          </p>
                        ) : null}
                        {hostWanUrl ? (
                          <p className="collab-hint inspector-collab-hint">
                            Internet: <code>{hostWanUrl}</code>
                          </p>
                        ) : null}
                        {natError ? (
                          <p
                            className="collab-hint collab-hint-warn inspector-collab-hint"
                            role="alert"
                          >
                            {natError} You can forward port {hostPort} in your
                            router settings. Some networks won&apos;t allow guests
                            over the internet.
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
      </div>
      <footer className="app-status-bar" role="contentinfo">
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
              title={
                hostingCopied
                  ? "Copied"
                  : "Copy invite link"
              }
            >
              {hostingCopied
                ? "Copied invite link"
                : `Hosting · ${roster.length} ${roster.length === 1 ? "person" : "people"
                }`}
            </button>
          ) : null}
        </div>
        {showFpsCounter ? (
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
    </div>
  );
}

export default App;
