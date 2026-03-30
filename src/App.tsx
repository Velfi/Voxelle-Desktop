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

function basename(path: string): string {
  const n = path.replace(/\\/g, "/");
  const i = n.lastIndexOf("/");
  return i >= 0 ? n.slice(i + 1) : n;
}

/** One-line hint for which copy will load (newest of saved file vs autosave). */
function lastProjectReopenBlurb(info: LastSessionInfo): string | null {
  if (!info.lastDocumentPath) return null;
  if (!info.documentExists && info.autosaveExists) {
    return "Saved file not found — restoring from backup.";
  }
  if (
    info.documentExists &&
    info.autosaveExists &&
    info.autosaveNewerThanDocument
  ) {
    return "Latest copy is the autosave (newer than the file on disk).";
  }
  if (
    info.documentExists &&
    info.autosaveExists &&
    !info.autosaveNewerThanDocument
  ) {
    return "Latest copy is the saved file.";
  }
  return null;
}

type InteractionMode = "navigate" | "add" | "remove";

function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
  /** Physical pixel size of the GPU surface; kept in sync with Rust (may differ slightly from CSS×dpr). */
  const viewportPhysRef = useRef({ w: 0, h: 0 });
  const lastRef = useRef({ x: 0, y: 0 });
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
  const loadingRef = useRef(false);
  const interactionBlockedRef = useRef(false);
  const pendingJoinUrlRef = useRef<string | null>(null);
  const collabActiveMenuRef = useRef(false);
  const startHostMenuRef = useRef<() => void>(() => {});
  const leaveSessionMenuRef = useRef<() => void>(() => {});
  const [interactionMode, setInteractionMode] =
    useState<InteractionMode>("navigate");
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

  const [joinModalOpen, setJoinModalOpen] = useState(false);
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
      .catch(() => {});
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
      listen<string>("voxelle-loaded", (e) => {
        setLoadError(null);
        setPathLabel(e.payload);
        setLoading(false);
        setLoadProgress(1);
        setLoadPhase("");
        refreshSceneObjects();
      }),
      listen<string>("voxelle-load-error", (e) => {
        setLoadError(e.payload);
        setLoading(false);
        setLoadPhase("");
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
      listen<string>("collab-error", (e) => {
        pendingJoinUrlRef.current = null;
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
          text: `You were removed from the session: ${msg}`,
          tone: "alert",
        });
        clearCollabSessionUi();
      }),
      listen("voxelle-check-updates", async () => {
        try {
          const update = await check();
          if (!update) {
            window.alert("You're on the latest version.");
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
          window.alert(
            e instanceof Error ? e.message : `Update failed: ${String(e)}`,
          );
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
  }, [sendResize, sidebarExpanded, rightSidebarExpanded]);

  useEffect(() => {
    interactionModeRef.current = interactionMode;
  }, [interactionMode]);

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
    const p = loadPreferences();
    void invoke("set_tone_mapping", {
      mode: toneMappingToGpuMode(p.toneMapping),
    }).catch(() => {});
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(LS_RENDERING_MODE) as RenderingMode | null;
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
      }
      if (meta && e.key === "s") {
        e.preventDefault();
        void invoke("save_voxelle").catch(() => {
          void invoke("save_voxelle_as").catch(() => {});
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const clearPreview = useCallback(() => {
    void invoke("sync_preview_input", {
      args: { x: -1, y: 0, mode: "navigate" },
    }).catch(() => {});
  }, []);

  useEffect(() => {
    void invoke("sync_preview_input", {
      args: { x: -1, y: 0, mode: interactionMode },
    }).catch(() => {});
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
    const navigate = mode === "navigate";
    const forceCamera =
      middleButton ||
      e.shiftKey ||
      (mode === "add" && e.button !== 0) ||
      (mode === "remove" && e.button !== 0);

    let hitSolid = false;
    if (
      !loading &&
      !workBusy &&
      !forceCamera &&
      !navigate &&
      (mode === "add" || mode === "remove") &&
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

    if (gestureRef.current.mode === "camera") {
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
    if (
      !probingRef.current &&
      (interactionModeRef.current === "add" ||
        interactionModeRef.current === "remove") &&
      !interactionBlockedRef.current
    ) {
      const m = interactionModeRef.current;
      void invoke("sync_preview_input", {
        args: { x: px, y: py, mode: m },
      }).catch(() => {});
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
      moved < 5 &&
      !e.shiftKey &&
      e.button === 0
    ) {
      const { x, y } = clientToViewportPhysical(e);
      const m = interactionModeRef.current;
      if (m === "add") {
        void invoke("voxel_edit_at_screen", {
          args: { x, y, add: true },
        }).catch(() => {});
      } else if (m === "remove") {
        void invoke("voxel_edit_at_screen", {
          args: { x, y, add: false },
        }).catch(() => {});
      }
    }

    if (isThisPointer && g?.mode === "camera") {
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
          `Could not start hosting.\n\n${msg}\n\nIf you are already in a session, leave it first. Otherwise try another port.`,
        );
      });
  };

  const joinSession = (urlOverride?: string) => {
    if (collabActive) return;
    setCollabBanner(null);
    setLoadError(null);
    const u = (urlOverride ?? joinUrl).trim();
    if (!u) {
      setLoadError("Enter a server URL (ws://…).");
      return;
    }
    setJoinUrl(u);
    pendingJoinUrlRef.current = u;
    const rgb = hexToRgb(normalizeCollabAccentColor(accentColor));
    void invoke("collab_join", {
      url: u,
      displayName: normalizeCollabDisplayName(displayName),
      colorRgb: rgb,
    });
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
      if (collabActive) return `${base} · Collaborating`;
      return base;
    }
    return "No file open";
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

  return (
    <div className="app">
      <div className="app-main">
        <aside
          className={
            sidebarExpanded
              ? "app-sidebar is-expanded"
              : "app-sidebar is-collapsed"
          }
          aria-label="Tools"
        >
          <div className="sidebar-header">
            <button
              type="button"
              className="sidebar-expand-toggle"
              onClick={() => setSidebarExpanded((v) => !v)}
              aria-expanded={sidebarExpanded}
              title={
                sidebarExpanded ? "Collapse tool sidebar" : "Expand tool sidebar"
              }
            >
              <span className="sidebar-expand-toggle-icon" aria-hidden>
                {sidebarExpanded ? "«" : "»"}
              </span>
              {sidebarExpanded ? (
                <span className="sidebar-expand-toggle-label">Tools</span>
              ) : null}
            </button>
          </div>
          <div className="sidebar-scroll">
            {sidebarExpanded ? (
              <div className="sidebar-section-label">Mode</div>
            ) : null}
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
                title="Orbit, pan, dolly — clicks do not edit voxels"
              >
                <span className="sidebar-mode-icon" aria-hidden>
                  ✋
                </span>
                {sidebarExpanded ? (
                  <span className="sidebar-mode-label">Navigate</span>
                ) : null}
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
                title="Click a face to place a voxel (green preview)"
              >
                <span className="sidebar-mode-icon" aria-hidden>
                  👇
                </span>
                {sidebarExpanded ? (
                  <span className="sidebar-mode-label">Add</span>
                ) : null}
              </button>
              <button
                type="button"
                className={
                  interactionMode === "remove"
                    ? "sidebar-mode-btn is-active"
                    : "sidebar-mode-btn"
                }
                disabled={loading || workBusy}
                onClick={() => setInteractionMode("remove")}
                title="Click a voxel to remove it (red preview)"
              >
                <span className="sidebar-mode-icon" aria-hidden>
                  👊
                </span>
                {sidebarExpanded ? (
                  <span className="sidebar-mode-label">Remove</span>
                ) : null}
              </button>
            </div>
            {sidebarExpanded ? (
              <div
                className="sidebar-expanded-slot"
                aria-label="Additional tools"
              >
                {/* Web parity: palette, layers, etc. */}
              </div>
            ) : null}
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
                      ? "Already in a session — leave before joining another"
                      : "Connect to a host by URL"
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
                        ? "Stop hosting and end the session for guests"
                        : "Leave the collaboration session"
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
              <div key={t.id} className="chat-toast" role="status">
                <span className="chat-toast-text">{t.text}</span>
                <button
                  type="button"
                  className="chat-toast-dismiss"
                  aria-label="Dismiss notification"
                  onClick={() =>
                    setChatToasts((prev) => prev.filter((x) => x.id !== t.id))
                  }
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
                  collabActive ? "Message…" : "Join or host to use chat"
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
                            show
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
                        ? "Hosting — guests can join using the link in the status bar."
                        : "Connected as a guest."}
                    </p>
                    {hostWsUrl ? (
                      <>
                        <p className="collab-hint inspector-collab-hint">
                          On your network: <code>{hostWsUrl}</code>
                        </p>
                        {prefsEnableUpnp && natPending ? (
                          <p
                            className="collab-hint collab-hint-muted inspector-collab-hint"
                            role="status"
                          >
                            Contacting your router for internet sharing…
                          </p>
                        ) : null}
                        {hostWanUrl ? (
                          <p className="collab-hint inspector-collab-hint">
                            Internet (UPnP): <code>{hostWanUrl}</code>
                          </p>
                        ) : null}
                        {natError ? (
                          <p
                            className="collab-hint collab-hint-warn inspector-collab-hint"
                            role="alert"
                          >
                            {natError} If UPnP is disabled on the router, forward
                            TCP port {hostPort} manually. CGNAT can block internet
                            guests even when UPnP succeeds.
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
                            title="Click to match their camera"
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
                                edit
                              </label>
                              <button
                                type="button"
                                className="collab-kick"
                                title="Remove from session"
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
                    <button
                      type="button"
                      className="inspector-ping-origin"
                      onClick={() =>
                        void invoke("collab_send_ping", { x: 0, y: 0, z: 0 })
                      }
                    >
                      Ping origin
                    </button>
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
                  ? "Join address copied"
                  : "Copy join address (internet link if UPnP succeeded, else LAN)"
              }
            >
              {hostingCopied
                ? "Copied join address"
                : `Hosting · ${roster.length} ${
                    roster.length === 1 ? "user" : "users"
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
            <h3>New grid</h3>
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
