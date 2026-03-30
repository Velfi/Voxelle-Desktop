import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "./App.css";

/** Desktop viewer: cap new-project grid edge length (web allows larger). */
const MAX_GRID_SIZE = 256;

const LS_NAME = "voxelleCollabDisplayName";
const LS_COLOR = "voxelleCollabColor";
const LS_AUTOSAVE = "voxelleAutosaveSecs";

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

function basename(path: string): string {
  const n = path.replace(/\\/g, "/");
  const i = n.lastIndexOf("/");
  return i >= 0 ? n.slice(i + 1) : n;
}

type InteractionMode = "navigate" | "add" | "remove";

function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
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
  const [interactionMode, setInteractionMode] =
    useState<InteractionMode>("navigate");
  const [pathLabel, setPathLabel] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [fpsDisplayed, setFpsDisplayed] = useState(0);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [newGridSize, setNewGridSize] = useState(32);
  const [newGridShape, setNewGridShape] = useState<StartShape>("circle");

  const [collabOpen, setCollabOpen] = useState(false);
  const [hostWsUrl, setHostWsUrl] = useState<string | null>(null);
  const [joinUrl, setJoinUrl] = useState("ws://127.0.0.1:27300");
  const [displayName, setDisplayName] = useState(() =>
    typeof localStorage !== "undefined"
      ? localStorage.getItem(LS_NAME) || "Artist"
      : "Artist",
  );
  const [accentColor, setAccentColor] = useState(() =>
    typeof localStorage !== "undefined"
      ? localStorage.getItem(LS_COLOR) || "#6699cc"
      : "#6699cc",
  );
  const [roster, setRoster] = useState<RosterEntry[]>([]);
  const [chatLines, setChatLines] = useState<string[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [hostPort, setHostPort] = useState(27300);
  const [autosaveSecs, setAutosaveSecs] = useState(() => {
    if (typeof localStorage === "undefined") return 120;
    const v = localStorage.getItem(LS_AUTOSAVE);
    return v ? Number(v) || 120 : 120;
  });
  const [collabActive, setCollabActive] = useState(false);
  /** Set when hosting or after welcome; 0 when solo. */
  const [localPeerId, setLocalPeerId] = useState(0);

  const hexToRgb = (hex: string): number => {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return n & 0xffffff;
  };

  const sendResize = useCallback(() => {
    const el = viewportRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(el.clientWidth * dpr));
    const height = Math.max(1, Math.floor(el.clientHeight * dpr));
    void invoke("viewer_resize", { width, height });
  }, []);

  useEffect(() => {
    sendResize();
    const ro = new ResizeObserver(() => sendResize());
    const el = viewportRef.current;
    if (el) ro.observe(el);
    const unlistenStart = listen<string>("voxelle-load-start", (e) => {
      setLoadError(null);
      setPathLabel(e.payload);
      setLoading(true);
      setLoadProgress(0);
    });
    const unlistenProgress = listen<number>("voxelle-load-progress", (e) => {
      setLoadProgress(e.payload);
    });
    const unlistenLoaded = listen<string>("voxelle-loaded", (e) => {
      setLoadError(null);
      setPathLabel(e.payload);
      setLoading(false);
      setLoadProgress(1);
    });
    const unlistenErr = listen<string>("voxelle-load-error", (e) => {
      setLoadError(e.payload);
      setLoading(false);
    });
    const unlistenFps = listen<number>("viewport-fps", (e) => {
      setFpsDisplayed(e.payload);
    });
    const unlistenNewProject = listen("voxelle-open-new-project", () => {
      setNewProjectOpen(true);
    });
    const unlistenToggleCollab = listen("voxelle-toggle-collab", () => {
      setCollabOpen((o) => !o);
    });
    const unlistenCollabChat = listen<string>("collab-chat", (e) => {
      try {
        const j = JSON.parse(e.payload) as {
          displayName?: string;
          display_name?: string;
          text?: string;
        };
        const who = j.displayName ?? j.display_name ?? "?";
        const line = `${who}: ${j.text ?? ""}`;
        setChatLines((prev) => [...prev.slice(-80), line]);
      } catch {
        setChatLines((prev) => [...prev.slice(-80), e.payload]);
      }
    });
    const unlistenCollabJoined = listen("collab-joined", () => {
      setCollabActive(true);
    });
    const unlistenCollabLocalPeer = listen<number>("collab-local-peer", (e) => {
      setLocalPeerId(typeof e.payload === "number" ? e.payload : 0);
    });
    const unlistenCollabRoster = listen<string>("collab-roster", (e) => {
      try {
        const arr = JSON.parse(e.payload) as RosterEntry[];
        setRoster(arr);
      } catch {
        /* ignore */
      }
    });
    const unlistenCollabErr = listen<string>("collab-error", (e) => {
      setLoadError(e.payload);
    });
    return () => {
      ro.disconnect();
      void unlistenStart.then((fn) => fn());
      void unlistenProgress.then((fn) => fn());
      void unlistenLoaded.then((fn) => fn());
      void unlistenErr.then((fn) => fn());
      void unlistenFps.then((fn) => fn());
      void unlistenNewProject.then((fn) => fn());
      void unlistenToggleCollab.then((fn) => fn());
      void unlistenCollabChat.then((fn) => fn());
      void unlistenCollabJoined.then((fn) => fn());
      void unlistenCollabLocalPeer.then((fn) => fn());
      void unlistenCollabRoster.then((fn) => fn());
      void unlistenCollabErr.then((fn) => fn());
    };
  }, [sendResize]);

  useEffect(() => {
    interactionModeRef.current = interactionMode;
  }, [interactionMode]);

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    localStorage.setItem(LS_NAME, displayName);
  }, [displayName]);

  useEffect(() => {
    localStorage.setItem(LS_COLOR, accentColor);
  }, [accentColor]);

  useEffect(() => {
    localStorage.setItem(LS_AUTOSAVE, String(autosaveSecs));
    void invoke("set_autosave_interval_secs", { secs: autosaveSecs }).catch(
      () => {},
    );
  }, [autosaveSecs]);

  useEffect(() => {
    void invoke("get_autosave_interval_secs").then((s) => {
      if (typeof s === "number" && s > 0) setAutosaveSecs(s);
    });
  }, []);

  useEffect(() => {
    if (!collabActive) return;
    const id = window.setInterval(() => {
      void invoke("collab_push_camera").catch(() => {});
    }, 150);
    return () => clearInterval(id);
  }, [collabActive]);

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
    if (loading) {
      clearPreview();
    }
  }, [loading, clearPreview]);

  const clientToViewportPhysical = useCallback((e: React.PointerEvent) => {
    const el = viewportRef.current;
    const dpr = window.devicePixelRatio || 1;
    if (!el) {
      return { x: e.clientX * dpr, y: e.clientY * dpr };
    }
    const rect = el.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * dpr,
      y: (e.clientY - rect.top) * dpr,
    };
  }, []);

  const createNewProject = useCallback(() => {
    if (loading) return;
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
  }, [loading, newGridSize, newGridShape]);

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
      !loadingRef.current
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
    const dx = e.clientX - lastRef.current.x;
    const dy = e.clientY - lastRef.current.y;
    lastRef.current = { x: e.clientX, y: e.clientY };
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
    void invoke("collab_host_start", { port: hostPort }).then((url) => {
      setHostWsUrl(url as string);
      setCollabActive(true);
    });
  };

  const joinSession = () => {
    const rgb = hexToRgb(accentColor);
    void invoke("collab_join", {
      url: joinUrl,
      displayName,
      colorRgb: rgb,
    }).then(() => setCollabActive(true));
  };

  const leaveSession = () => {
    void invoke("collab_leave").then(() => {
      setCollabActive(false);
      setHostWsUrl(null);
      setRoster([]);
      setLocalPeerId(0);
    });
  };

  const amLeader = roster.some(
    (r) => r.peerId === localPeerId && r.isLeader,
  );

  const sendChat = () => {
    const t = chatInput.trim();
    if (!t) return;
    void invoke("collab_send_chat", { text: t }).catch(() => {});
    setChatInput("");
  };

  const onRosterDoubleClick = (peerId: number) => {
    void invoke("collab_snap_camera", { peerId }).catch(() => {});
  };

  const setCanEdit = (peerId: number, canEdit: boolean) => {
    void invoke("collab_set_can_edit", { targetPeer: peerId, canEdit }).catch(
      () => {},
    );
  };

  return (
    <div className="app">
      <header className="app-chrome">
        <div className="fps-counter" role="status" aria-live="polite">
          {fpsDisplayed} FPS
        </div>
        <div className="toolbar-actions">
          <div
            className="toolbar-mode-group"
            role="group"
            aria-label="Interaction mode"
          >
            <button
              type="button"
              className={
                interactionMode === "navigate"
                  ? "toolbar-mode-btn is-active"
                  : "toolbar-mode-btn"
              }
              disabled={loading}
              onClick={() => setInteractionMode("navigate")}
              title="Orbit, pan, dolly — clicks do not edit voxels"
            >
              ✋
            </button>
            <button
              type="button"
              className={
                interactionMode === "add"
                  ? "toolbar-mode-btn is-active"
                  : "toolbar-mode-btn"
              }
              disabled={loading}
              onClick={() => setInteractionMode("add")}
              title="Click a face to place a voxel (green preview)"
            >
              👇
            </button>
            <button
              type="button"
              className={
                interactionMode === "remove"
                  ? "toolbar-mode-btn is-active"
                  : "toolbar-mode-btn"
              }
              disabled={loading}
              onClick={() => setInteractionMode("remove")}
              title="Click a voxel to remove it (red preview)"
            >
              👊
            </button>
          </div>
        </div>
      </header>
      <div className="viewport-wrap">
        {loading ? (
          <div className="load-bar" aria-hidden>
            <div
              className="load-bar-fill"
              style={{
                width: `${Math.round(Math.min(1, Math.max(0, loadProgress)) * 100)}%`,
              }}
            />
          </div>
        ) : null}
        <div
          ref={viewportRef}
          className="viewport"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerLeave}
          onContextMenu={(ev) => ev.preventDefault()}
          onWheel={onWheel}
          role="application"
          aria-label="3D viewport"
        />
        {loadError ? (
          <div className="viewport-error" role="alert" title={loadError}>
            {loadError}
          </div>
        ) : null}
        {collabOpen ? (
          <aside className="collab-panel" aria-label="Collaboration">
            <h3 className="collab-panel-title">Session</h3>
            <label className="modal-field">
              Display name
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                maxLength={32}
              />
            </label>
            <label className="modal-field">
              Accent color
              <input
                type="color"
                value={accentColor}
                onChange={(e) => setAccentColor(e.target.value)}
              />
            </label>
            <label className="modal-field">
              Autosave interval (seconds, 0 = off)
              <input
                type="number"
                min={0}
                max={3600}
                value={autosaveSecs}
                onChange={(e) =>
                  setAutosaveSecs(Math.max(0, Number(e.target.value) || 0))
                }
              />
            </label>
            <div className="collab-row">
              <label className="modal-field collab-grow">
                Port
                <input
                  type="number"
                  value={hostPort}
                  onChange={(e) =>
                    setHostPort(Math.max(1, Number(e.target.value) || 27300))
                  }
                />
              </label>
              <button type="button" onClick={startHost}>
                Host
              </button>
            </div>
            {hostWsUrl ? (
              <p className="collab-hint">
                Share: <code>{hostWsUrl}</code>
              </p>
            ) : null}
            <div className="collab-row">
              <label className="modal-field collab-grow">
                Join URL
                <input
                  type="text"
                  value={joinUrl}
                  onChange={(e) => setJoinUrl(e.target.value)}
                />
              </label>
              <button type="button" onClick={joinSession}>
                Join
              </button>
            </div>
            <button type="button" onClick={leaveSession}>
              Leave session
            </button>
            <h4 className="collab-subtitle">Roster</h4>
            <ul className="collab-roster">
              {roster.map((r) => (
                <li key={r.peerId}>
                  <button
                    type="button"
                    className="collab-roster-name"
                    onDoubleClick={() => onRosterDoubleClick(r.peerId)}
                    title="Double-click to match camera"
                  >
                    <span
                      className="collab-swatch"
                      style={{
                        background: `#${(r.colorRgb & 0xffffff).toString(16).padStart(6, "0")}`,
                      }}
                    />
                    {r.displayName}
                    {r.isLeader ? " (leader)" : ""}
                  </button>
                  {!r.isLeader && amLeader ? (
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
                  ) : null}
                </li>
              ))}
            </ul>
            <h4 className="collab-subtitle">Chat</h4>
            <div className="collab-chat-log" role="log">
              {chatLines.map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
            <div className="collab-row">
              <input
                className="collab-grow"
                type="text"
                value={chatInput}
                placeholder="Message…"
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendChat()}
              />
              <button type="button" onClick={sendChat}>
                Send
              </button>
            </div>
            <button
              type="button"
              onClick={() => void invoke("collab_send_ping", { x: 0, y: 0, z: 0 })}
            >
              Ping origin
            </button>
          </aside>
        ) : null}
      </div>
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
