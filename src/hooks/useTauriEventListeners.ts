import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import { loadPreferences } from "../preferences";
import { rememberJoinedUrl } from "../joinRecent";
import { defaultMoodState, moodWith } from "../types";
import type {
  MoodState,
  RosterEntry,
  ChatToast,
  InteractionMode,
  ViewportCursorDebugPayload,
  ViewportCursorDebugScreen,
} from "../types";
import {
  LS_RENDERING_MODE,
  LS_VIEWPORT_CURSOR_DEBUG,
  CHAT_TOAST_CAP,
  PING_HUD_MS,
  userFacingUpdaterError,
  playPingSound,
} from "../constants";
import type { BubbleInfo } from "../SpeechBubbleOverlay";

export type SelectionCombineModeApi = "replace" | "add" | "subtract" | "intersect";

export interface UseTauriEventListenersParams {
  viewportRef: React.RefObject<HTMLDivElement | null>;
  sendResize: () => void;
  refreshSceneObjects: () => void;

  /* Refs read inside listeners */
  viewportPhysRef: React.MutableRefObject<{ w: number; h: number }>;
  surfacePhysRef: React.MutableRefObject<{ w: number; h: number }>;
  fillOperationPendingRef: React.MutableRefObject<boolean>;
  collabActiveRef: React.RefObject<boolean>;
  chatPanelOpenRef: React.RefObject<boolean>;
  localPeerIdRef: React.RefObject<number>;
  chatToastIdRef: React.MutableRefObject<number>;
  pendingJoinUrlRef: React.MutableRefObject<string | null>;
  collabActiveMenuRef: React.RefObject<boolean>;
  startHostMenuRef: React.RefObject<() => void>;
  leaveSessionMenuRef: React.RefObject<() => void>;
  viewportCursorDebugScreenRef: React.MutableRefObject<ViewportCursorDebugScreen | null>;
  pingHudRef: React.MutableRefObject<{
    name: string;
    wx: number;
    wy: number;
    wz: number;
    until: number;
    emoji?: string;
  } | null>;

  /* State setters */
  setLoadError: React.Dispatch<React.SetStateAction<string | null>>;
  setCollabBanner: React.Dispatch<
    React.SetStateAction<{ text: string; tone: "info" | "alert" } | null>
  >;
  setStartScreenLogoLoaded: React.Dispatch<React.SetStateAction<boolean>>;
  setPathLabel: React.Dispatch<React.SetStateAction<string>>;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setLoadProgress: React.Dispatch<React.SetStateAction<number>>;
  setLoadPhase: React.Dispatch<React.SetStateAction<string>>;
  setSpeechBubbles: React.Dispatch<React.SetStateAction<BubbleInfo[]>>;
  setWorkProgress: React.Dispatch<React.SetStateAction<number>>;
  setWorkPhase: React.Dispatch<React.SetStateAction<string>>;
  setWorkBusy: React.Dispatch<React.SetStateAction<boolean>>;
  setFillOperationPending: React.Dispatch<React.SetStateAction<boolean>>;
  setLogoLightControlsVisible: React.Dispatch<React.SetStateAction<boolean>>;
  setMood: React.Dispatch<React.SetStateAction<MoodState>>;
  setFpsDisplayed: React.Dispatch<React.SetStateAction<number>>;
  setPingMs: React.Dispatch<React.SetStateAction<number | null>>;
  setNewProjectOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setJoinModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setChatPanelOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setPreferencesOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setStampBookOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setPingHudTick: React.Dispatch<React.SetStateAction<number>>;
  setChatLines: React.Dispatch<React.SetStateAction<string[]>>;
  setChatToasts: React.Dispatch<React.SetStateAction<ChatToast[]>>;
  setCollabActive: React.Dispatch<React.SetStateAction<boolean>>;
  setCollabJoinPending: React.Dispatch<React.SetStateAction<boolean>>;
  setLocalPeerId: React.Dispatch<React.SetStateAction<number>>;
  setRoster: React.Dispatch<React.SetStateAction<RosterEntry[]>>;
  setNatPending: React.Dispatch<React.SetStateAction<boolean>>;
  setNatError: React.Dispatch<React.SetStateAction<string | null>>;
  setHostWsUrl: React.Dispatch<React.SetStateAction<string | null>>;
  setHostWanUrl: React.Dispatch<React.SetStateAction<string | null>>;
  setHostingCopied: React.Dispatch<React.SetStateAction<boolean>>;
  setChatInput: React.Dispatch<React.SetStateAction<string>>;
  setInteractionMode: React.Dispatch<React.SetStateAction<InteractionMode>>;
  setMatchMaterialSelectColor: React.Dispatch<React.SetStateAction<boolean>>;
  setViewportCursorDebugEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  setViewportCursorDebugJs: React.Dispatch<React.SetStateAction<{ nx: number; ny: number } | null>>;
  setViewportCursorDebugRust: React.Dispatch<
    React.SetStateAction<ViewportCursorDebugPayload | null>
  >;
  setViewportCursorDebugScreen: React.Dispatch<
    React.SetStateAction<ViewportCursorDebugScreen | null>
  >;
  setHideUI: React.Dispatch<React.SetStateAction<boolean>>;
  setSelectionCount: React.Dispatch<React.SetStateAction<number>>;
  setSelectionCombineMode: React.Dispatch<React.SetStateAction<SelectionCombineModeApi>>;
  setRotateDialogOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setScaleDialogOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

/**
 * Registers all Tauri event listeners in a single useEffect.
 * Also sets up the ResizeObserver for the viewport.
 */
export function useTauriEventListeners(params: UseTauriEventListenersParams): void {
  const {
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
  } = params;

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
        startHostMenuRef.current?.();
      }),
      listen("voxelle-collab-join-session", () => {
        setJoinModalOpen(true);
      }),
      listen("voxelle-collab-leave-session", () => {
        if (!collabActiveMenuRef.current) return;
        leaveSessionMenuRef.current?.();
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
        if (m === "greedy" || m === "marchingCubes" || m === "dualContour") {
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

  useEffect(() => {
    if (!loadPreferences().autoCheckUpdates) return;
    void (async () => {
      try {
        const update = await check();
        if (!update) return;
        const ok = await invoke<boolean>("confirm_app_update_dialog", {
          message: `Download and install Voxelle Desktop ${update.version}?`,
          title: "Update available",
        });
        if (!ok) return;
        await update.downloadAndInstall();
        await relaunch();
      } catch {
        /* silently ignore startup check failures */
      }
    })();
  }, []);
}
