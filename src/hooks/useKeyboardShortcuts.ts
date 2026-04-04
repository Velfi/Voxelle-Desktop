import { useCallback, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { RADIAL_HOLD_MS } from "../RadialPingMenu";
import { loadPreferences } from "../preferences";
import { PING_HUD_MS, playPingSound } from "../constants";

export interface UseKeyboardShortcutsParams {
  /** Modal / overlay flags — keyboard shortcuts are suppressed when any is open. */
  preferencesOpen: boolean;
  stampBookOpen: boolean;
  joinModalOpen: boolean;
  newProjectOpen: boolean;
  collabJoinPending: boolean;
  loading: boolean;
  workBusy: boolean;
  fillOperationPending: boolean;
  selectionCount: number;

  /* Refs shared with App */
  lastViewportPickNormRef: React.RefObject<{ nx: number; ny: number } | null>;
  pendingPingRef: React.MutableRefObject<{ nx: number; ny: number } | null>;
  radialHoldTimerRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>;
  lastCursorScreenRef: React.RefObject<{ x: number; y: number }>;
  fillOperationPendingRef: React.RefObject<boolean>;
  workPhaseRef: React.RefObject<string>;
  pingHudRef: React.MutableRefObject<{
    name: string;
    wx: number;
    wy: number;
    wz: number;
    until: number;
    emoji?: string;
  } | null>;

  /* Setters */
  setPingHudTick: React.Dispatch<React.SetStateAction<number>>;
  setRadialMenu: React.Dispatch<React.SetStateAction<{ x: number; y: number; visible: boolean }>>;
}

/**
 * Global keyboard shortcuts:
 * - Cmd+Z / Cmd+Shift+Z: undo / redo
 * - Cmd+S: save
 * - Z key: collab ping (tap) or radial emoji menu (hold)
 * - Escape during flood fill: cancel
 * - Selection arrow key translate / rotate
 * - X key: delete selected voxels
 */
export function useKeyboardShortcuts({
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
}: UseKeyboardShortcutsParams): {
  firePing: (p: { nx: number; ny: number }, emoji?: string) => void;
  onRadialSelect: (emoji: string | null) => void;
} {
  const firePing = useCallback(
    (p: { nx: number; ny: number }, emoji?: string) => {
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
    },
    [pingHudRef, setPingHudTick],
  );

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
    [firePing, setRadialMenu, pendingPingRef],
  );

  // ── Escape cancels in-progress flood fill ──
  useEffect(() => {
    if (!workBusy && !fillOperationPending) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Escape") return;
      if (e.repeat) return;
      if (!fillOperationPendingRef.current && !/fill/i.test(workPhaseRef.current ?? "")) return;
      e.preventDefault();
      void invoke("voxel_fill_cancel").catch(() => {});
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [workBusy, fillOperationPending, fillOperationPendingRef, workPhaseRef]);

  // ── Undo/Redo, Save, and Z-key ping ──
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
      pendingPingRef.current = { nx: p.nx, ny: p.ny };
      const scr = lastCursorScreenRef.current ?? { x: 0, y: 0 };
      if (radialHoldTimerRef.current) clearTimeout(radialHoldTimerRef.current);
      radialHoldTimerRef.current = setTimeout(() => {
        radialHoldTimerRef.current = null;
        setRadialMenu({ x: scr.x, y: scr.y, visible: true });
      }, RADIAL_HOLD_MS);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key !== "z" && e.key !== "Z") return;
      if (radialHoldTimerRef.current) {
        clearTimeout(radialHoldTimerRef.current);
        radialHoldTimerRef.current = null;
        const p = pendingPingRef.current;
        pendingPingRef.current = null;
        if (p) firePing(p);
      }
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
  }, [
    preferencesOpen,
    stampBookOpen,
    joinModalOpen,
    newProjectOpen,
    collabJoinPending,
    firePing,
    lastViewportPickNormRef,
    pendingPingRef,
    radialHoldTimerRef,
    lastCursorScreenRef,
    setRadialMenu,
  ]);

  // ── Selection shortcuts: X to delete, arrow keys to translate/rotate ──
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

  return { firePing, onRadialSelect };
}
