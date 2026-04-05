import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import type { InteractionMode } from "../types";
import type { GamepadFrameOutput } from "../useGamepad";

export interface UseFlyModeParams {
  interactionMode: InteractionMode;
  viewportRef: React.RefObject<HTMLDivElement | null>;
  pollGamepad: () => GamepadFrameOutput;
  /** Shared look-delta refs (also used by useGamepad). */
  flyPendingLookDxRef: React.MutableRefObject<number>;
  flyPendingLookDyRef: React.MutableRefObject<number>;
}

export interface UseFlyModeResult {
  flySpeed: 1 | 2 | 4;
  setFlySpeed: (s: 1 | 2 | 4) => void;
  releaseFlyMouseLook: () => Promise<void>;
  activateFlyMouseLook: (pointerId: number) => Promise<void>;
  flyMouseLookActiveRef: React.MutableRefObject<boolean>;
  keysDownRef: React.MutableRefObject<Set<string>>;
  flyRafRef: React.MutableRefObject<number>;
  flyLastClientRef: React.MutableRefObject<{ x: number; y: number } | null>;
  flySkipNextFlyMoveRef: React.MutableRefObject<boolean>;
  flyCapturedPointerIdRef: React.MutableRefObject<number | null>;
  flySpeedRef: React.MutableRefObject<1 | 2 | 4>;
}

/**
 * Manages fly-mode camera: WASD movement, pointer-lock mouse look, RAF tick loop.
 * Exports shared refs used by walk mode as well.
 */
export function useFlyMode({
  interactionMode,
  viewportRef,
  pollGamepad,
  flyPendingLookDxRef,
  flyPendingLookDyRef,
}: UseFlyModeParams): UseFlyModeResult {
  const flyRafRef = useRef<number>(0);
  const flyMouseLookActiveRef = useRef(false);
  const flyCapturedPointerIdRef = useRef<number | null>(null);
  const flyLastClientRef = useRef<{ x: number; y: number } | null>(null);
  const flySkipNextFlyMoveRef = useRef(false);
  const keysDownRef = useRef<Set<string>>(new Set());
  const [flySpeed, setFlySpeed] = useState<1 | 2 | 4>(1);
  const flySpeedRef = useRef<1 | 2 | 4>(1);
  flySpeedRef.current = flySpeed;

  const releaseFlyMouseLook = useCallback(async () => {
    flyMouseLookActiveRef.current = false;
    flyLastClientRef.current = null;
    flySkipNextFlyMoveRef.current = false;
    flyPendingLookDxRef.current = 0;
    flyPendingLookDyRef.current = 0;
    flyCapturedPointerIdRef.current = null;
    void invoke("set_native_look", { active: false }).catch(() => {});
    if (document.pointerLockElement) {
      try {
        document.exitPointerLock();
      } catch {
        /* */
      }
    }
    const w = getCurrentWindow();
    try {
      await w.setCursorGrab(false);
    } catch {
      /* */
    }
    try {
      await w.setCursorVisible(true);
    } catch {
      /* */
    }
  }, [flyPendingLookDxRef, flyPendingLookDyRef]);

  const activateFlyMouseLook = useCallback(
    async (_pointerId: number) => {
      const el = viewportRef.current;
      if (!el) return;
      flySkipNextFlyMoveRef.current = false;
      flyPendingLookDxRef.current = 0;
      flyPendingLookDyRef.current = 0;
      flyCapturedPointerIdRef.current = null;
      // Try pointer lock first (works on Linux/Windows; fails silently on macOS/WKWebView).
      try {
        await el.requestPointerLock();
      } catch {
        /* expected on macOS */
      }
      if (document.pointerLockElement !== el) {
        // macOS/WKWebView fallback: setCursorGrab dissociates cursor from mouse via
        // CGAssociateMouseAndMouseCursorPosition; cursor is hidden and frozen in place.
        // Do NOT call setCursorPosition here — any warp queues a delta in
        // CGGetLastMouseDelta that would jump the camera on the first move.
        const w = getCurrentWindow();
        try {
          await w.setCursorGrab(true);
        } catch {
          /* unsupported on some platforms */
        }
        try {
          await w.setCursorVisible(false);
        } catch {
          /* */
        }
        void invoke("set_native_look", { active: true }).catch(() => {});
      }
      flyMouseLookActiveRef.current = true;
    },
    [viewportRef, flyPendingLookDxRef, flyPendingLookDyRef],
  );

  // ── Fly mode useEffect ──
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

      // On macOS, setCursorGrab(true) dissociates cursor from mouse via
      // CGAssociateMouseAndMouseCursorPosition — cursor is frozen, no pointermove
      // events reach WKWebView.  The frame loop reads CGGetLastMouseDelta directly.
      // Nothing to do here for the fallback path.
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("pointermove", onFlyPointerMove, true);
    const tick = () => {
      // Poll gamepad (accumulates look deltas into flyPendingLookDx/DyRef)
      const gp = pollGamepad();
      const pdx = flyPendingLookDxRef.current;
      const pdy = flyPendingLookDyRef.current;
      flyPendingLookDxRef.current = 0;
      flyPendingLookDyRef.current = 0;
      const k = keysDownRef.current;
      let forward = gp.forward;
      let right = gp.right;
      let up = gp.up;
      if (k.has("KeyW")) forward += 1;
      if (k.has("KeyS")) forward -= 1;
      if (k.has("KeyD")) right += 1;
      if (k.has("KeyA")) right -= 1;
      if (k.has("KeyE")) up += 1;
      if (k.has("KeyQ")) up -= 1;
      const slow = k.has("ShiftLeft") || k.has("ShiftRight");
      const speedScale = (slow ? 1 / 8 : 1) * flySpeedRef.current * gp.speedScale;
      // Bundle look delta with movement args — halves IPC calls per frame.
      // On macOS the frame loop handles camera rotation via CGGetLastMouseDelta;
      // look deltas here are only non-zero on other platforms (pointer lock path).
      void invoke("sync_fly_input", {
        args: { forward, right, up, speedScale, lookDx: pdx, lookDy: pdy },
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
  }, [
    interactionMode,
    releaseFlyMouseLook,
    viewportRef,
    pollGamepad,
    flyPendingLookDxRef,
    flyPendingLookDyRef,
  ]);

  return {
    flySpeed,
    setFlySpeed,
    releaseFlyMouseLook,
    activateFlyMouseLook,
    flyMouseLookActiveRef,
    keysDownRef,
    flyRafRef,
    flyLastClientRef,
    flySkipNextFlyMoveRef,
    flyCapturedPointerIdRef,
    flySpeedRef,
  };
}
