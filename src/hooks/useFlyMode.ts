import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow, LogicalPosition } from "@tauri-apps/api/window";
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
  }, [flyPendingLookDxRef, flyPendingLookDyRef]);

  const activateFlyMouseLook = useCallback(
    async (_pointerId: number) => {
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
      // Poll gamepad (accumulates look deltas into flyPendingLookDx/DyRef)
      const gp = pollGamepad();
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
