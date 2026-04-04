import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow, LogicalPosition } from "@tauri-apps/api/window";
import type { InteractionMode } from "../types";
import type { GamepadFrameOutput } from "../useGamepad";

export interface UseWalkModeParams {
  interactionMode: InteractionMode;
  viewportRef: React.RefObject<HTMLDivElement | null>;
  pollGamepad: () => GamepadFrameOutput;
  releaseFlyMouseLook: () => Promise<void>;
  flyMouseLookActiveRef: React.MutableRefObject<boolean>;
  keysDownRef: React.MutableRefObject<Set<string>>;
  flyRafRef: React.MutableRefObject<number>;
  flyLastClientRef: React.MutableRefObject<{ x: number; y: number } | null>;
  flySkipNextFlyMoveRef: React.MutableRefObject<boolean>;
  /** Shared look-delta refs (also used by useGamepad and useFlyMode). */
  flyPendingLookDxRef: React.MutableRefObject<number>;
  flyPendingLookDyRef: React.MutableRefObject<number>;
}

/**
 * Walk-mode camera: first-person with gravity, collision, jumping.
 * Re-uses the fly-mode shared refs for mouse look.
 */
export function useWalkMode({
  interactionMode,
  viewportRef,
  pollGamepad,
  releaseFlyMouseLook,
  flyMouseLookActiveRef,
  keysDownRef,
  flyRafRef,
  flyLastClientRef,
  flySkipNextFlyMoveRef,
  flyPendingLookDxRef,
  flyPendingLookDyRef,
}: UseWalkModeParams): void {
  useEffect(() => {
    if (interactionMode !== "walk") {
      void invoke("set_walk_mode", { enabled: false }).catch(() => {});
      keysDownRef.current.clear();
      void releaseFlyMouseLook();
      return;
    }
    console.log("[walk-debug] walk useEffect SETUP — activating walk mode");
    void invoke("set_walk_mode", { enabled: true })
      .then(() => {
        console.log("[walk-debug] set_walk_mode(true) resolved OK");
      })
      .catch((err) => {
        console.error("[walk-debug] set_walk_mode(true) FAILED:", err);
      });
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
        e.code === "Space" ||
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
        e.code === "Space" ||
        e.code === "ShiftLeft" ||
        e.code === "ShiftRight"
      ) {
        e.preventDefault();
      }
    };
    const dpr = () => window.devicePixelRatio || 1;
    const onWalkPointerMove = (e: PointerEvent) => {
      const vp = viewportRef.current;
      const s = dpr();
      if (!flyMouseLookActiveRef.current || !vp) return;
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
    document.addEventListener("pointermove", onWalkPointerMove, true);
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
      if (k.has("KeyW")) forward += 1;
      if (k.has("KeyS")) forward -= 1;
      if (k.has("KeyD")) right += 1;
      if (k.has("KeyA")) right -= 1;
      const jump = k.has("Space");
      const slow = k.has("ShiftLeft") || k.has("ShiftRight");
      const speedScale = (slow ? 1 / 3 : 1) * gp.speedScale;
      void invoke("sync_fly_input", {
        args: { forward, right, up: 0, speedScale, jump },
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
      document.removeEventListener("pointermove", onWalkPointerMove, true);
      void invoke("set_walk_mode", { enabled: false }).catch(() => {});
      void releaseFlyMouseLook();
    };
  }, [interactionMode, releaseFlyMouseLook, viewportRef, pollGamepad,
      flyMouseLookActiveRef, keysDownRef, flyRafRef, flyLastClientRef,
      flySkipNextFlyMoveRef, flyPendingLookDxRef, flyPendingLookDyRef]);
}
