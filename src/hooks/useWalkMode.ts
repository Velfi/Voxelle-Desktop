import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
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
    void invoke("set_walk_mode", { enabled: true }).catch(() => {});
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
      // On macOS, setCursorGrab(true) dissociates cursor from mouse via
      // CGAssociateMouseAndMouseCursorPosition — cursor is frozen, no pointermove
      // events reach WKWebView.  The frame loop reads CGGetLastMouseDelta directly.
      // Nothing to do here for the fallback path.
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
      // Bundle look delta with movement args — halves IPC calls per frame.
      // On macOS the frame loop handles camera rotation via CGGetLastMouseDelta;
      // look deltas here are only non-zero on other platforms (pointer lock path).
      void invoke("sync_fly_input", {
        args: { forward, right, up: 0, speedScale, jump, lookDx: pdx, lookDy: pdy },
      }).catch(() => {});
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
  }, [
    interactionMode,
    releaseFlyMouseLook,
    viewportRef,
    pollGamepad,
    flyMouseLookActiveRef,
    keysDownRef,
    flyRafRef,
    flyLastClientRef,
    flySkipNextFlyMoveRef,
    flyPendingLookDxRef,
    flyPendingLookDyRef,
  ]);
}
