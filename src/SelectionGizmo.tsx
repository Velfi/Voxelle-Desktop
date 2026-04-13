/**
 * Selection transform gizmo — interaction only.
 *
 * The gizmo is rendered on the GPU by the wgpu pipeline (sync_gizmo_gpu in lib.rs).
 * Drag state, hit-testing, accumulator math, and translate/rotate dispatch all live
 * in Rust. This component is a thin IPC bridge: it forwards raw pointer events and
 * reports whether the cursor is over a handle.
 */
import { invoke } from "@tauri-apps/api/core";
import { forwardRef, useCallback, useImperativeHandle, useRef, type MutableRefObject } from "react";

export type SelectionGizmoRef = {
  /** Call on pointer-down. Returns true and starts a drag if a handle was hit. */
  startDragIfHit(clientX: number, clientY: number): Promise<boolean>;
  /** Call on pointer-move while gestureRef.mode === "selectionGizmo". */
  pointerMove(clientX: number, clientY: number, prevClientX: number, prevClientY: number): void;
  /** Call on pointer-up. */
  pointerUp(): void;
  /** Call on pointer-move when no drag is active. Returns true if cursor is over a handle. */
  updateHover(clientX: number, clientY: number): Promise<boolean>;
};

export const SelectionGizmo = forwardRef<
  SelectionGizmoRef,
  {
    selectionCount: number;
    flyMode: boolean;
    loadingOrBusy: boolean;
    stampOrPunch: boolean;
    viewportEl: HTMLElement | null;
    viewportPhysRef: MutableRefObject<{ w: number; h: number }>;
  }
>(({ viewportEl, viewportPhysRef }, ref) => {
  const dprRef = useRef(window.devicePixelRatio || 1);

  /** Convert clientX/Y to physical pixel coords matching Rust projections. */
  const toPhysical = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      if (!viewportEl) return [0, 0];
      const rect = viewportEl.getBoundingClientRect();
      const rw = rect.width;
      const rh = rect.height;
      if (rw <= 0 || rh <= 0) return [0, 0];
      const phys = viewportPhysRef.current;
      const scaleX = phys.w > 0 ? phys.w / rw : window.devicePixelRatio || 1;
      const scaleY = phys.h > 0 ? phys.h / rh : window.devicePixelRatio || 1;
      dprRef.current = (scaleX + scaleY) * 0.5;
      return [
        ((clientX - rect.left) / rw) * (phys.w || rw * scaleX),
        ((clientY - rect.top) / rh) * (phys.h || rh * scaleY),
      ];
    },
    [viewportEl, viewportPhysRef],
  );

  useImperativeHandle(
    ref,
    () => ({
      async startDragIfHit(clientX, clientY): Promise<boolean> {
        const [sx, sy] = toPhysical(clientX, clientY);
        return invoke<boolean>("gizmo_pointer_down", { sx, sy, dpr: dprRef.current });
      },

      pointerMove(clientX, clientY, prevClientX, prevClientY) {
        const rect = viewportEl?.getBoundingClientRect();
        const rw = rect?.width ?? 0;
        const rh = rect?.height ?? 0;
        const phys = viewportPhysRef.current;
        const scaleX = rw > 0 ? (phys.w > 0 ? phys.w / rw : window.devicePixelRatio || 1) : 1;
        const scaleY = rh > 0 ? (phys.h > 0 ? phys.h / rh : window.devicePixelRatio || 1) : 1;
        const dcx = (clientX - prevClientX) * scaleX;
        const dcy = (clientY - prevClientY) * scaleY;
        void invoke("gizmo_pointer_move", { dcx, dcy }).catch(() => {});
      },

      pointerUp() {
        void invoke("gizmo_pointer_up").catch(() => {});
      },

      async updateHover(clientX, clientY): Promise<boolean> {
        const [sx, sy] = toPhysical(clientX, clientY);
        return invoke<boolean>("gizmo_hit_test", { sx, sy, dpr: dprRef.current });
      },
    }),
    [toPhysical, viewportEl, viewportPhysRef],
  );

  // No DOM output — rendering is done by the wgpu pipeline.
  return null;
});

SelectionGizmo.displayName = "SelectionGizmo";
