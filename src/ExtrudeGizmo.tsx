/**
 * Selection extrude gizmo — interaction only.
 *
 * Rendered on the GPU by sync_gizmo_gpu (SelectExtrude mode: arrows only, no rings).
 * This component is a thin IPC bridge forwarding raw pointer events.
 */
import { invoke } from "@tauri-apps/api/core";
import { forwardRef, useCallback, useImperativeHandle, useRef, type MutableRefObject } from "react";

export type ExtrudeGizmoRef = {
  /** Call on pointer-down. Returns true and starts a drag if a handle was hit. */
  startDragIfHit(clientX: number, clientY: number): Promise<boolean>;
  /** Call on pointer-move while gestureRef.mode === "extrudeGizmo". */
  pointerMove(
    clientX: number,
    clientY: number,
    prevClientX: number,
    prevClientY: number,
    color: number,
    material: string,
  ): void;
  /** Call on pointer-up. */
  pointerUp(): void;
  /** Call on pointer-move when no drag is active. Returns true if cursor is over a handle. */
  updateHover(clientX: number, clientY: number): Promise<boolean>;
};

export const ExtrudeGizmo = forwardRef<
  ExtrudeGizmoRef,
  {
    viewportEl: HTMLElement | null;
    viewportPhysRef: MutableRefObject<{ w: number; h: number }>;
  }
>(({ viewportEl, viewportPhysRef }, ref) => {
  const dprRef = useRef(window.devicePixelRatio || 1);

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
        return invoke<boolean>("extrude_gizmo_pointer_down", {
          sx,
          sy,
          dpr: dprRef.current,
        });
      },

      pointerMove(clientX, clientY, prevClientX, prevClientY, color, material) {
        const rect = viewportEl?.getBoundingClientRect();
        const rw = rect?.width ?? 0;
        const rh = rect?.height ?? 0;
        const phys = viewportPhysRef.current;
        const scaleX = rw > 0 ? (phys.w > 0 ? phys.w / rw : window.devicePixelRatio || 1) : 1;
        const scaleY = rh > 0 ? (phys.h > 0 ? phys.h / rh : window.devicePixelRatio || 1) : 1;
        const dcx = (clientX - prevClientX) * scaleX;
        const dcy = (clientY - prevClientY) * scaleY;
        void invoke("extrude_gizmo_pointer_move", {
          dcx,
          dcy,
          color,
          material,
        }).catch(() => {});
      },

      pointerUp() {
        void invoke("extrude_gizmo_pointer_up").catch(() => {});
      },

      async updateHover(clientX, clientY): Promise<boolean> {
        const [sx, sy] = toPhysical(clientX, clientY);
        return invoke<boolean>("extrude_gizmo_hit_test", {
          sx,
          sy,
          dpr: dprRef.current,
        });
      },
    }),
    [toPhysical, viewportEl, viewportPhysRef],
  );

  return null;
});

ExtrudeGizmo.displayName = "ExtrudeGizmo";
