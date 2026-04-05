/**
 * Selection extrude gizmo — interaction only.
 *
 * Rendered on the GPU by sync_gizmo_gpu (SelectExtrude mode: arrows only, no rings).
 * This component is a thin IPC bridge forwarding raw pointer events.
 */
import { invoke } from "@tauri-apps/api/core";
import { forwardRef, useCallback, useImperativeHandle, useRef } from "react";

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

export const ExtrudeGizmo = forwardRef<ExtrudeGizmoRef, { viewportEl: HTMLElement | null }>(
  ({ viewportEl }, ref) => {
    const dprRef = useRef(window.devicePixelRatio || 1);

    const toPhysical = useCallback(
      (clientX: number, clientY: number): [number, number] => {
        if (!viewportEl) return [0, 0];
        const rect = viewportEl.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        dprRef.current = dpr;
        return [(clientX - rect.left) * dpr, (clientY - rect.top) * dpr];
      },
      [viewportEl],
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
          const dpr = window.devicePixelRatio || 1;
          const dcx = (clientX - prevClientX) * dpr;
          const dcy = (clientY - prevClientY) * dpr;
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
      [toPhysical],
    );

    return null;
  },
);

ExtrudeGizmo.displayName = "ExtrudeGizmo";
