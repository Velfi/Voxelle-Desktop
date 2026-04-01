/**
 * Selection transform gizmo — interaction only.
 *
 * The gizmo is rendered on the GPU by the wgpu pipeline (sync_gizmo_gpu in lib.rs).
 * This component handles pointer hit-testing and drag logic using 2D projected
 * positions returned by the `get_selection_gizmo_projected` Tauri command.
 */
import { invoke } from "@tauri-apps/api/core";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

type GizmoProj = { sx: number; sy: number; inFront: boolean };

type GizmoData = {
  centerSx: number;
  centerSy: number;
  /** [+X, -X, +Y, -Y, +Z, -Z] */
  moveHandles: GizmoProj[];
  /** 3 rings × 16 samples. Ring 0=X-axis, 1=Y-axis, 2=Z-axis */
  rotateRings: GizmoProj[];
  pxPerWorld: number;
};

type MoveDrag = {
  kind: "move";
  /** index into moveHandles (0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z) */
  handleIdx: number;
  accum: number;
};
type RotateDrag = {
  kind: "rotate";
  ring: 0 | 1 | 2;
  tangentX: number;
  tangentY: number;
  accum: number;
};
type ActiveDrag = MoveDrag | RotateDrag;

export type SelectionGizmoRef = {
  /** Call on pointer-down. Returns true and starts a drag if a handle was hit. */
  startDragIfHit(clientX: number, clientY: number): boolean;
  /** Call on pointer-move while gestureRef.mode === "selectionGizmo". */
  pointerMove(
    clientX: number,
    clientY: number,
    prevClientX: number,
    prevClientY: number,
  ): void;
  /** Call on pointer-up. */
  pointerUp(): void;
  /** Call on pointer-move when no drag is active. Returns true if cursor is over a handle. */
  updateHover(clientX: number, clientY: number): boolean;
};

// ── Constants ─────────────────────────────────────────────────────────────────

const RING_SAMPLES = 16;
/** CSS pixels — scaled by dpr for hit testing */
const MOVE_HIT_CSS = 16;
const RING_HIT_CSS = 11;
/** CSS pixels of drag per 1-voxel translation step */
const PX_PER_MOVE_STEP_CSS = 26;
/** CSS pixels of drag per 90° rotation step */
const PX_PER_ROTATE_STEP_CSS = 65;

// ── Helpers ───────────────────────────────────────────────────────────────────

function distToSegmentSq(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return (px - ax) ** 2 + (py - ay) ** 2;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
  return (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2;
}

const MOVE_AXIS = [0, 0, 1, 1, 2, 2] as const;

// ── Component ─────────────────────────────────────────────────────────────────

export const SelectionGizmo = forwardRef<
  SelectionGizmoRef,
  {
    selectionCount: number;
    flyMode: boolean;
    loadingOrBusy: boolean;
    viewportEl: HTMLElement | null;
  }
>(({ selectionCount, flyMode, loadingOrBusy, viewportEl }, ref) => {
  const dataRef = useRef<GizmoData | null>(null);
  const dragRef = useRef<ActiveDrag | null>(null);

  // ── Screen-space coordinate helpers ───────────────────────────────────────

  /** Convert clientX/Y to physical pixel coords matching Rust projections. */
  const toCanvasPx = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      if (!viewportEl) return [0, 0];
      const rect = viewportEl.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      return [(clientX - rect.left) * dpr, (clientY - rect.top) * dpr];
    },
    [viewportEl],
  );

  // ── RAF polling loop (projection data for hit testing) ────────────────────

  const active = selectionCount > 0 && !flyMode && !loadingOrBusy;

  useEffect(() => {
    if (!active) {
      dataRef.current = null;
      return;
    }
    let alive = true;
    let raf = 0;
    const tick = () => {
      if (!alive) return;
      void invoke<GizmoData | null>("get_selection_gizmo_projected")
        .then((d) => {
          if (!alive) return;
          dataRef.current = d ?? null;
          raf = requestAnimationFrame(tick);
        })
        .catch(() => {
          if (alive) raf = requestAnimationFrame(tick);
        });
    };
    raf = requestAnimationFrame(tick);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      dataRef.current = null;
    };
  }, [active]);

  // ── Imperative API ─────────────────────────────────────────────────────────

  useImperativeHandle(ref, () => ({
    startDragIfHit(clientX, clientY): boolean {
      const data = dataRef.current;
      if (!data) return false;
      const [cx, cy] = toCanvasPx(clientX, clientY);
      const dpr = window.devicePixelRatio || 1;
      const moveHitSq = (MOVE_HIT_CSS * dpr) ** 2;
      const ringHit = RING_HIT_CSS * dpr;

      // Move handles take priority
      for (let i = 0; i < 6; i++) {
        const h = data.moveHandles[i];
        if ((cx - h.sx) ** 2 + (cy - h.sy) ** 2 <= moveHitSq) {
          dragRef.current = { kind: "move", handleIdx: i, accum: 0 };
          return true;
        }
      }
      // Rotate rings
      for (let ring = 0; ring < 3; ring++) {
        const start = ring * RING_SAMPLES;
        let bestSq = Infinity;
        let bestTx = 1, bestTy = 0;
        for (let i = 0; i < RING_SAMPLES; i++) {
          const p = data.rotateRings[start + i];
          const next = data.rotateRings[start + (i + 1) % RING_SAMPLES];
          const sq = distToSegmentSq(cx, cy, p.sx, p.sy, next.sx, next.sy);
          if (sq < bestSq) {
            bestSq = sq;
            const tdx = next.sx - p.sx;
            const tdy = next.sy - p.sy;
            const tlen = Math.hypot(tdx, tdy);
            if (tlen > 0.5) { bestTx = tdx / tlen; bestTy = tdy / tlen; }
          }
        }
        if (bestSq <= ringHit * ringHit) {
          dragRef.current = {
            kind: "rotate",
            ring: ring as 0 | 1 | 2,
            tangentX: bestTx,
            tangentY: bestTy,
            accum: 0,
          };
          return true;
        }
      }
      return false;
    },

    pointerMove(clientX, clientY, prevClientX, prevClientY) {
      const drag = dragRef.current;
      const data = dataRef.current;
      if (!drag || !data) return;
      const dpr = window.devicePixelRatio || 1;
      const dcx = (clientX - prevClientX) * dpr;
      const dcy = (clientY - prevClientY) * dpr;

      if (drag.kind === "move") {
        const h = data.moveHandles[drag.handleIdx];
        const adx = h.sx - data.centerSx;
        const ady = h.sy - data.centerSy;
        const alen = Math.hypot(adx, ady);
        if (alen < 1) return;
        drag.accum += (dcx * adx + dcy * ady) / alen;

        const threshold = PX_PER_MOVE_STEP_CSS * dpr;
        const steps = Math.trunc(drag.accum / threshold);
        if (steps !== 0) {
          drag.accum -= steps * threshold;
          const axis = MOVE_AXIS[drag.handleIdx];
          const magnitude = drag.handleIdx % 2 === 0 ? steps : -steps;
          const dx = axis === 0 ? magnitude : 0;
          const dy = axis === 1 ? magnitude : 0;
          const dz = axis === 2 ? magnitude : 0;
          void invoke("selection_translate", { dx, dy, dz }).catch(() => {});
        }
      } else {
        drag.accum += dcx * drag.tangentX + dcy * drag.tangentY;
        const threshold = PX_PER_ROTATE_STEP_CSS * dpr;
        const steps = Math.trunc(drag.accum / threshold);
        if (steps !== 0) {
          drag.accum -= steps * threshold;
          void invoke("selection_rotate", { axis: drag.ring, quarters: steps }).catch(() => {});
        }
      }
    },

    pointerUp() {
      dragRef.current = null;
    },

    updateHover(clientX, clientY): boolean {
      const data = dataRef.current;
      if (!data) return false;
      const [cx, cy] = toCanvasPx(clientX, clientY);
      const dpr = window.devicePixelRatio || 1;
      const moveHitSq = (MOVE_HIT_CSS * dpr) ** 2;
      const ringHit = RING_HIT_CSS * dpr;
      for (let i = 0; i < 6; i++) {
        const h = data.moveHandles[i];
        if ((cx - h.sx) ** 2 + (cy - h.sy) ** 2 <= moveHitSq) return true;
      }
      for (let ring = 0; ring < 3; ring++) {
        const start = ring * RING_SAMPLES;
        for (let i = 0; i < RING_SAMPLES; i++) {
          const p = data.rotateRings[start + i];
          const next = data.rotateRings[start + (i + 1) % RING_SAMPLES];
          if (distToSegmentSq(cx, cy, p.sx, p.sy, next.sx, next.sy) <= ringHit * ringHit) return true;
        }
      }
      return false;
    },
  }), [toCanvasPx]);

  // No DOM output — rendering is done by the wgpu pipeline.
  return null;
});

SelectionGizmo.displayName = "SelectionGizmo";
