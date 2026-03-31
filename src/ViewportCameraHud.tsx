import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useRef, useState } from "react";

const GIZMO_SIZE = 120;
const GIZMO_RADIUS = 40;
const LABEL_HIT_RADIUS = 14;
const EDGE_BAND = 10;

type ProjItem = { sx: number; sy: number; depth: number };

function sortForDraw(a: ProjItem, b: ProjItem) {
  return a.depth - b.depth;
}

function sortForHit(a: ProjItem, b: ProjItem) {
  return b.depth - a.depth;
}

export function ViewportCameraHud(props: {
  flyMode: boolean;
  loadingOrBusy: boolean;
}) {
  const { flyMode, loadingOrBusy } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const lastProjRef = useRef<ProjItem[] | null>(null);
  const hoveredIndexRef = useRef(-1);
  const hoverEdgeRef = useRef(false);
  const draggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const thetaOnlyRef = useRef(false);

  const draw = useCallback(() => {
    const ctx = ctxRef.current;
    const canvas = canvasRef.current;
    const items = lastProjRef.current;
    if (!ctx || !canvas || !items || items.length !== 6) return;

    const dpr = window.devicePixelRatio || 1;
    const w = GIZMO_SIZE * dpr;
    const cx = w / 2;
    const cy = w / 2;
    const hoveredIndex = hoveredIndexRef.current;
    const hoverEdge = hoverEdgeRef.current;

    ctx.clearRect(0, 0, w, w);

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, GIZMO_RADIUS * dpr + 1, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(40, 40, 40, 0.55)";
    ctx.fill();
    ctx.strokeStyle = hoverEdge
      ? "rgba(255,255,255,0.45)"
      : "rgba(255,255,255,0.12)";
    ctx.lineWidth = hoverEdge ? 2.5 * dpr : 1;
    ctx.stroke();
    ctx.restore();

    const axes = [
      { label: "X", color: "#E05555", dimColor: "#8B3535", neg: false },
      { label: "Y", color: "#55B855", dimColor: "#357035", neg: false },
      { label: "Z", color: "#5580E0", dimColor: "#354F8B", neg: false },
      { label: "", color: "#E05555", dimColor: "#8B3535", neg: true },
      { label: "", color: "#55B855", dimColor: "#357035", neg: true },
      { label: "", color: "#5580E0", dimColor: "#354F8B", neg: true },
    ];

    const sorted = items.map((p, idx) => ({ ...p, idx })).sort(sortForDraw);

    for (const item of sorted) {
      const ax = axes[item.idx];
      const front = !ax.neg;
      const sx = item.sx * dpr;
      const sy = item.sy * dpr;
      const hovered = hoveredIndex === item.idx;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + sx, cy + sy);
      ctx.strokeStyle = front ? ax.color : ax.dimColor;
      ctx.lineWidth = (hovered ? 2.5 : 1.5) * dpr;
      ctx.globalAlpha = front ? 0.9 : 0.4;
      ctx.stroke();
      ctx.restore();

      const dotRadius = ax.neg ? (hovered ? 6 : 4.5) : hovered ? 8 : 6.5;
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx + sx, cy + sy, dotRadius * dpr, 0, Math.PI * 2);
      ctx.fillStyle = front ? ax.color : ax.dimColor;
      ctx.globalAlpha = hovered ? 1 : front ? 0.95 : 0.5;
      ctx.fill();
      ctx.restore();

      if (!ax.neg) {
        ctx.save();
        const fontSize = (hovered ? 12 : 10) * dpr;
        ctx.font = `bold ${fontSize}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#fff";
        ctx.globalAlpha = front ? 1 : 0.45;
        ctx.fillText(ax.label, cx + sx, cy + sy + 0.5 * dpr);
        ctx.restore();
      }
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = GIZMO_SIZE * dpr;
    canvas.height = GIZMO_SIZE * dpr;
    ctxRef.current = canvas.getContext("2d");
    draw();
  }, [draw]);

  useEffect(() => {
    if (flyMode || loadingOrBusy) return;

    let alive = true;
    let raf = 0;
    const tick = () => {
      if (!alive) return;
      void invoke<ProjItem[]>("get_orbit_gizmo_projection")
        .then((items) => {
          if (!alive) return;
          lastProjRef.current = items;
          draw();
        })
        .catch(() => {});
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
    };
  }, [flyMode, loadingOrBusy, draw]);

  const [zoomPercent, setZoomPercent] = useState(100);
  useEffect(() => {
    if (flyMode || loadingOrBusy) return;
    const id = window.setInterval(() => {
      void invoke<number>("get_camera_zoom_percent")
        .then((z) => setZoomPercent(z))
        .catch(() => {});
    }, 200);
    void invoke<number>("get_camera_zoom_percent")
      .then((z) => setZoomPercent(z))
      .catch(() => {});
    return () => clearInterval(id);
  }, [flyMode, loadingOrBusy]);

  const projectAxes = useCallback((): {
    sx: number;
    sy: number;
    depth: number;
    idx: number;
  }[] => {
    const items = lastProjRef.current;
    if (!items) return [];
    return items
      .map((p, idx) => ({ sx: p.sx, sy: p.sy, depth: p.depth, idx }))
      .sort(sortForHit);
  }, []);

  const edgeTest = useCallback(
    (ex: number, ey: number, canvas: HTMLCanvasElement) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ex - rect.left - GIZMO_SIZE / 2;
      const my = ey - rect.top - GIZMO_SIZE / 2;
      const dist = Math.sqrt(mx * mx + my * my);
      return (
        dist >= GIZMO_RADIUS - EDGE_BAND && dist <= GIZMO_RADIUS + EDGE_BAND
      );
    },
    [],
  );

  const hitTest = useCallback(
    (ex: number, ey: number, canvas: HTMLCanvasElement) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ex - rect.left - GIZMO_SIZE / 2;
      const my = ey - rect.top - GIZMO_SIZE / 2;
      const items = projectAxes();
      for (const item of items) {
        const dx = mx - item.sx;
        const dy = my - item.sy;
        if (dx * dx + dy * dy <= LABEL_HIT_RADIUS * LABEL_HIT_RADIUS) {
          return item.idx;
        }
      }
      return -1;
    },
    [projectAxes],
  );

  const onPointerDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    if (flyMode || loadingOrBusy) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const idx = hitTest(e.clientX, e.clientY, canvas);
    if (idx >= 0) {
      void invoke("camera_snap_orbit_axis", { axis: idx }).catch(() => {});
      return;
    }
    draggingRef.current = true;
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    thetaOnlyRef.current = edgeTest(e.clientX, e.clientY, canvas);
    canvas.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    e.stopPropagation();
    if (flyMode || loadingOrBusy) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (draggingRef.current) {
      const dx = e.clientX - dragStartRef.current.x;
      const dy = e.clientY - dragStartRef.current.y;
      dragStartRef.current = { x: e.clientX, y: e.clientY };
      void invoke("camera_orbit_gizmo_drag", {
        args: { dx, dy, thetaOnly: thetaOnlyRef.current },
      }).catch(() => {});
      return;
    }

    const idx = hitTest(e.clientX, e.clientY, canvas);
    const nowEdge = idx < 0 && edgeTest(e.clientX, e.clientY, canvas);
    if (idx !== hoveredIndexRef.current || nowEdge !== hoverEdgeRef.current) {
      hoveredIndexRef.current = idx;
      hoverEdgeRef.current = nowEdge;
      canvas.style.cursor =
        idx >= 0 ? "pointer" : nowEdge ? "ew-resize" : "grab";
      draw();
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    e.stopPropagation();
    if (draggingRef.current) {
      draggingRef.current = false;
      thetaOnlyRef.current = false;
      try {
        canvasRef.current?.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    }
  };

  const onPointerLeave = () => {
    if (hoveredIndexRef.current !== -1 || hoverEdgeRef.current) {
      hoveredIndexRef.current = -1;
      hoverEdgeRef.current = false;
      if (canvasRef.current) canvasRef.current.style.cursor = "grab";
      draw();
    }
  };

  if (flyMode) {
    return null;
  }

  return (
    <>
      <canvas
        ref={canvasRef}
        className="viewport-orbit-gizmo"
        style={{ width: GIZMO_SIZE, height: GIZMO_SIZE }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
        role="toolbar"
        aria-label="Orbit gizmo"
      />
      <div
        className="viewport-zoom-controls"
        role="toolbar"
        aria-label="Zoom controls"
        tabIndex={0}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          disabled={loadingOrBusy}
          onClick={() =>
            void invoke("camera_zoom_step", { inward: false }).catch(() => {})
          }
          title="Zoom out"
          aria-label="Zoom out"
        >
          −
        </button>
        <span className="viewport-zoom-percent">{zoomPercent}%</span>
        <button
          type="button"
          disabled={loadingOrBusy}
          onClick={() =>
            void invoke("camera_zoom_step", { inward: true }).catch(() => {})
          }
          title="Zoom in"
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          className="viewport-zoom-fit"
          disabled={loadingOrBusy}
          onClick={() => void invoke("camera_fit_to_scene").catch(() => {})}
          title="Fit to view"
          aria-label="Fit sculpture to view"
        >
          Fit
        </button>
        <button
          type="button"
          className="viewport-zoom-fit"
          disabled={loadingOrBusy}
          onClick={() => void invoke("camera_reset_view").catch(() => {})}
          title="Reset camera"
          aria-label="Reset camera to default view"
        >
          Reset
        </button>
      </div>
    </>
  );
}
