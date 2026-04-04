import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

/** Inset from screen edge in CSS pixels. */
const EDGE_MARGIN = 32;

interface Props {
  /** World-space X of the active ping. */
  wx: number;
  /** World-space Y of the active ping. */
  wy: number;
  /** World-space Z of the active ping. */
  wz: number;
  /** Whether a ping is currently active. */
  active: boolean;
  /** Optional emoji to show next to the arrow. */
  emoji?: string;
}

interface ProjectResult {
  onScreen: boolean;
  sx: number;
  sy: number;
  vw: number;
  vh: number;
}

export function PingArrowIndicator({ wx, wy, wz, active, emoji }: Props) {
  const [pos, setPos] = useState<{
    x: number;
    y: number;
    angle: number;
    visible: boolean;
  }>({ x: 0, y: 0, angle: 0, visible: false });

  const rafRef = useRef(0);
  const activeRef = useRef(active);
  activeRef.current = active;

  useEffect(() => {
    if (!active) {
      setPos((p) => (p.visible ? { ...p, visible: false } : p));
      return;
    }

    let cancelled = false;
    const poll = () => {
      if (cancelled || !activeRef.current) return;
      invoke<ProjectResult>("project_world_point", {
        args: { x: wx, y: wy, z: wz },
      })
        .then((r) => {
          if (cancelled) return;
          if (r.onScreen || r.vw === 0) {
            setPos((p) => (p.visible ? { ...p, visible: false } : p));
          } else {
            // Compute arrow position clamped to viewport edges
            const cx = r.vw / 2;
            const cy = r.vh / 2;
            const dx = r.sx - cx;
            const dy = r.sy - cy;
            const angle = Math.atan2(dy, dx);

            // Cast a ray from center at `angle` to find intersection with viewport rect
            const halfW = cx - EDGE_MARGIN;
            const halfH = cy - EDGE_MARGIN;
            const absCos = Math.abs(Math.cos(angle));
            const absSin = Math.abs(Math.sin(angle));
            let t: number;
            if (absCos * halfH > absSin * halfW) {
              t = halfW / absCos;
            } else {
              t = halfH / absSin;
            }
            const ax = cx + Math.cos(angle) * t;
            const ay = cy + Math.sin(angle) * t;
            setPos({ x: ax, y: ay, angle, visible: true });
          }
          rafRef.current = requestAnimationFrame(poll);
        })
        .catch(() => {
          if (!cancelled) rafRef.current = requestAnimationFrame(poll);
        });
    };
    rafRef.current = requestAnimationFrame(poll);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [active, wx, wy, wz]);

  if (!pos.visible) return null;

  const arrowDeg = (pos.angle * 180) / Math.PI;

  return (
    <div
      style={{
        position: "fixed",
        left: pos.x,
        top: pos.y,
        transform: `translate(-50%, -50%)`,
        zIndex: 99998,
        pointerEvents: "none",
        display: "flex",
        alignItems: "center",
        gap: 4,
        flexDirection: arrowDeg > 90 || arrowDeg < -90 ? "row-reverse" : "row",
      }}
    >
      {/* Arrow triangle */}
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        style={{
          transform: `rotate(${arrowDeg}deg)`,
          filter: "drop-shadow(0 0 4px rgba(0,0,0,0.6))",
        }}
      >
        <polygon
          points="2,6 22,12 2,18"
          fill="rgba(255, 220, 80, 0.95)"
          stroke="rgba(0,0,0,0.4)"
          strokeWidth="1"
        />
      </svg>
      {emoji && (
        <span
          style={{
            fontSize: 20,
            filter: "drop-shadow(0 0 3px rgba(0,0,0,0.5))",
            userSelect: "none",
          }}
        >
          {emoji}
        </span>
      )}
    </div>
  );
}
