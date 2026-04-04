import { useEffect, useRef, useState, useCallback } from "react";

/** Emoji choices arranged around the radial menu. */
const EMOJI_OPTIONS = ["👍", "❤️", "😂", "👀", "🔥", "❓", "⭐", "💀"];

/** Radius in pixels from cursor to center of each emoji slot. */
const RADIUS = 64;

/** How long the user must hold Z before the radial menu appears (ms). */
export const RADIAL_HOLD_MS = 200;

interface Props {
  /** Screen X where the menu should appear (cursor position on Z-down). */
  x: number;
  /** Screen Y where the menu should appear. */
  y: number;
  /** Whether the menu is currently visible. */
  visible: boolean;
  /** Called when user releases Z while hovering an emoji (or null for no selection). */
  onSelect: (emoji: string | null) => void;
}

export default function RadialPingMenu({ x, y, visible, onSelect }: Props) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Track mouse movement to determine which slice is hovered
  const handlePointerMove = useCallback(
    (e: PointerEvent) => {
      if (!visible) return;
      const dx = e.clientX - x;
      const dy = e.clientY - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // Dead zone in the center — no selection if cursor stays near origin
      if (dist < 20) {
        setHoveredIndex(null);
        return;
      }
      // Angle from center, starting at top (-Y = 0), clockwise
      let angle = Math.atan2(dx, -dy); // 0 = up, positive = clockwise
      if (angle < 0) angle += 2 * Math.PI;
      const sliceSize = (2 * Math.PI) / EMOJI_OPTIONS.length;
      const idx = Math.floor(angle / sliceSize);
      setHoveredIndex(idx);
    },
    [visible, x, y],
  );

  useEffect(() => {
    if (!visible) return;
    window.addEventListener("pointermove", handlePointerMove);
    return () => window.removeEventListener("pointermove", handlePointerMove);
  }, [visible, handlePointerMove]);

  // Listen for Z key up to commit the selection
  useEffect(() => {
    if (!visible) return;
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key !== "z" && e.key !== "Z") return;
      onSelect(hoveredIndex != null ? EMOJI_OPTIONS[hoveredIndex] : null);
    };
    window.addEventListener("keyup", onKeyUp);
    return () => window.removeEventListener("keyup", onKeyUp);
  }, [visible, hoveredIndex, onSelect]);

  // Reset hover when menu appears/disappears
  useEffect(() => {
    if (!visible) setHoveredIndex(null);
  }, [visible]);

  if (!visible) return null;

  return (
    <div
      ref={containerRef}
      style={{
        position: "fixed",
        left: 0,
        top: 0,
        width: "100vw",
        height: "100vh",
        zIndex: 99999,
        pointerEvents: "none",
      }}
    >
      {/* Subtle backdrop circle */}
      <div
        style={{
          position: "absolute",
          left: x - RADIUS - 24,
          top: y - RADIUS - 24,
          width: (RADIUS + 24) * 2,
          height: (RADIUS + 24) * 2,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(0,0,0,0.45) 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />
      {/* Center dot */}
      <div
        style={{
          position: "absolute",
          left: x - 4,
          top: y - 4,
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.6)",
          pointerEvents: "none",
        }}
      />
      {/* Emoji slots */}
      {EMOJI_OPTIONS.map((emoji, i) => {
        const angle = (i / EMOJI_OPTIONS.length) * 2 * Math.PI - Math.PI / 2;
        const ex = x + Math.cos(angle) * RADIUS;
        const ey = y + Math.sin(angle) * RADIUS;
        const isHovered = hoveredIndex === i;
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: ex - 20,
              top: ey - 20,
              width: 40,
              height: 40,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: isHovered ? 30 : 22,
              borderRadius: "50%",
              background: isHovered ? "rgba(255,255,255,0.2)" : "transparent",
              transition: "font-size 0.1s ease, background 0.1s ease",
              pointerEvents: "none",
              userSelect: "none",
              filter: isHovered ? "drop-shadow(0 0 6px rgba(255,255,255,0.5))" : "none",
            }}
          >
            {emoji}
          </div>
        );
      })}
    </div>
  );
}
