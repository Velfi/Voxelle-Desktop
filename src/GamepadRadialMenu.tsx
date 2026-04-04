import type { RadialSlice } from "./gamepadRadialMenuData";

const RADIUS = 90;
const SLOT_SIZE = 48;

interface Props {
  visible: boolean;
  slices: RadialSlice[];
  selectedIndex: number | null;
  /** Title shown in the centre of the radial (e.g. "Tool" or "Options"). */
  title?: string;
}

export default function GamepadRadialMenu({
  visible,
  slices,
  selectedIndex,
  title,
}: Props) {
  if (!visible || slices.length === 0) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 99998,
        pointerEvents: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {/* Backdrop circle */}
      <div
        style={{
          position: "absolute",
          width: (RADIUS + 36) * 2,
          height: (RADIUS + 36) * 2,
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.25) 60%, transparent 80%)",
        }}
      />
      {/* Centre label */}
      {title && (
        <div
          style={{
            position: "absolute",
            color: "rgba(255,255,255,0.55)",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 1,
            textTransform: "uppercase",
            userSelect: "none",
          }}
        >
          {title}
        </div>
      )}
      {/* Slices */}
      {slices.map((slice, i) => {
        const angle = (i / slices.length) * 2 * Math.PI - Math.PI / 2;
        const sx = Math.cos(angle) * RADIUS;
        const sy = Math.sin(angle) * RADIUS;
        const isSelected = selectedIndex === i;
        return (
          <div
            key={slice.id}
            style={{
              position: "absolute",
              left: `calc(50% + ${sx}px - ${SLOT_SIZE / 2}px)`,
              top: `calc(50% + ${sy}px - ${SLOT_SIZE / 2}px)`,
              width: SLOT_SIZE,
              height: SLOT_SIZE,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "50%",
              background: isSelected
                ? "rgba(255,255,255,0.22)"
                : "rgba(255,255,255,0.06)",
              border: isSelected
                ? "2px solid rgba(255,255,255,0.5)"
                : "2px solid transparent",
              transition:
                "background 0.1s ease, border 0.1s ease, transform 0.1s ease",
              transform: isSelected ? "scale(1.15)" : "scale(1)",
              pointerEvents: "none",
              userSelect: "none",
            }}
          >
            <span style={{ fontSize: 18, lineHeight: 1 }}>{slice.icon}</span>
            <span
              style={{
                fontSize: 9,
                color: isSelected
                  ? "rgba(255,255,255,0.9)"
                  : "rgba(255,255,255,0.6)",
                marginTop: 2,
                fontWeight: isSelected ? 700 : 400,
                whiteSpace: "nowrap",
              }}
            >
              {slice.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
