import { forwardRef } from "react";

const CURSOR_SIZE = 24;

interface Props {
  visible: boolean;
}

/**
 * A gamepad-driven virtual cursor overlay.
 *
 * The outer div is positioned via `style.transform` directly from the gamepad
 * poll loop (no React re-renders per frame). The `ref` points to the element
 * whose transform is updated.
 */
const VirtualCursor = forwardRef<HTMLDivElement, Props>(({ visible }, ref) => {
  if (!visible) return null;

  return (
    <div
      ref={ref}
      style={{
        position: "fixed",
        left: 0,
        top: 0,
        width: CURSOR_SIZE,
        height: CURSOR_SIZE,
        zIndex: 100000,
        pointerEvents: "none",
        willChange: "transform",
      }}
    >
      {/* Crosshair */}
      <svg
        width={CURSOR_SIZE}
        height={CURSOR_SIZE}
        viewBox="0 0 24 24"
        style={{ display: "block" }}
      >
        {/* Outer ring */}
        <circle cx="12" cy="12" r="9" fill="none" stroke="rgba(0,0,0,0.5)" strokeWidth="2.5" />
        <circle
          cx="12"
          cy="12"
          r="9"
          fill="none"
          stroke="rgba(255,255,255,0.9)"
          strokeWidth="1.5"
        />
        {/* Center dot */}
        <circle cx="12" cy="12" r="2" fill="white" stroke="rgba(0,0,0,0.5)" strokeWidth="1" />
      </svg>
    </div>
  );
});

VirtualCursor.displayName = "VirtualCursor";
export default VirtualCursor;
