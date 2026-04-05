import { type RefObject } from "react";
import RadialPingMenu from "./RadialPingMenu";
import GamepadRadialMenu from "./GamepadRadialMenu";
import VirtualCursor from "./VirtualCursor";
import { PingArrowIndicator } from "./PingArrowIndicator";
import type { GamepadHandle } from "./useGamepad";

interface PingHudEntry {
  name: string;
  wx: number;
  wy: number;
  wz: number;
  until: number;
  emoji?: string;
}

interface Props {
  // Radial emoji-ping menu
  radialMenu: { x: number; y: number; visible: boolean };
  onRadialSelect: (emoji: string | null) => void;

  // Gamepad radial menus + HUD
  gamepad: GamepadHandle;

  // Virtual cursor
  virtualCursorElRef: RefObject<HTMLDivElement | null>;

  // Ping arrow indicator
  pingHudRef: RefObject<PingHudEntry | null>;
  pingHudTick: number;
}

export function GameHUD({
  radialMenu,
  onRadialSelect,
  gamepad,
  virtualCursorElRef,
  pingHudRef,
  pingHudTick,
}: Props) {
  return (
    <>
      {/* Radial emoji-ping menu (hold Z) */}
      <RadialPingMenu
        x={radialMenu.x}
        y={radialMenu.y}
        visible={radialMenu.visible}
        onSelect={onRadialSelect}
      />

      {/* Gamepad radial menus (LT / RT triggers) */}
      <GamepadRadialMenu
        visible={gamepad.radialMenu != null}
        slices={gamepad.radialSlices}
        selectedIndex={gamepad.selectedSliceIndex}
        title={
          gamepad.radialMenu === "tools"
            ? "Tool"
            : gamepad.radialMenu === "subOptions"
              ? "Options"
              : undefined
        }
      />

      {/* Gamepad speed HUD */}
      {gamepad.connected && gamepad.speedMultiplier > 1 && (
        <div
          style={{
            position: "fixed",
            bottom: 48,
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(0,0,0,0.5)",
            color: "rgba(255,255,255,0.8)",
            padding: "4px 12px",
            borderRadius: 6,
            fontSize: 13,
            fontWeight: 600,
            pointerEvents: "none",
            zIndex: 99990,
            userSelect: "none",
          }}
        >
          {gamepad.speedMultiplier}x
        </div>
      )}

      {/* Gamepad virtual cursor (X to toggle) */}
      <VirtualCursor
        ref={virtualCursorElRef as RefObject<HTMLDivElement>}
        visible={gamepad.cursorMode}
      />

      {/* Off-screen ping arrow indicator */}
      {(() => {
        const p = pingHudRef.current;
        // pingHudTick is read to subscribe to re-renders
        void pingHudTick;
        const isActive = !!p && Date.now() < p.until;
        return (
          <PingArrowIndicator
            wx={p?.wx ?? 0}
            wy={p?.wy ?? 0}
            wz={p?.wz ?? 0}
            active={isActive}
            emoji={p?.emoji}
          />
        );
      })()}
    </>
  );
}
