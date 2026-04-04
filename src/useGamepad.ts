import { useCallback, useEffect, useRef, useState } from "react";
import type { MutableRefObject } from "react";
import type { InteractionMode } from "./types";
import type { RadialSlice, SubOptionChoice } from "./gamepadRadialMenuData";
import {
  TOOL_SLICES,
  getSubOptionSlices,
  subOptionSliceToChoice,
} from "./gamepadRadialMenuData";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STICK_DEADZONE = 0.15;
const TRIGGER_OPEN_THRESHOLD = 0.5;
const TRIGGER_CLOSE_THRESHOLD = 0.3;
const GAMEPAD_LOOK_SPEED = 10.0; // px/frame at full stick deflection
const RADIAL_STICK_DEADZONE = 0.3;

const SPEED_LEVELS = [1, 2, 4, 8] as const;

// Virtual cursor
const CURSOR_SPEED_MAX = 14; // CSS px/frame at full deflection
const CURSOR_SCROLL_SPEED = 12; // px/frame at full trigger

// Standard gamepad button indices (W3C "standard" mapping)
const BTN_B = 1;
const BTN_X = 2;
const BTN_Y = 3;
const BTN_LB = 4;
const BTN_RB = 5;
const BTN_LT = 6;
const BTN_RT = 7;
const BTN_DPAD_UP = 12;
const BTN_DPAD_DOWN = 13;
const BTN_DPAD_LEFT = 14;
const BTN_DPAD_RIGHT = 15;

// Synthetic pointer ID for virtual cursor events
const VIRTUAL_POINTER_ID = 9999;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GamepadOpts {
  flyPendingLookDxRef: MutableRefObject<number>;
  flyPendingLookDyRef: MutableRefObject<number>;
  onToolActivate: () => void;
  onEyedropper: () => void;
  onUndo: () => void;
  onToolSelect: (sliceId: string) => void;
  onSubOptionSelect: (choice: SubOptionChoice) => void;
  onRequestFlyMode: () => void;
  onToggleLocomotion: (direction: "fly" | "walk") => void;
  interactionModeRef: MutableRefObject<InteractionMode>;
  /** Ref to the virtual cursor DOM element — positioned directly for zero-lag movement. */
  cursorElRef: MutableRefObject<HTMLDivElement | null>;
}

export interface GamepadFrameOutput {
  forward: number;
  right: number;
  up: number;
  speedScale: number;
}

export interface GamepadHandle {
  connected: boolean;
  cursorMode: boolean;
  radialMenu: "tools" | "subOptions" | null;
  radialSlices: RadialSlice[];
  selectedSliceIndex: number | null;
  speedMultiplier: number;
  pollGamepad: () => GamepadFrameOutput;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Apply radial deadzone and re-normalize. Returns [x, y, magnitude]. */
function applyDeadzone(
  rawX: number,
  rawY: number,
  deadzone: number,
): [number, number, number] {
  const mag = Math.sqrt(rawX * rawX + rawY * rawY);
  if (mag < deadzone) return [0, 0, 0];
  const normMag = Math.min((mag - deadzone) / (1.0 - deadzone), 1.0);
  const scale = normMag / mag;
  return [rawX * scale, rawY * scale, normMag];
}

function buttonPressed(gp: Gamepad, idx: number): boolean {
  const btn = gp.buttons[idx];
  if (!btn) return false;
  return btn.pressed || btn.value > 0.5;
}

function triggerValue(gp: Gamepad, idx: number): number {
  const btn = gp.buttons[idx];
  if (!btn) return 0;
  return btn.value;
}

/** Angle from stick deflection, 0 = up, clockwise. Returns null in deadzone. */
function stickAngle(
  x: number,
  y: number,
  deadzone: number,
): number | null {
  const mag = Math.sqrt(x * x + y * y);
  if (mag < deadzone) return null;
  let a = Math.atan2(x, -y); // 0 = up, positive = clockwise
  if (a < 0) a += 2 * Math.PI;
  return a;
}

function sliceIndexFromAngle(
  angle: number | null,
  sliceCount: number,
): number | null {
  if (angle === null || sliceCount === 0) return null;
  const sliceSize = (2 * Math.PI) / sliceCount;
  return Math.floor(angle / sliceSize);
}

/** Dispatch a synthetic pointer event on the element under (cx, cy). */
function dispatchPointerAt(
  type: "pointerdown" | "pointermove" | "pointerup",
  cx: number,
  cy: number,
  button: number,
  buttons: number,
) {
  const target = document.elementFromPoint(cx, cy);
  if (!target) return;
  target.dispatchEvent(
    new PointerEvent(type, {
      clientX: cx,
      clientY: cy,
      screenX: cx,
      screenY: cy,
      button,
      buttons,
      pointerId: VIRTUAL_POINTER_ID,
      pointerType: "mouse",
      isPrimary: true,
      bubbles: true,
      cancelable: true,
      composed: true,
    }),
  );
}

/** Dispatch a click (mousedown + mouseup + click) at (cx, cy). */
function dispatchClickAt(cx: number, cy: number, button: number) {
  const target = document.elementFromPoint(cx, cy);
  if (!target) return;
  const shared = {
    clientX: cx,
    clientY: cy,
    screenX: cx,
    screenY: cy,
    button,
    bubbles: true,
    cancelable: true,
    composed: true,
  };
  target.dispatchEvent(new MouseEvent("mousedown", { ...shared, buttons: button === 0 ? 1 : 2 }));
  target.dispatchEvent(new MouseEvent("mouseup", { ...shared, buttons: 0 }));
  target.dispatchEvent(new MouseEvent("click", shared));
}

/** Dispatch a wheel event at (cx, cy). */
function dispatchScrollAt(cx: number, cy: number, deltaY: number) {
  const target = document.elementFromPoint(cx, cy);
  if (!target) return;
  target.dispatchEvent(
    new WheelEvent("wheel", {
      clientX: cx,
      clientY: cy,
      deltaY,
      deltaMode: 0, // pixels
      bubbles: true,
      cancelable: true,
      composed: true,
    }),
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useGamepad(opts: GamepadOpts): GamepadHandle {
  const {
    flyPendingLookDxRef,
    flyPendingLookDyRef,
    onToolActivate,
    onEyedropper,
    onUndo,
    onToolSelect,
    onSubOptionSelect,
    onRequestFlyMode,
    onToggleLocomotion,
    interactionModeRef,
    cursorElRef,
  } = opts;

  const [connected, setConnected] = useState(false);
  const [cursorMode, setCursorMode] = useState(false);
  const [radialMenu, setRadialMenu] = useState<"tools" | "subOptions" | null>(
    null,
  );
  const [radialSlices, setRadialSlices] = useState<RadialSlice[]>([]);
  const [selectedSliceIndex, setSelectedSliceIndex] = useState<number | null>(
    null,
  );
  const [speedMultiplier, setSpeedMultiplier] = useState(1);

  const gamepadIndexRef = useRef<number | null>(null);
  const prevButtonsRef = useRef<boolean[]>([]);
  const speedLevelIndexRef = useRef(0);
  const hasRequestedFlyRef = useRef(false);

  // Virtual cursor position (CSS pixels, updated directly — no React state)
  const cursorXRef = useRef(0);
  const cursorYRef = useRef(0);
  const cursorModeRef = useRef(false);
  // Track whether RB/LB are held for pointer drag sequences in cursor mode
  const cursorPointerDownRef = useRef(false);

  // Radial state refs (so pollGamepad reads current values without re-render)
  const radialMenuRef = useRef<"tools" | "subOptions" | null>(null);
  const radialSlicesRef = useRef<RadialSlice[]>([]);
  const selectedSliceIndexRef = useRef<number | null>(null);

  // Keep refs in sync with state
  const setRadialMenuSync = useCallback(
    (v: "tools" | "subOptions" | null) => {
      radialMenuRef.current = v;
      setRadialMenu(v);
    },
    [],
  );
  const setRadialSlicesSync = useCallback((v: RadialSlice[]) => {
    radialSlicesRef.current = v;
    setRadialSlices(v);
  }, []);
  const setSelectedSliceIndexSync = useCallback((v: number | null) => {
    selectedSliceIndexRef.current = v;
    setSelectedSliceIndex(v);
  }, []);

  const setCursorModeSync = useCallback((v: boolean) => {
    cursorModeRef.current = v;
    setCursorMode(v);
    if (v) {
      // Initialize cursor at screen center
      cursorXRef.current = window.innerWidth / 2;
      cursorYRef.current = window.innerHeight / 2;
      const el = cursorElRef.current;
      if (el) {
        el.style.transform = `translate(${cursorXRef.current - 12}px, ${cursorYRef.current - 12}px)`;
      }
    }
    // Clean up any in-progress pointer on exit
    if (!v && cursorPointerDownRef.current) {
      dispatchPointerAt(
        "pointerup",
        cursorXRef.current,
        cursorYRef.current,
        0,
        0,
      );
      cursorPointerDownRef.current = false;
    }
  }, [cursorElRef]);

  // -------------------------------------------------------------------------
  // Connection events
  // -------------------------------------------------------------------------
  useEffect(() => {
    const onConnect = (e: GamepadEvent) => {
      if (e.gamepad.mapping === "standard" || gamepadIndexRef.current === null) {
        gamepadIndexRef.current = e.gamepad.index;
        setConnected(true);
        hasRequestedFlyRef.current = false;
      }
    };
    const onDisconnect = (e: GamepadEvent) => {
      if (e.gamepad.index === gamepadIndexRef.current) {
        gamepadIndexRef.current = null;
        setConnected(false);
        setCursorModeSync(false);
        setRadialMenuSync(null);
        setRadialSlicesSync([]);
        setSelectedSliceIndexSync(null);
        setSpeedMultiplier(1);
        speedLevelIndexRef.current = 0;
        prevButtonsRef.current = [];
        hasRequestedFlyRef.current = false;
      }
    };
    window.addEventListener("gamepadconnected", onConnect);
    window.addEventListener("gamepaddisconnected", onDisconnect);

    // Check if a gamepad is already connected
    const gamepads = navigator.getGamepads();
    for (const gp of gamepads) {
      if (gp && (gp.mapping === "standard" || gamepadIndexRef.current === null)) {
        gamepadIndexRef.current = gp.index;
        setConnected(true);
        break;
      }
    }

    return () => {
      window.removeEventListener("gamepadconnected", onConnect);
      window.removeEventListener("gamepaddisconnected", onDisconnect);
    };
  }, [setCursorModeSync, setRadialMenuSync, setRadialSlicesSync, setSelectedSliceIndexSync]);

  // -------------------------------------------------------------------------
  // pollGamepad — called from RAF tick
  // -------------------------------------------------------------------------
  const pollGamepad = useCallback((): GamepadFrameOutput => {
    const zero: GamepadFrameOutput = {
      forward: 0,
      right: 0,
      up: 0,
      speedScale: 1,
    };
    if (gamepadIndexRef.current === null) return zero;

    const gp = navigator.getGamepads()[gamepadIndexRef.current];
    if (!gp) return zero;

    // ----- Edge detection setup -----
    const prev = prevButtonsRef.current;
    const curr: boolean[] = [];
    for (let i = 0; i < gp.buttons.length; i++) {
      curr[i] = buttonPressed(gp, i);
    }
    const rising = (idx: number) => curr[idx] && !prev[idx];
    const falling = (idx: number) => !curr[idx] && prev[idx];
    prevButtonsRef.current = curr;

    // ----- X button: toggle cursor mode -----
    if (rising(BTN_X)) {
      setCursorModeSync(!cursorModeRef.current);
      return zero; // consume this frame
    }

    // ----- Sticks with deadzone -----
    const [lx, ly] = applyDeadzone(gp.axes[0] ?? 0, gp.axes[1] ?? 0, STICK_DEADZONE);
    const [rx, ry] = applyDeadzone(gp.axes[2] ?? 0, gp.axes[3] ?? 0, STICK_DEADZONE);

    // =======================================================================
    // CURSOR MODE — sticks drive cursor, bumpers click, triggers scroll
    // =======================================================================
    if (cursorModeRef.current) {
      // B also exits cursor mode
      if (rising(BTN_B)) {
        setCursorModeSync(false);
        return zero;
      }

      // Move cursor with left stick (quadratic acceleration curve)
      if (lx !== 0 || ly !== 0) {
        const mag = Math.sqrt(lx * lx + ly * ly);
        const speed = mag * mag * CURSOR_SPEED_MAX; // quadratic
        const nx = lx / mag;
        const ny = ly / mag;
        cursorXRef.current = Math.max(0, Math.min(window.innerWidth - 1, cursorXRef.current + nx * speed));
        cursorYRef.current = Math.max(0, Math.min(window.innerHeight - 1, cursorYRef.current + ny * speed));
      }

      // Update cursor element position directly (no React state)
      const el = cursorElRef.current;
      if (el) {
        el.style.transform = `translate(${cursorXRef.current - 12}px, ${cursorYRef.current - 12}px)`;
      }

      const cx = cursorXRef.current;
      const cy = cursorYRef.current;

      // RB = left click (pointer down/up sequence)
      if (rising(BTN_RB)) {
        dispatchPointerAt("pointerdown", cx, cy, 0, 1);
        cursorPointerDownRef.current = true;
      }
      if (cursorPointerDownRef.current && (lx !== 0 || ly !== 0) && curr[BTN_RB]) {
        // Pointer drag while RB held and cursor moving
        dispatchPointerAt("pointermove", cx, cy, 0, 1);
      }
      if (falling(BTN_RB)) {
        dispatchPointerAt("pointerup", cx, cy, 0, 0);
        cursorPointerDownRef.current = false;
        // Also fire a click for simple tap interactions (buttons, links)
        dispatchClickAt(cx, cy, 0);
      }

      // LB = right click
      if (rising(BTN_LB)) {
        dispatchPointerAt("pointerdown", cx, cy, 2, 2);
      }
      if (falling(BTN_LB)) {
        dispatchPointerAt("pointerup", cx, cy, 2, 0);
        dispatchClickAt(cx, cy, 2);
      }

      // Triggers = scroll
      const ltVal = triggerValue(gp, BTN_LT);
      const rtVal = triggerValue(gp, BTN_RT);
      const scrollDelta = (rtVal - ltVal) * CURSOR_SCROLL_SPEED;
      if (Math.abs(scrollDelta) > 0.5) {
        dispatchScrollAt(cx, cy, scrollDelta);
      }

      // Right stick = fine-tune cursor (slower, for precision)
      if (rx !== 0 || ry !== 0) {
        cursorXRef.current = Math.max(0, Math.min(window.innerWidth - 1, cursorXRef.current + rx * 3));
        cursorYRef.current = Math.max(0, Math.min(window.innerHeight - 1, cursorYRef.current + ry * 3));
      }

      // Y = undo still works in cursor mode
      if (rising(BTN_Y)) onUndo();

      // Suppress all camera movement
      return zero;
    }

    // =======================================================================
    // NORMAL MODE — camera controls, radials, tools
    // =======================================================================

    // ----- Auto-enter fly mode on first meaningful input -----
    if (
      !hasRequestedFlyRef.current &&
      (lx !== 0 || ly !== 0 || rx !== 0 || ry !== 0)
    ) {
      hasRequestedFlyRef.current = true;
      onRequestFlyMode();
    }

    // ----- Trigger radial menus -----
    const ltVal = triggerValue(gp, BTN_LT);
    const rtVal = triggerValue(gp, BTN_RT);
    const currentRadial = radialMenuRef.current;

    // Open LT radial
    if (currentRadial === null && ltVal > TRIGGER_OPEN_THRESHOLD) {
      setRadialMenuSync("tools");
      setRadialSlicesSync(TOOL_SLICES);
      setSelectedSliceIndexSync(null);
    }
    // Open RT radial
    if (currentRadial === null && rtVal > TRIGGER_OPEN_THRESHOLD) {
      const im = interactionModeRef.current;
      const slices = getSubOptionSlices(im);
      if (slices.length > 0) {
        setRadialMenuSync("subOptions");
        setRadialSlicesSync(slices);
        setSelectedSliceIndexSync(null);
      }
    }

    // Close LT radial on release
    if (currentRadial === "tools" && ltVal < TRIGGER_CLOSE_THRESHOLD) {
      const idx = selectedSliceIndexRef.current;
      const slices = radialSlicesRef.current;
      if (idx !== null && slices[idx]) {
        onToolSelect(slices[idx].id);
      }
      setRadialMenuSync(null);
      setRadialSlicesSync([]);
      setSelectedSliceIndexSync(null);
    }
    // Close RT radial on release
    if (currentRadial === "subOptions" && rtVal < TRIGGER_CLOSE_THRESHOLD) {
      const idx = selectedSliceIndexRef.current;
      const slices = radialSlicesRef.current;
      if (idx !== null && slices[idx]) {
        const im = interactionModeRef.current;
        const choice = subOptionSliceToChoice(slices[idx].id, im);
        if (choice) onSubOptionSelect(choice);
      }
      setRadialMenuSync(null);
      setRadialSlicesSync([]);
      setSelectedSliceIndexSync(null);
    }

    // ----- Radial selection via right stick -----
    if (radialMenuRef.current !== null) {
      const angle = stickAngle(rx, ry, RADIAL_STICK_DEADZONE);
      const slices = radialSlicesRef.current;
      setSelectedSliceIndexSync(sliceIndexFromAngle(angle, slices.length));

      // While radial is open: suppress movement + look, suppress bumpers
      return zero;
    }

    // ----- Discrete button actions (only when no radial open) -----
    if (rising(BTN_RB)) onToolActivate();
    if (rising(BTN_LB)) onEyedropper();
    if (rising(BTN_Y)) onUndo();

    // D-pad: locomotion toggle
    if (rising(BTN_DPAD_LEFT)) onToggleLocomotion("walk");
    if (rising(BTN_DPAD_RIGHT)) onToggleLocomotion("fly");

    // D-pad: speed cycle
    if (rising(BTN_DPAD_UP)) {
      const idx = Math.min(speedLevelIndexRef.current + 1, SPEED_LEVELS.length - 1);
      speedLevelIndexRef.current = idx;
      setSpeedMultiplier(SPEED_LEVELS[idx]);
    }
    if (rising(BTN_DPAD_DOWN)) {
      const idx = Math.max(speedLevelIndexRef.current - 1, 0);
      speedLevelIndexRef.current = idx;
      setSpeedMultiplier(SPEED_LEVELS[idx]);
    }

    // ----- Right stick → look deltas -----
    if (rx !== 0 || ry !== 0) {
      flyPendingLookDxRef.current += rx * GAMEPAD_LOOK_SPEED;
      flyPendingLookDyRef.current += ry * GAMEPAD_LOOK_SPEED;
    }

    // ----- Left stick → movement -----
    // Stick Y: up = negative = forward
    const forward = -ly;
    const right = lx;

    return {
      forward,
      right,
      up: 0,
      speedScale: SPEED_LEVELS[speedLevelIndexRef.current],
    };
  }, [
    flyPendingLookDxRef,
    flyPendingLookDyRef,
    onToolActivate,
    onEyedropper,
    onUndo,
    onToolSelect,
    onSubOptionSelect,
    onRequestFlyMode,
    onToggleLocomotion,
    interactionModeRef,
    cursorElRef,
    setCursorModeSync,
    setRadialMenuSync,
    setRadialSlicesSync,
    setSelectedSliceIndexSync,
  ]);

  return {
    connected,
    cursorMode,
    radialMenu,
    radialSlices,
    selectedSliceIndex,
    speedMultiplier,
    pollGamepad,
  };
}
