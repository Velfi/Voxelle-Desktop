/**
 * Squishy (metaball) tool: mode, shell options, session phase, and Rust flag sync.
 */

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";
import { useLatestRef } from "./useLatestRef";

export interface SquishyToolContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
}

export interface SquishyToolState {
  squishyMode: "add" | "edit" | "delete";
  setSquishyMode: React.Dispatch<React.SetStateAction<"add" | "edit" | "delete">>;
  squishyModeRef: React.MutableRefObject<"add" | "edit" | "delete">;
  squishyHollow: boolean;
  setSquishyHollow: React.Dispatch<React.SetStateAction<boolean>>;
  squishyWallThickness: number;
  setSquishyWallThickness: React.Dispatch<React.SetStateAction<number>>;
  squishySnapToSurface: boolean;
  setSquishySnapToSurface: React.Dispatch<React.SetStateAction<boolean>>;
  squishyBallCount: number;
  setSquishyBallCount: React.Dispatch<React.SetStateAction<number>>;
  squishyPhase: StrokePhaseHandle<Record<string, never>>;
}

export function useSquishyToolState(ctx: SquishyToolContext): SquishyToolState {
  const { activeColorRef, activeMaterialRef } = ctx;
  const [squishyMode, setSquishyMode] = useState<"add" | "edit" | "delete">("add");
  const squishyModeRef = useLatestRef(squishyMode);
  const [squishyHollow, setSquishyHollow] = useState(false);
  const [squishyWallThickness, setSquishyWallThickness] = useState(1);
  const [squishySnapToSurface, setSquishySnapToSurface] = useState(true);
  const [squishyBallCount, setSquishyBallCount] = useState(0);

  const squishyPhase = useStrokePhase<Record<string, never>>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("squishy_session_clear")
        .then(() => setSquishyBallCount(0))
        .catch(() => {});
    },
    onCommit: () => {
      void invoke("squishy_session_commit", {
        args: {
          color: activeColorRef.current,
          material: activeMaterialRef.current,
        },
      })
        .then(() => invoke("squishy_session_clear"))
        .then(() => setSquishyBallCount(0))
        .catch(() => {});
    },
  });

  useEffect(() => {
    void invoke("squishy_session_set_flags", {
      args: {
        hollow: squishyHollow,
        addSnapToSurface: squishySnapToSurface,
        wallThickness: Math.max(1, squishyWallThickness | 0),
      },
    }).catch(() => {});
  }, [squishyHollow, squishySnapToSurface, squishyWallThickness]);

  return {
    squishyMode,
    setSquishyMode,
    squishyModeRef,
    squishyHollow,
    setSquishyHollow,
    squishyWallThickness,
    setSquishyWallThickness,
    squishySnapToSurface,
    setSquishySnapToSurface,
    squishyBallCount,
    setSquishyBallCount,
    squishyPhase,
  };
}
