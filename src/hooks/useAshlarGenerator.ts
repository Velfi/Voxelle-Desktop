import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";
import { mirrorAxesFromRefs } from "../symmetry";
import { useLatestRef } from "./useLatestRef";

interface AshlarGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  generatorSphereRadiusRef: React.MutableRefObject<number>;
  mirrorXRef: React.MutableRefObject<boolean>;
  mirrorYRef: React.MutableRefObject<boolean>;
  mirrorZRef: React.MutableRefObject<boolean>;
  /** Shared with rocks generator. */
  rockRoughnessRef: React.MutableRefObject<number>;
}

export interface AshlarGeneratorState {
  ashlarAutoCommitOnMouseUp: boolean;
  setAshlarAutoCommitOnMouseUp: React.Dispatch<React.SetStateAction<boolean>>;
  ashlarAutoCommitOnMouseUpRef: React.MutableRefObject<boolean>;
  ashlarThickness: number;
  setAshlarThickness: React.Dispatch<React.SetStateAction<number>>;
  ashlarThicknessRef: React.MutableRefObject<number>;
  ashlarPreviewSeedRef: React.MutableRefObject<number>;
  placeAshlarAtScreen: (nx: number, ny: number, seed?: number) => void;
  ashlarPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function useAshlarGenerator(ctx: AshlarGeneratorContext): AshlarGeneratorState {
  const [ashlarAutoCommitOnMouseUp, setAshlarAutoCommitOnMouseUp] = useState(true);
  const [ashlarThickness, setAshlarThickness] = useState(3);

  const ashlarAutoCommitOnMouseUpRef = useLatestRef(ashlarAutoCommitOnMouseUp);
  const ashlarThicknessRef = useLatestRef(ashlarThickness);
  const ashlarPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  function placeAshlarAtScreen(nx: number, ny: number, seed = ashlarPreviewSeedRef.current) {
    void invoke("generator_ashlar_at_screen", {
      args: {
        nx,
        ny,
        seed,
        size: Math.max(1, ctx.generatorSphereRadiusRef.current),
        roughness: ctx.rockRoughnessRef.current,
        color: ctx.activeColorRef.current,
        material: ctx.activeMaterialRef.current,
        mirrorAxes: mirrorAxesFromRefs(ctx.mirrorXRef, ctx.mirrorYRef, ctx.mirrorZRef),
        thickness: ashlarThicknessRef.current,
      },
    })
      .catch((err: unknown) => {
        console.error("[ashlar] placement failed:", err);
      })
      .finally(() => {
        void invoke("unlock_generator_preview_camera").catch(() => {});
      });
    ashlarPreviewSeedRef.current = (Math.random() * 1e9) | 0;
  }

  const ashlarPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      const { nx, ny, seed } = snap.data;
      placeAshlarAtScreen(nx, ny, seed);
    },
  });

  return {
    ashlarAutoCommitOnMouseUp,
    setAshlarAutoCommitOnMouseUp,
    ashlarAutoCommitOnMouseUpRef,
    ashlarThickness,
    setAshlarThickness,
    ashlarThicknessRef,
    ashlarPreviewSeedRef,
    placeAshlarAtScreen,
    ashlarPhase,
  };
}
