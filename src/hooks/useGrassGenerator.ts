import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";
import { mirrorAxesFromRefs } from "../symmetry";
import { useLatestRef } from "./useLatestRef";

interface GrassGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  generatorSphereRadiusRef: React.MutableRefObject<number>;
  mirrorXRef: React.MutableRefObject<boolean>;
  mirrorYRef: React.MutableRefObject<boolean>;
  mirrorZRef: React.MutableRefObject<boolean>;
}

export interface GrassGeneratorState {
  grassAutoCommitOnMouseUp: boolean;
  setGrassAutoCommitOnMouseUp: React.Dispatch<React.SetStateAction<boolean>>;
  grassAutoCommitOnMouseUpRef: React.MutableRefObject<boolean>;
  grassDensity: number;
  setGrassDensity: React.Dispatch<React.SetStateAction<number>>;
  grassDensityRef: React.MutableRefObject<number>;
  grassMaxHeight: number;
  setGrassMaxHeight: React.Dispatch<React.SetStateAction<number>>;
  grassMaxHeightRef: React.MutableRefObject<number>;
  grassPreviewSeedRef: React.MutableRefObject<number>;
  placeGrassAtScreen: (nx: number, ny: number, seed?: number) => void;
  grassPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function useGrassGenerator(ctx: GrassGeneratorContext): GrassGeneratorState {
  const [grassAutoCommitOnMouseUp, setGrassAutoCommitOnMouseUp] = useState(true);
  const [grassDensity, setGrassDensity] = useState(0.6);
  const [grassMaxHeight, setGrassMaxHeight] = useState(3);

  const grassAutoCommitOnMouseUpRef = useLatestRef(grassAutoCommitOnMouseUp);
  const grassDensityRef = useLatestRef(grassDensity);
  const grassMaxHeightRef = useLatestRef(grassMaxHeight);
  const grassPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  function placeGrassAtScreen(nx: number, ny: number, seed = grassPreviewSeedRef.current) {
    void invoke("generator_grass_at_screen", {
      args: {
        nx,
        ny,
        seed,
        radius: Math.max(1, ctx.generatorSphereRadiusRef.current),
        density: grassDensityRef.current,
        maxHeight: grassMaxHeightRef.current,
        color: ctx.activeColorRef.current,
        material: ctx.activeMaterialRef.current,
        mirrorAxes: mirrorAxesFromRefs(ctx.mirrorXRef, ctx.mirrorYRef, ctx.mirrorZRef),
      },
    })
      .catch(() => {})
      .finally(() => {
        void invoke("unlock_generator_preview_camera").catch(() => {});
      });
    grassPreviewSeedRef.current = (Math.random() * 1e9) | 0;
  }

  const grassPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      const { nx, ny, seed } = snap.data;
      placeGrassAtScreen(nx, ny, seed);
    },
  });

  return {
    grassAutoCommitOnMouseUp,
    setGrassAutoCommitOnMouseUp,
    grassAutoCommitOnMouseUpRef,
    grassDensity,
    setGrassDensity,
    grassDensityRef,
    grassMaxHeight,
    setGrassMaxHeight,
    grassMaxHeightRef,
    grassPreviewSeedRef,
    placeGrassAtScreen,
    grassPhase,
  };
}
