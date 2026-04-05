import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface GrassGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  generatorSphereRadiusRef: React.MutableRefObject<number>;
}

export interface GrassGeneratorState {
  grassDensity: number;
  setGrassDensity: React.Dispatch<React.SetStateAction<number>>;
  grassDensityRef: React.MutableRefObject<number>;
  grassMaxHeight: number;
  setGrassMaxHeight: React.Dispatch<React.SetStateAction<number>>;
  grassMaxHeightRef: React.MutableRefObject<number>;
  grassPreviewSeedRef: React.MutableRefObject<number>;
  grassPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function useGrassGenerator(ctx: GrassGeneratorContext): GrassGeneratorState {
  const [grassDensity, setGrassDensity] = useState(0.6);
  const [grassMaxHeight, setGrassMaxHeight] = useState(3);

  const grassDensityRef = useRef(grassDensity);
  const grassMaxHeightRef = useRef(grassMaxHeight);
  const grassPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  useEffect(() => {
    grassDensityRef.current = grassDensity;
  }, [grassDensity]);
  useEffect(() => {
    grassMaxHeightRef.current = grassMaxHeight;
  }, [grassMaxHeight]);

  const grassPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
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
        },
      }).catch(() => {});
      grassPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });

  return {
    grassDensity,
    setGrassDensity,
    grassDensityRef,
    grassMaxHeight,
    setGrassMaxHeight,
    grassMaxHeightRef,
    grassPreviewSeedRef,
    grassPhase,
  };
}
