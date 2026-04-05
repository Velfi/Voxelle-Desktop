import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface AshlarGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  generatorSphereRadiusRef: React.MutableRefObject<number>;
  /** Shared with rocks generator. */
  rockRoughnessRef: React.MutableRefObject<number>;
}

export interface AshlarGeneratorState {
  ashlarThickness: number;
  setAshlarThickness: React.Dispatch<React.SetStateAction<number>>;
  ashlarThicknessRef: React.MutableRefObject<number>;
  ashlarPreviewSeedRef: React.MutableRefObject<number>;
  ashlarPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function useAshlarGenerator(ctx: AshlarGeneratorContext): AshlarGeneratorState {
  const [ashlarThickness, setAshlarThickness] = useState(3);

  const ashlarThicknessRef = useRef(ashlarThickness);
  const ashlarPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  useEffect(() => {
    ashlarThicknessRef.current = ashlarThickness;
  }, [ashlarThickness]);

  const ashlarPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_ashlar_at_screen", {
        args: {
          nx,
          ny,
          seed,
          size: Math.max(1, ctx.generatorSphereRadiusRef.current),
          roughness: ctx.rockRoughnessRef.current,
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
          thickness: ashlarThicknessRef.current,
        },
      }).catch((err: unknown) => {
        console.error("[ashlar] placement failed:", err);
      });
      ashlarPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });

  return {
    ashlarThickness,
    setAshlarThickness,
    ashlarThicknessRef,
    ashlarPreviewSeedRef,
    ashlarPhase,
  };
}
