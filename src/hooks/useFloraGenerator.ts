import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface FloraGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
}

export interface FloraGeneratorState {
  floraPreset: string;
  setFloraPreset: React.Dispatch<React.SetStateAction<string>>;
  floraHeight: number;
  setFloraHeight: React.Dispatch<React.SetStateAction<number>>;
  floraGirth: number;
  setFloraGirth: React.Dispatch<React.SetStateAction<number>>;
  floraWobble: number;
  setFloraWobble: React.Dispatch<React.SetStateAction<number>>;
  floraTaper: number;
  setFloraTaper: React.Dispatch<React.SetStateAction<number>>;
  floraStemCount: number;
  setFloraStemCount: React.Dispatch<React.SetStateAction<number>>;
  floraClusterRadius: number;
  setFloraClusterRadius: React.Dispatch<React.SetStateAction<number>>;
  floraBranchCount: number;
  setFloraBranchCount: React.Dispatch<React.SetStateAction<number>>;
  floraBranchDepth: number;
  setFloraBranchDepth: React.Dispatch<React.SetStateAction<number>>;
  floraBranchStart: number;
  setFloraBranchStart: React.Dispatch<React.SetStateAction<number>>;
  floraBranchSpread: number;
  setFloraBranchSpread: React.Dispatch<React.SetStateAction<number>>;
  floraBraidStrands: number;
  setFloraBraidStrands: React.Dispatch<React.SetStateAction<number>>;
  floraBraidTwist: number;
  setFloraBraidTwist: React.Dispatch<React.SetStateAction<number>>;
  floraCanopy: number;
  setFloraCanopy: React.Dispatch<React.SetStateAction<number>>;
  floraPreviewSeedRef: React.MutableRefObject<number>;
  floraPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function useFloraGenerator(ctx: FloraGeneratorContext): FloraGeneratorState {
  const [floraPreset, setFloraPreset] = useState<string>("stalk");
  const [floraHeight, setFloraHeight] = useState(14);
  const [floraGirth, setFloraGirth] = useState(0);
  const [floraWobble, setFloraWobble] = useState(0.12);
  const [floraTaper, setFloraTaper] = useState(0.12);
  const [floraStemCount, setFloraStemCount] = useState(1);
  const [floraClusterRadius, setFloraClusterRadius] = useState(0);
  const [floraBranchCount, setFloraBranchCount] = useState(0);
  const [floraBranchDepth, setFloraBranchDepth] = useState(1);
  const [floraBranchStart, setFloraBranchStart] = useState(0.5);
  const [floraBranchSpread, setFloraBranchSpread] = useState(1.0);
  const [floraBraidStrands, setFloraBraidStrands] = useState(1);
  const [floraBraidTwist, setFloraBraidTwist] = useState(0.35);
  const [floraCanopy, setFloraCanopy] = useState(0.18);

  const floraPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  // Flora uses state values directly in onCommit (not refs), matching the original code.
  const floraPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_flora_at_screen", {
        args: {
          nx,
          ny,
          seed,
          height: floraHeight,
          girth: floraGirth,
          wobble: floraWobble,
          taper: floraTaper,
          stemCount: floraStemCount,
          clusterRadius: floraClusterRadius,
          branchCount: floraBranchCount,
          branchDepth: floraBranchDepth,
          branchStart: floraBranchStart,
          branchSpread: floraBranchSpread,
          braidStrands: floraBraidStrands,
          braidTwist: floraBraidTwist,
          canopy: floraCanopy,
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
        },
      }).catch(() => {});
      floraPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });

  return {
    floraPreset, setFloraPreset,
    floraHeight, setFloraHeight,
    floraGirth, setFloraGirth,
    floraWobble, setFloraWobble,
    floraTaper, setFloraTaper,
    floraStemCount, setFloraStemCount,
    floraClusterRadius, setFloraClusterRadius,
    floraBranchCount, setFloraBranchCount,
    floraBranchDepth, setFloraBranchDepth,
    floraBranchStart, setFloraBranchStart,
    floraBranchSpread, setFloraBranchSpread,
    floraBraidStrands, setFloraBraidStrands,
    floraBraidTwist, setFloraBraidTwist,
    floraCanopy, setFloraCanopy,
    floraPreviewSeedRef,
    floraPhase,
  };
}
