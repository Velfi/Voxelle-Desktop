import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface RocksGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  generatorSphereRadiusRef: React.MutableRefObject<number>;
}

export interface RocksGeneratorState {
  rockRoughness: number;
  setRockRoughness: React.Dispatch<React.SetStateAction<number>>;
  rockRoughnessRef: React.MutableRefObject<number>;
  rockCount: number;
  setRockCount: React.Dispatch<React.SetStateAction<number>>;
  rockCountRef: React.MutableRefObject<number>;
  rockClusterRadius: number;
  setRockClusterRadius: React.Dispatch<React.SetStateAction<number>>;
  rockClusterRadiusRef: React.MutableRefObject<number>;
  rockSinkDirection: "none" | "under" | "over";
  setRockSinkDirection: React.Dispatch<React.SetStateAction<"none" | "under" | "over">>;
  rockSinkDirectionRef: React.MutableRefObject<"none" | "under" | "over">;
  rockSinkAmount: number;
  setRockSinkAmount: React.Dispatch<React.SetStateAction<number>>;
  rockSinkAmountRef: React.MutableRefObject<number>;
  rockPreviewSeedRef: React.MutableRefObject<number>;
  rocksPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function useRocksGenerator(ctx: RocksGeneratorContext): RocksGeneratorState {
  const [rockRoughness, setRockRoughness] = useState(0.4);
  const [rockCount, setRockCount] = useState(1);
  const [rockClusterRadius, setRockClusterRadius] = useState(1);
  const [rockSinkDirection, setRockSinkDirection] = useState<"none" | "under" | "over">("none");
  const [rockSinkAmount, setRockSinkAmount] = useState(0);

  const rockRoughnessRef = useRef(rockRoughness);
  const rockCountRef = useRef(rockCount);
  const rockClusterRadiusRef = useRef(rockClusterRadius);
  const rockSinkDirectionRef = useRef(rockSinkDirection);
  const rockSinkAmountRef = useRef(rockSinkAmount);
  const rockPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  useEffect(() => {
    rockRoughnessRef.current = rockRoughness;
  }, [rockRoughness]);
  useEffect(() => {
    rockCountRef.current = rockCount;
  }, [rockCount]);
  useEffect(() => {
    rockClusterRadiusRef.current = rockClusterRadius;
  }, [rockClusterRadius]);
  useEffect(() => {
    rockSinkDirectionRef.current = rockSinkDirection;
  }, [rockSinkDirection]);
  useEffect(() => {
    rockSinkAmountRef.current = rockSinkAmount;
  }, [rockSinkAmount]);

  const rocksPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_rocks_at_screen", {
        args: {
          nx,
          ny,
          seed,
          size: Math.max(1, ctx.generatorSphereRadiusRef.current),
          roughness: rockRoughnessRef.current,
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
          count: rockCountRef.current,
          clusterRadius: rockClusterRadiusRef.current,
          sinkDirection:
            rockSinkDirectionRef.current === "under"
              ? -1
              : rockSinkDirectionRef.current === "over"
                ? 1
                : 0,
          sinkAmount: rockSinkAmountRef.current,
        },
      }).catch(() => {});
      rockPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });

  return {
    rockRoughness,
    setRockRoughness,
    rockRoughnessRef,
    rockCount,
    setRockCount,
    rockCountRef,
    rockClusterRadius,
    setRockClusterRadius,
    rockClusterRadiusRef,
    rockSinkDirection,
    setRockSinkDirection,
    rockSinkDirectionRef,
    rockSinkAmount,
    setRockSinkAmount,
    rockSinkAmountRef,
    rockPreviewSeedRef,
    rocksPhase,
  };
}
