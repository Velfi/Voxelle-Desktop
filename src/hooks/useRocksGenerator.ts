import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";
import { mirrorAxesFromRefs } from "../symmetry";
import { useLatestRef } from "./useLatestRef";

interface RocksGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  generatorSphereRadiusRef: React.MutableRefObject<number>;
  mirrorXRef: React.MutableRefObject<boolean>;
  mirrorYRef: React.MutableRefObject<boolean>;
  mirrorZRef: React.MutableRefObject<boolean>;
}

export interface RocksGeneratorState {
  rocksAutoCommitOnMouseUp: boolean;
  setRocksAutoCommitOnMouseUp: React.Dispatch<React.SetStateAction<boolean>>;
  rocksAutoCommitOnMouseUpRef: React.MutableRefObject<boolean>;
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
  placeRocksAtScreen: (nx: number, ny: number, seed?: number) => void;
  rocksPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function useRocksGenerator(ctx: RocksGeneratorContext): RocksGeneratorState {
  const [rocksAutoCommitOnMouseUp, setRocksAutoCommitOnMouseUp] = useState(true);
  const [rockRoughness, setRockRoughness] = useState(0.4);
  const [rockCount, setRockCount] = useState(1);
  const [rockClusterRadius, setRockClusterRadius] = useState(1);
  const [rockSinkDirection, setRockSinkDirection] = useState<"none" | "under" | "over">("none");
  const [rockSinkAmount, setRockSinkAmount] = useState(0);

  const rocksAutoCommitOnMouseUpRef = useLatestRef(rocksAutoCommitOnMouseUp);
  const rockRoughnessRef = useLatestRef(rockRoughness);
  const rockCountRef = useLatestRef(rockCount);
  const rockClusterRadiusRef = useLatestRef(rockClusterRadius);
  const rockSinkDirectionRef = useLatestRef(rockSinkDirection);
  const rockSinkAmountRef = useLatestRef(rockSinkAmount);
  const rockPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  function placeRocksAtScreen(nx: number, ny: number, seed = rockPreviewSeedRef.current) {
    void invoke("generator_rocks_at_screen", {
      args: {
        nx,
        ny,
        seed,
        size: Math.max(1, ctx.generatorSphereRadiusRef.current),
        roughness: rockRoughnessRef.current,
        color: ctx.activeColorRef.current,
        material: ctx.activeMaterialRef.current,
        mirrorAxes: mirrorAxesFromRefs(ctx.mirrorXRef, ctx.mirrorYRef, ctx.mirrorZRef),
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
    })
      .catch(() => {})
      .finally(() => {
        void invoke("unlock_generator_preview_camera").catch(() => {});
      });
    rockPreviewSeedRef.current = (Math.random() * 1e9) | 0;
  }

  const rocksPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      const { nx, ny, seed } = snap.data;
      placeRocksAtScreen(nx, ny, seed);
    },
  });

  return {
    rocksAutoCommitOnMouseUp,
    setRocksAutoCommitOnMouseUp,
    rocksAutoCommitOnMouseUpRef,
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
    placeRocksAtScreen,
    rocksPhase,
  };
}
