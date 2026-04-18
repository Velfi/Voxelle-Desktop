import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";
import { mirrorAxesFromRefs } from "../symmetry";
import type { StartShapeId } from "../types";
import { useLatestRef } from "./useLatestRef";

interface ShapeGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  mirrorXRef: React.MutableRefObject<boolean>;
  mirrorYRef: React.MutableRefObject<boolean>;
  mirrorZRef: React.MutableRefObject<boolean>;
}

export interface ShapeGeneratorState {
  shapeKind: StartShapeId;
  setShapeKind: React.Dispatch<React.SetStateAction<StartShapeId>>;
  shapeKindRef: React.MutableRefObject<StartShapeId>;
  shapeSize: number;
  setShapeSize: React.Dispatch<React.SetStateAction<number>>;
  shapeSizeRef: React.MutableRefObject<number>;
  shapeRotX: number;
  setShapeRotX: React.Dispatch<React.SetStateAction<number>>;
  shapeRotXRef: React.MutableRefObject<number>;
  shapeRotY: number;
  setShapeRotY: React.Dispatch<React.SetStateAction<number>>;
  shapeRotYRef: React.MutableRefObject<number>;
  shapeRotZ: number;
  setShapeRotZ: React.Dispatch<React.SetStateAction<number>>;
  shapeRotZRef: React.MutableRefObject<number>;
  shapeOverwrite: boolean;
  setShapeOverwrite: React.Dispatch<React.SetStateAction<boolean>>;
  shapeOverwriteRef: React.MutableRefObject<boolean>;
  shapePhase: StrokePhaseHandle<{ nx: number; ny: number }>;
  /** World-space gizmo position (set during settings phase). */
  shapeGizmoPosRef: React.MutableRefObject<[number, number, number] | null>;
}

export function useShapeGenerator(ctx: ShapeGeneratorContext): ShapeGeneratorState {
  const [shapeKind, setShapeKind] = useState<StartShapeId>("cube");
  const [shapeSize, setShapeSize] = useState(8);
  const [shapeRotX, setShapeRotX] = useState(0);
  const [shapeRotY, setShapeRotY] = useState(0);
  const [shapeRotZ, setShapeRotZ] = useState(0);
  const [shapeOverwrite, setShapeOverwrite] = useState(true);

  const shapeKindRef = useLatestRef(shapeKind);
  const shapeSizeRef = useLatestRef(shapeSize);
  const shapeRotXRef = useLatestRef(shapeRotX);
  const shapeRotYRef = useLatestRef(shapeRotY);
  const shapeRotZRef = useLatestRef(shapeRotZ);
  const shapeOverwriteRef = useLatestRef(shapeOverwrite);
  const shapeGizmoPosRef = useRef<[number, number, number] | null>(null);

  // Listen for gizmo-moved events from Rust.
  useEffect(() => {
    const unlisten = listen<[number, number, number]>("generator-gizmo-moved", (ev) => {
      shapeGizmoPosRef.current = ev.payload;
    });
    return () => {
      void unlisten.then((u) => u());
    };
  }, []);

  // Listen for gizmo-rotated events from Rust (ring drag during shape phase).
  // Payload is [axis: 0=X 1=Y 2=Z, degrees].
  useEffect(() => {
    const unlisten = listen<[number, number]>("generator-gizmo-rotated", (ev) => {
      const [axis, degrees] = ev.payload;
      const clamp = (v: number) => ((v % 360) + 360) % 360;
      if (axis === 0) setShapeRotX((prev) => clamp(prev + degrees));
      else if (axis === 1) setShapeRotY((prev) => clamp(prev + degrees));
      else setShapeRotZ((prev) => clamp(prev + degrees));
    });
    return () => {
      void unlisten.then((u) => u());
    };
  }, []);

  const clearGizmo = () => {
    shapeGizmoPosRef.current = null;
    void invoke("clear_generator_gizmo_center").catch(() => {});
  };

  const shapePhase = useStrokePhase<{ nx: number; ny: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
      clearGizmo();
    },
    onCommit: (snap) => {
      const { nx, ny } = snap.data;
      const gizmoPos = shapeGizmoPosRef.current;
      void invoke("generator_shape_at_screen", {
        args: {
          nx,
          ny,
          shape: shapeKindRef.current,
          size: shapeSizeRef.current,
          rotX: shapeRotXRef.current,
          rotY: shapeRotYRef.current,
          rotZ: shapeRotZRef.current,
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
          mirrorAxes: mirrorAxesFromRefs(ctx.mirrorXRef, ctx.mirrorYRef, ctx.mirrorZRef),
          overwrite: shapeOverwriteRef.current,
          gizmoCenter: gizmoPos,
        },
      }).finally(() => {
        void invoke("unlock_generator_preview_camera").catch(() => {});
        clearGizmo();
      });
    },
  });

  return {
    shapeKind,
    setShapeKind,
    shapeKindRef,
    shapeSize,
    setShapeSize,
    shapeSizeRef,
    shapeRotX,
    setShapeRotX,
    shapeRotXRef,
    shapeRotY,
    setShapeRotY,
    shapeRotYRef,
    shapeRotZ,
    setShapeRotZ,
    shapeRotZRef,
    shapeOverwrite,
    setShapeOverwrite,
    shapeOverwriteRef,
    shapePhase,
    shapeGizmoPosRef,
  };
}
