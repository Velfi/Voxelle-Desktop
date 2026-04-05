import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";
import type { StartShapeId } from "../types";

interface ShapeGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
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

  const shapeKindRef = useRef(shapeKind);
  const shapeSizeRef = useRef(shapeSize);
  const shapeRotXRef = useRef(shapeRotX);
  const shapeRotYRef = useRef(shapeRotY);
  const shapeRotZRef = useRef(shapeRotZ);
  const shapeOverwriteRef = useRef(shapeOverwrite);
  const shapeGizmoPosRef = useRef<[number, number, number] | null>(null);

  useEffect(() => { shapeKindRef.current = shapeKind; }, [shapeKind]);
  useEffect(() => { shapeSizeRef.current = shapeSize; }, [shapeSize]);
  useEffect(() => { shapeRotXRef.current = shapeRotX; }, [shapeRotX]);
  useEffect(() => { shapeRotYRef.current = shapeRotY; }, [shapeRotY]);
  useEffect(() => { shapeRotZRef.current = shapeRotZ; }, [shapeRotZ]);
  useEffect(() => { shapeOverwriteRef.current = shapeOverwrite; }, [shapeOverwrite]);

  // Listen for gizmo-moved events from Rust.
  useEffect(() => {
    const unlisten = listen<[number, number, number]>("generator-gizmo-moved", (ev) => {
      shapeGizmoPosRef.current = ev.payload;
    });
    return () => { void unlisten.then((u) => u()); };
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
    return () => { void unlisten.then((u) => u()); };
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
      void invoke("unlock_generator_preview_camera").catch(() => {});
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
          overwrite: shapeOverwriteRef.current,
          gizmoCenter: gizmoPos,
        },
      }).finally(() => clearGizmo());
    },
  });

  return {
    shapeKind, setShapeKind, shapeKindRef,
    shapeSize, setShapeSize, shapeSizeRef,
    shapeRotX, setShapeRotX, shapeRotXRef,
    shapeRotY, setShapeRotY, shapeRotYRef,
    shapeRotZ, setShapeRotZ, shapeRotZRef,
    shapeOverwrite, setShapeOverwrite, shapeOverwriteRef,
    shapePhase,
    shapeGizmoPosRef,
  };
}
