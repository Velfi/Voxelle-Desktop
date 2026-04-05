import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";
import type { ClothGravityDirectionId } from "../types";
import { sculptBrushShapeToRust } from "../constants";

interface RopeClothGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
  selectionStrokeSnapToSurfaceRef: React.MutableRefObject<boolean>;
}

export interface RopeClothGeneratorState {
  // Shared
  clothGravityDirection: ClothGravityDirectionId;
  setClothGravityDirection: React.Dispatch<React.SetStateAction<ClothGravityDirectionId>>;
  clothGravityDirectionRef: React.MutableRefObject<ClothGravityDirectionId>;
  ropeBrushRadiusIndex: number;
  setRopeBrushRadiusIndex: React.Dispatch<React.SetStateAction<number>>;
  ropeBrushRadiusIndexRef: React.MutableRefObject<number>;
  ropeBrushShapeUi: "sphere" | "cube";
  setRopeBrushShapeUi: React.Dispatch<React.SetStateAction<"sphere" | "cube">>;
  ropeBrushShapeUiRef: React.MutableRefObject<"sphere" | "cube">;
  // Rope
  ropeFirstScreen: { nx: number; ny: number } | null;
  setRopeFirstScreen: React.Dispatch<React.SetStateAction<{ nx: number; ny: number } | null>>;
  ropeFirstScreenRef: React.MutableRefObject<{ nx: number; ny: number } | null>;
  ropeFirstVoxel: [number, number, number] | null;
  setRopeFirstVoxel: React.Dispatch<React.SetStateAction<[number, number, number] | null>>;
  ropeFirstVoxelRef: React.MutableRefObject<[number, number, number] | null>;
  ropeSag: number;
  ropeSagRef: React.MutableRefObject<number>;
  ropeTension: number;
  setRopeTension: React.Dispatch<React.SetStateAction<number>>;
  ropeTensionRef: React.MutableRefObject<number>;
  ropePhase: StrokePhaseHandle<{ nx1: number; ny1: number; nx2: number; ny2: number }>;
  // Cloth
  clothPins: [number, number, number][];
  setClothPins: React.Dispatch<React.SetStateAction<[number, number, number][]>>;
  clothPinsRef: React.MutableRefObject<[number, number, number][]>;
  clothTension: number;
  setClothTension: React.Dispatch<React.SetStateAction<number>>;
  clothTensionRef: React.MutableRefObject<number>;
  clothSimGravityPct: number;
  setClothSimGravityPct: React.Dispatch<React.SetStateAction<number>>;
  clothSimGravityPctRef: React.MutableRefObject<number>;
  clothSimStiffnessPct: number;
  setClothSimStiffnessPct: React.Dispatch<React.SetStateAction<number>>;
  clothSimStiffnessPctRef: React.MutableRefObject<number>;
  clothSimIterations: number;
  setClothSimIterations: React.Dispatch<React.SetStateAction<number>>;
  clothSimIterationsRef: React.MutableRefObject<number>;
  clothSimConstraintPasses: number;
  setClothSimConstraintPasses: React.Dispatch<React.SetStateAction<number>>;
  clothSimConstraintPassesRef: React.MutableRefObject<number>;
  clothPhase: StrokePhaseHandle<Record<string, never>>;
  handleClothPinClick: (nx: number, ny: number) => Promise<void>;
}

export function useRopeClothGenerator(ctx: RopeClothGeneratorContext): RopeClothGeneratorState {
  // -- Shared state -----------------------------------------------------------
  const [clothGravityDirection, setClothGravityDirection] =
    useState<ClothGravityDirectionId>("down");
  const [ropeBrushRadiusIndex, setRopeBrushRadiusIndex] = useState(2);
  const [ropeBrushShapeUi, setRopeBrushShapeUi] = useState<"sphere" | "cube">("sphere");

  const clothGravityDirectionRef = useRef(clothGravityDirection);
  const ropeBrushRadiusIndexRef = useRef(ropeBrushRadiusIndex);
  const ropeBrushShapeUiRef = useRef<"sphere" | "cube">(ropeBrushShapeUi);

  useEffect(() => {
    clothGravityDirectionRef.current = clothGravityDirection;
  }, [clothGravityDirection]);
  useEffect(() => {
    ropeBrushRadiusIndexRef.current = ropeBrushRadiusIndex;
  }, [ropeBrushRadiusIndex]);
  useEffect(() => {
    ropeBrushShapeUiRef.current = ropeBrushShapeUi;
  }, [ropeBrushShapeUi]);

  // -- Rope state -------------------------------------------------------------
  const [ropeFirstScreen, setRopeFirstScreen] = useState<{ nx: number; ny: number } | null>(null);
  const [ropeFirstVoxel, setRopeFirstVoxel] = useState<[number, number, number] | null>(null);
  const [ropeSag, _setRopeSag] = useState(2.5);
  const [ropeTension, setRopeTension] = useState(0.5);

  const ropeFirstScreenRef = useRef<{ nx: number; ny: number } | null>(null);
  const ropeFirstVoxelRef = useRef<[number, number, number] | null>(null);
  const ropeSagRef = useRef(ropeSag);
  const ropeTensionRef = useRef(ropeTension);

  useEffect(() => {
    ropeFirstScreenRef.current = ropeFirstScreen;
  }, [ropeFirstScreen]);
  useEffect(() => {
    ropeFirstVoxelRef.current = ropeFirstVoxel;
  }, [ropeFirstVoxel]);
  useEffect(() => {
    ropeSagRef.current = ropeSag;
  }, [ropeSag]);
  useEffect(() => {
    ropeTensionRef.current = ropeTension;
  }, [ropeTension]);

  // -- Cloth state ------------------------------------------------------------
  const [clothPins, setClothPins] = useState<[number, number, number][]>([]);
  const clothPinsRef = useRef<[number, number, number][]>([]);
  const [clothTension, setClothTension] = useState(0.5);
  const [clothSimGravityPct, setClothSimGravityPct] = useState(100);
  const [clothSimStiffnessPct, setClothSimStiffnessPct] = useState(100);
  const [clothSimIterations, setClothSimIterations] = useState(0);
  const [clothSimConstraintPasses, setClothSimConstraintPasses] = useState(2);

  const clothTensionRef = useRef(clothTension);
  const clothSimGravityPctRef = useRef(clothSimGravityPct);
  const clothSimStiffnessPctRef = useRef(clothSimStiffnessPct);
  const clothSimIterationsRef = useRef(clothSimIterations);
  const clothSimConstraintPassesRef = useRef(clothSimConstraintPasses);

  useEffect(() => {
    clothPinsRef.current = clothPins;
  }, [clothPins]);
  useEffect(() => {
    clothTensionRef.current = clothTension;
  }, [clothTension]);
  useEffect(() => {
    clothSimGravityPctRef.current = clothSimGravityPct;
  }, [clothSimGravityPct]);
  useEffect(() => {
    clothSimStiffnessPctRef.current = clothSimStiffnessPct;
  }, [clothSimStiffnessPct]);
  useEffect(() => {
    clothSimIterationsRef.current = clothSimIterations;
  }, [clothSimIterations]);
  useEffect(() => {
    clothSimConstraintPassesRef.current = clothSimConstraintPasses;
  }, [clothSimConstraintPasses]);

  // -- Rope phase -------------------------------------------------------------
  const ropePhase = useStrokePhase<{
    nx1: number;
    ny1: number;
    nx2: number;
    ny2: number;
  }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      setRopeFirstScreen(null);
      setRopeFirstVoxel(null);
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx1, ny1, nx2, ny2 } = snap.data;
      void invoke("generator_rope_at_screen", {
        args: {
          nx1,
          ny1,
          nx2,
          ny2,
          tension: ropeTensionRef.current,
          gravityDirection: clothGravityDirectionRef.current,
          brushRadius: ropeBrushRadiusIndexRef.current,
          brushShape: sculptBrushShapeToRust(ropeBrushShapeUiRef.current),
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
        },
      }).catch(() => {});
      setRopeFirstScreen(null);
      setRopeFirstVoxel(null);
    },
  });

  // -- Cloth phase ------------------------------------------------------------
  const clothPhase = useStrokePhase<Record<string, never>>({
    phases: ["settings"],
    onCancel: () => {
      setClothPins([]);
      clothPinsRef.current = [];
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: () => {
      const pins = clothPinsRef.current;
      if (pins.length < 3) return;
      void invoke("generator_cloth_from_pins_cmd", {
        args: {
          pins: pins.map((p) => [p[0], p[1], p[2]]),
          tension: clothTensionRef.current,
          gravityDirection: clothGravityDirectionRef.current,
          brushRadius: ropeBrushRadiusIndexRef.current,
          brushShape: sculptBrushShapeToRust(ropeBrushShapeUiRef.current),
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
          gravityScale: clothSimGravityPctRef.current / 100,
          stiffnessScale: clothSimStiffnessPctRef.current / 100,
          clothIterations: clothSimIterationsRef.current,
          clothConstraintPasses: clothSimConstraintPassesRef.current,
        },
      }).catch(() => {});
      setClothPins([]);
      clothPinsRef.current = [];
    },
  });

  // -- Cloth pin click handler ------------------------------------------------
  async function handleClothPinClick(nx: number, ny: number) {
    const c = await invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
      args: {
        nx,
        ny,
        tool: "add",
        strokeSnapToSurface: ctx.selectionStrokeSnapToSurfaceRef.current,
      },
    });
    if (!c) return;
    setClothPins((v) => {
      const idx = v.findIndex((p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2]);
      const next = idx >= 0 ? v.filter((_, i) => i !== idx) : [...v, c];
      clothPinsRef.current = next;
      return next;
    });
  }

  return {
    // Shared
    clothGravityDirection,
    setClothGravityDirection,
    clothGravityDirectionRef,
    ropeBrushRadiusIndex,
    setRopeBrushRadiusIndex,
    ropeBrushRadiusIndexRef,
    ropeBrushShapeUi,
    setRopeBrushShapeUi,
    ropeBrushShapeUiRef,
    // Rope
    ropeFirstScreen,
    setRopeFirstScreen,
    ropeFirstScreenRef,
    ropeFirstVoxel,
    setRopeFirstVoxel,
    ropeFirstVoxelRef,
    ropeSag,
    ropeSagRef,
    ropeTension,
    setRopeTension,
    ropeTensionRef,
    ropePhase,
    // Cloth
    clothPins,
    setClothPins,
    clothPinsRef,
    clothTension,
    setClothTension,
    clothTensionRef,
    clothSimGravityPct,
    setClothSimGravityPct,
    clothSimGravityPctRef,
    clothSimStiffnessPct,
    setClothSimStiffnessPct,
    clothSimStiffnessPctRef,
    clothSimIterations,
    setClothSimIterations,
    clothSimIterationsRef,
    clothSimConstraintPasses,
    setClothSimConstraintPasses,
    clothSimConstraintPassesRef,
    clothPhase,
    handleClothPinClick,
  };
}
