import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface FaunaGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
}

export interface FaunaGeneratorState {
  faunaStance: string;
  setFaunaStance: React.Dispatch<React.SetStateAction<string>>;
  faunaArchetype: string;
  setFaunaArchetype: React.Dispatch<React.SetStateAction<string>>;
  faunaBodyYawDeg: number;
  setFaunaBodyYawDeg: React.Dispatch<React.SetStateAction<number>>;
  faunaBodyArch: number;
  setFaunaBodyArch: React.Dispatch<React.SetStateAction<number>>;
  faunaSpineSegments: number;
  setFaunaSpineSegments: React.Dispatch<React.SetStateAction<number>>;
  faunaBodyLength: number;
  setFaunaBodyLength: React.Dispatch<React.SetStateAction<number>>;
  faunaBodyHalfWidth: number;
  setFaunaBodyHalfWidth: React.Dispatch<React.SetStateAction<number>>;
  faunaBodyHalfHeight: number;
  setFaunaBodyHalfHeight: React.Dispatch<React.SetStateAction<number>>;
  faunaNeckLength: number;
  setFaunaNeckLength: React.Dispatch<React.SetStateAction<number>>;
  faunaNeckHalfWidth: number;
  setFaunaNeckHalfWidth: React.Dispatch<React.SetStateAction<number>>;
  faunaNeckHalfHeight: number;
  setFaunaNeckHalfHeight: React.Dispatch<React.SetStateAction<number>>;
  faunaHeadLength: number;
  setFaunaHeadLength: React.Dispatch<React.SetStateAction<number>>;
  faunaHeadHalfWidth: number;
  setFaunaHeadHalfWidth: React.Dispatch<React.SetStateAction<number>>;
  faunaHeadHalfHeight: number;
  setFaunaHeadHalfHeight: React.Dispatch<React.SetStateAction<number>>;
  faunaTailLength: number;
  setFaunaTailLength: React.Dispatch<React.SetStateAction<number>>;
  faunaShoulderOffsetForward: number;
  setFaunaShoulderOffsetForward: React.Dispatch<React.SetStateAction<number>>;
  faunaHipOffsetForward: number;
  setFaunaHipOffsetForward: React.Dispatch<React.SetStateAction<number>>;
  faunaFrontUpperLength: number;
  setFaunaFrontUpperLength: React.Dispatch<React.SetStateAction<number>>;
  faunaFrontLowerLength: number;
  setFaunaFrontLowerLength: React.Dispatch<React.SetStateAction<number>>;
  faunaHindUpperLength: number;
  setFaunaHindUpperLength: React.Dispatch<React.SetStateAction<number>>;
  faunaHindLowerLength: number;
  setFaunaHindLowerLength: React.Dispatch<React.SetStateAction<number>>;
  faunaAnchorU: number;
  setFaunaAnchorU: React.Dispatch<React.SetStateAction<number>>;
  faunaAnchorV: number;
  setFaunaAnchorV: React.Dispatch<React.SetStateAction<number>>;
  faunaAutoFootPlacement: boolean;
  setFaunaAutoFootPlacement: React.Dispatch<React.SetStateAction<boolean>>;
  faunaPhase: StrokePhaseHandle<{ nx: number; ny: number }>;
}

export function useFaunaGenerator(ctx: FaunaGeneratorContext): FaunaGeneratorState {
  const [faunaStance, setFaunaStance] = useState<string>("quadruped");
  const [faunaArchetype, setFaunaArchetype] = useState<string>("ungulate");
  const [faunaBodyYawDeg, setFaunaBodyYawDeg] = useState(0);
  const [faunaBodyArch, setFaunaBodyArch] = useState(0.02);
  const [faunaSpineSegments, setFaunaSpineSegments] = useState(7);
  const [faunaBodyLength, setFaunaBodyLength] = useState(17);
  const [faunaBodyHalfWidth, setFaunaBodyHalfWidth] = useState(2);
  const [faunaBodyHalfHeight, setFaunaBodyHalfHeight] = useState(3);
  const [faunaNeckLength, setFaunaNeckLength] = useState(8);
  const [faunaNeckHalfWidth, setFaunaNeckHalfWidth] = useState(2);
  const [faunaNeckHalfHeight, setFaunaNeckHalfHeight] = useState(3);
  const [faunaHeadLength, setFaunaHeadLength] = useState(6);
  const [faunaHeadHalfWidth, setFaunaHeadHalfWidth] = useState(2);
  const [faunaHeadHalfHeight, setFaunaHeadHalfHeight] = useState(3);
  const [faunaTailLength, setFaunaTailLength] = useState(1);
  const [faunaShoulderOffsetForward, setFaunaShoulderOffsetForward] = useState(3);
  const [faunaHipOffsetForward, setFaunaHipOffsetForward] = useState(-3);
  const [faunaFrontUpperLength, setFaunaFrontUpperLength] = useState(7);
  const [faunaFrontLowerLength, setFaunaFrontLowerLength] = useState(7);
  const [faunaHindUpperLength, setFaunaHindUpperLength] = useState(8);
  const [faunaHindLowerLength, setFaunaHindLowerLength] = useState(8);
  const [faunaAnchorU, setFaunaAnchorU] = useState(0);
  const [faunaAnchorV, setFaunaAnchorV] = useState(0);
  const [faunaAutoFootPlacement, setFaunaAutoFootPlacement] = useState(false);

  const faunaPhase = useStrokePhase<{ nx: number; ny: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny } = snap.data;
      void invoke("generator_fauna_at_screen", {
        args: {
          nx,
          ny,
          stance: faunaStance,
          archetype: faunaArchetype,
          anchorOffsetU: faunaAnchorU,
          anchorOffsetV: faunaAnchorV,
          bodyYaw: faunaBodyYawDeg * (Math.PI / 180),
          bodyArch: faunaBodyArch,
          spineSegments: faunaSpineSegments,
          bodyLength: faunaBodyLength,
          bodyHalfWidth: faunaBodyHalfWidth,
          bodyHalfHeight: faunaBodyHalfHeight,
          neckLength: faunaNeckLength,
          neckHalfWidth: faunaNeckHalfWidth,
          neckHalfHeight: faunaNeckHalfHeight,
          headLength: faunaHeadLength,
          headHalfWidth: faunaHeadHalfWidth,
          headHalfHeight: faunaHeadHalfHeight,
          tailLength: faunaTailLength,
          shoulderOffsetForward: faunaShoulderOffsetForward,
          hipOffsetForward: faunaHipOffsetForward,
          frontUpperLength: faunaFrontUpperLength,
          frontLowerLength: faunaFrontLowerLength,
          hindUpperLength: faunaHindUpperLength,
          hindLowerLength: faunaHindLowerLength,
          limbTargets: [
            [20, -2.1, -19],
            [20, 2.1, -19],
            [-3.5, -2.2, -20],
            [-3.5, 2.2, -20],
          ],
          limbPoles: [
            [20, -2.4, 0.6],
            [20, 2.4, 0.6],
            [1.8, -2.8, 1.2],
            [1.8, 2.8, 1.2],
          ],
          spinePoseChest: [0, 0, 0],
          spinePoseNeck: [0, 0, 0],
          spinePoseHead: [0, 0, 0],
          autoFootPlacement: faunaAutoFootPlacement,
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
        },
      }).catch(() => {});
    },
  });

  return {
    faunaStance, setFaunaStance,
    faunaArchetype, setFaunaArchetype,
    faunaBodyYawDeg, setFaunaBodyYawDeg,
    faunaBodyArch, setFaunaBodyArch,
    faunaSpineSegments, setFaunaSpineSegments,
    faunaBodyLength, setFaunaBodyLength,
    faunaBodyHalfWidth, setFaunaBodyHalfWidth,
    faunaBodyHalfHeight, setFaunaBodyHalfHeight,
    faunaNeckLength, setFaunaNeckLength,
    faunaNeckHalfWidth, setFaunaNeckHalfWidth,
    faunaNeckHalfHeight, setFaunaNeckHalfHeight,
    faunaHeadLength, setFaunaHeadLength,
    faunaHeadHalfWidth, setFaunaHeadHalfWidth,
    faunaHeadHalfHeight, setFaunaHeadHalfHeight,
    faunaTailLength, setFaunaTailLength,
    faunaShoulderOffsetForward, setFaunaShoulderOffsetForward,
    faunaHipOffsetForward, setFaunaHipOffsetForward,
    faunaFrontUpperLength, setFaunaFrontUpperLength,
    faunaFrontLowerLength, setFaunaFrontLowerLength,
    faunaHindUpperLength, setFaunaHindUpperLength,
    faunaHindLowerLength, setFaunaHindLowerLength,
    faunaAnchorU, setFaunaAnchorU,
    faunaAnchorV, setFaunaAnchorV,
    faunaAutoFootPlacement, setFaunaAutoFootPlacement,
    faunaPhase,
  };
}
