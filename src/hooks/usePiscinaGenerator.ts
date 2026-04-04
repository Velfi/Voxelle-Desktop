import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface PiscinaGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
}

export interface PiscinaGeneratorState {
  piscinaSpecies: string;
  setPiscinaSpecies: React.Dispatch<React.SetStateAction<string>>;
  piscinaLength: number;
  setPiscinaLength: React.Dispatch<React.SetStateAction<number>>;
  piscinaWidth: number;
  setPiscinaWidth: React.Dispatch<React.SetStateAction<number>>;
  piscinaThickness: number;
  setPiscinaThickness: React.Dispatch<React.SetStateAction<number>>;
  piscinaSpineBend: number;
  setPiscinaSpineBend: React.Dispatch<React.SetStateAction<number>>;
  piscinaSpineSCurve: number;
  setPiscinaSpineSCurve: React.Dispatch<React.SetStateAction<number>>;
  piscinaShowFinDorsal: boolean;
  setPiscinaShowFinDorsal: React.Dispatch<React.SetStateAction<boolean>>;
  piscinaFinDorsal: number;
  setPiscinaFinDorsal: React.Dispatch<React.SetStateAction<number>>;
  piscinaShowFinAnal: boolean;
  setPiscinaShowFinAnal: React.Dispatch<React.SetStateAction<boolean>>;
  piscinaFinAnal: number;
  setPiscinaFinAnal: React.Dispatch<React.SetStateAction<number>>;
  piscinaShowFinCaudal: boolean;
  setPiscinaShowFinCaudal: React.Dispatch<React.SetStateAction<boolean>>;
  piscinaFinCaudal: number;
  setPiscinaFinCaudal: React.Dispatch<React.SetStateAction<number>>;
  piscinaShowFinPectoral: boolean;
  setPiscinaShowFinPectoral: React.Dispatch<React.SetStateAction<boolean>>;
  piscinaFinPectoral: number;
  setPiscinaFinPectoral: React.Dispatch<React.SetStateAction<number>>;
  piscinaShowFinPelvic: boolean;
  setPiscinaShowFinPelvic: React.Dispatch<React.SetStateAction<boolean>>;
  piscinaFinPelvic: number;
  setPiscinaFinPelvic: React.Dispatch<React.SetStateAction<number>>;
  piscinaShowFinAdipose: boolean;
  setPiscinaShowFinAdipose: React.Dispatch<React.SetStateAction<boolean>>;
  piscinaFinAdipose: number;
  setPiscinaFinAdipose: React.Dispatch<React.SetStateAction<number>>;
  piscinaAnchorU: number;
  setPiscinaAnchorU: React.Dispatch<React.SetStateAction<number>>;
  piscinaAnchorV: number;
  setPiscinaAnchorV: React.Dispatch<React.SetStateAction<number>>;
  piscinaPreviewSeedRef: React.MutableRefObject<number>;
  piscinaPhase: StrokePhaseHandle<{ nx: number; ny: number; seed: number }>;
}

export function usePiscinaGenerator(ctx: PiscinaGeneratorContext): PiscinaGeneratorState {
  const [piscinaSpecies, setPiscinaSpecies] = useState<string>("trout");
  const [piscinaLength, setPiscinaLength] = useState(16);
  const [piscinaWidth, setPiscinaWidth] = useState(4);
  const [piscinaThickness, setPiscinaThickness] = useState(3);
  const [piscinaSpineBend, setPiscinaSpineBend] = useState(0);
  const [piscinaSpineSCurve, setPiscinaSpineSCurve] = useState(0);
  const [piscinaShowFinDorsal, setPiscinaShowFinDorsal] = useState(true);
  const [piscinaFinDorsal, setPiscinaFinDorsal] = useState(3);
  const [piscinaShowFinAnal, setPiscinaShowFinAnal] = useState(true);
  const [piscinaFinAnal, setPiscinaFinAnal] = useState(3);
  const [piscinaShowFinCaudal, setPiscinaShowFinCaudal] = useState(true);
  const [piscinaFinCaudal, setPiscinaFinCaudal] = useState(3);
  const [piscinaShowFinPectoral, setPiscinaShowFinPectoral] = useState(true);
  const [piscinaFinPectoral, setPiscinaFinPectoral] = useState(3);
  const [piscinaShowFinPelvic, setPiscinaShowFinPelvic] = useState(true);
  const [piscinaFinPelvic, setPiscinaFinPelvic] = useState(3);
  const [piscinaShowFinAdipose, setPiscinaShowFinAdipose] = useState(true);
  const [piscinaFinAdipose, setPiscinaFinAdipose] = useState(3);
  const [piscinaAnchorU, setPiscinaAnchorU] = useState(0);
  const [piscinaAnchorV, setPiscinaAnchorV] = useState(0);

  const piscinaPreviewSeedRef = useRef((Math.random() * 1e9) | 0);

  const piscinaPhase = useStrokePhase<{ nx: number; ny: number; seed: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny, seed } = snap.data;
      void invoke("generator_piscina_at_screen", {
        args: {
          nx,
          ny,
          seed,
          species: piscinaSpecies,
          length: piscinaLength,
          widthParam: piscinaWidth,
          thickness: piscinaThickness,
          spineBend: piscinaSpineBend,
          spineSCurve: piscinaSpineSCurve,
          finDorsal: piscinaFinDorsal,
          finAnal: piscinaFinAnal,
          finCaudal: piscinaFinCaudal,
          finPectoral: piscinaFinPectoral,
          finPelvic: piscinaFinPelvic,
          finAdipose: piscinaFinAdipose,
          showFinDorsal: piscinaShowFinDorsal,
          showFinAnal: piscinaShowFinAnal,
          showFinCaudal: piscinaShowFinCaudal,
          showFinPectoral: piscinaShowFinPectoral,
          showFinPelvic: piscinaShowFinPelvic,
          showFinAdipose: piscinaShowFinAdipose,
          anchorOffsetU: piscinaAnchorU,
          anchorOffsetV: piscinaAnchorV,
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
        },
      }).catch(() => {});
      piscinaPreviewSeedRef.current = (Math.random() * 1e9) | 0;
    },
  });

  return {
    piscinaSpecies, setPiscinaSpecies,
    piscinaLength, setPiscinaLength,
    piscinaWidth, setPiscinaWidth,
    piscinaThickness, setPiscinaThickness,
    piscinaSpineBend, setPiscinaSpineBend,
    piscinaSpineSCurve, setPiscinaSpineSCurve,
    piscinaShowFinDorsal, setPiscinaShowFinDorsal,
    piscinaFinDorsal, setPiscinaFinDorsal,
    piscinaShowFinAnal, setPiscinaShowFinAnal,
    piscinaFinAnal, setPiscinaFinAnal,
    piscinaShowFinCaudal, setPiscinaShowFinCaudal,
    piscinaFinCaudal, setPiscinaFinCaudal,
    piscinaShowFinPectoral, setPiscinaShowFinPectoral,
    piscinaFinPectoral, setPiscinaFinPectoral,
    piscinaShowFinPelvic, setPiscinaShowFinPelvic,
    piscinaFinPelvic, setPiscinaFinPelvic,
    piscinaShowFinAdipose, setPiscinaShowFinAdipose,
    piscinaFinAdipose, setPiscinaFinAdipose,
    piscinaAnchorU, setPiscinaAnchorU,
    piscinaAnchorV, setPiscinaAnchorV,
    piscinaPreviewSeedRef,
    piscinaPhase,
  };
}
