import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface InsectaGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
}

export interface InsectaGeneratorState {
  insectaSpecies: string;
  setInsectaSpecies: React.Dispatch<React.SetStateAction<string>>;
  insectaTotalLength: number;
  setInsectaTotalLength: React.Dispatch<React.SetStateAction<number>>;
  insectaHeadRatio: number;
  setInsectaHeadRatio: React.Dispatch<React.SetStateAction<number>>;
  insectaThoraxRatio: number;
  setInsectaThoraxRatio: React.Dispatch<React.SetStateAction<number>>;
  insectaAbdomenRatio: number;
  setInsectaAbdomenRatio: React.Dispatch<React.SetStateAction<number>>;
  insectaBodyHalfWidth: number;
  setInsectaBodyHalfWidth: React.Dispatch<React.SetStateAction<number>>;
  insectaBodyHalfHeight: number;
  setInsectaBodyHalfHeight: React.Dispatch<React.SetStateAction<number>>;
  insectaAbdomenTaper: number;
  setInsectaAbdomenTaper: React.Dispatch<React.SetStateAction<number>>;
  insectaHeadShape: number;
  setInsectaHeadShape: React.Dispatch<React.SetStateAction<number>>;
  insectaBodyYawDeg: number;
  setInsectaBodyYawDeg: React.Dispatch<React.SetStateAction<number>>;
  insectaBodyArch: number;
  setInsectaBodyArch: React.Dispatch<React.SetStateAction<number>>;
  insectaAnchorU: number;
  setInsectaAnchorU: React.Dispatch<React.SetStateAction<number>>;
  insectaAnchorV: number;
  setInsectaAnchorV: React.Dispatch<React.SetStateAction<number>>;
  insectaAntennaLength: number;
  setInsectaAntennaLength: React.Dispatch<React.SetStateAction<number>>;
  insectaAntennaSpread: number;
  setInsectaAntennaSpread: React.Dispatch<React.SetStateAction<number>>;
  insectaAntennaPitch: number;
  setInsectaAntennaPitch: React.Dispatch<React.SetStateAction<number>>;
  insectaAntennaRoot: number;
  setInsectaAntennaRoot: React.Dispatch<React.SetStateAction<number>>;
  insectaMandibleLength: number;
  setInsectaMandibleLength: React.Dispatch<React.SetStateAction<number>>;
  insectaMandibleSpread: number;
  setInsectaMandibleSpread: React.Dispatch<React.SetStateAction<number>>;
  insectaMandibleForward: number;
  setInsectaMandibleForward: React.Dispatch<React.SetStateAction<number>>;
  insectaWingShape: number;
  setInsectaWingShape: React.Dispatch<React.SetStateAction<number>>;
  insectaShowWingFore: boolean;
  setInsectaShowWingFore: React.Dispatch<React.SetStateAction<boolean>>;
  insectaWingForeLength: number;
  setInsectaWingForeLength: React.Dispatch<React.SetStateAction<number>>;
  insectaWingForeWidth: number;
  setInsectaWingForeWidth: React.Dispatch<React.SetStateAction<number>>;
  insectaWingForeSpread: number;
  setInsectaWingForeSpread: React.Dispatch<React.SetStateAction<number>>;
  insectaWingForePitch: number;
  setInsectaWingForePitch: React.Dispatch<React.SetStateAction<number>>;
  insectaWingForeOffset: number;
  setInsectaWingForeOffset: React.Dispatch<React.SetStateAction<number>>;
  insectaWingForeForwardCant: number;
  setInsectaWingForeForwardCant: React.Dispatch<React.SetStateAction<number>>;
  insectaShowWingHind: boolean;
  setInsectaShowWingHind: React.Dispatch<React.SetStateAction<boolean>>;
  insectaWingHindLength: number;
  setInsectaWingHindLength: React.Dispatch<React.SetStateAction<number>>;
  insectaWingHindWidth: number;
  setInsectaWingHindWidth: React.Dispatch<React.SetStateAction<number>>;
  insectaWingHindSpread: number;
  setInsectaWingHindSpread: React.Dispatch<React.SetStateAction<number>>;
  insectaWingHindPitch: number;
  setInsectaWingHindPitch: React.Dispatch<React.SetStateAction<number>>;
  insectaWingHindOffset: number;
  setInsectaWingHindOffset: React.Dispatch<React.SetStateAction<number>>;
  insectaPhase: StrokePhaseHandle<{ nx: number; ny: number }>;
}

export function useInsectaGenerator(ctx: InsectaGeneratorContext): InsectaGeneratorState {
  const [insectaSpecies, setInsectaSpecies] = useState<string>("bee");
  const [insectaTotalLength, setInsectaTotalLength] = useState(24);
  const [insectaHeadRatio, setInsectaHeadRatio] = useState(1.0);
  const [insectaThoraxRatio, setInsectaThoraxRatio] = useState(1.2);
  const [insectaAbdomenRatio, setInsectaAbdomenRatio] = useState(2.0);
  const [insectaBodyHalfWidth, setInsectaBodyHalfWidth] = useState(3);
  const [insectaBodyHalfHeight, setInsectaBodyHalfHeight] = useState(3);
  const [insectaAbdomenTaper, setInsectaAbdomenTaper] = useState(0.6);
  const [insectaHeadShape, setInsectaHeadShape] = useState(60);
  const [insectaBodyYawDeg, setInsectaBodyYawDeg] = useState(0);
  const [insectaBodyArch, setInsectaBodyArch] = useState(0);
  const [insectaAnchorU, setInsectaAnchorU] = useState(0);
  const [insectaAnchorV, setInsectaAnchorV] = useState(0);
  const [insectaAntennaLength, setInsectaAntennaLength] = useState(6);
  const [insectaAntennaSpread, setInsectaAntennaSpread] = useState(20);
  const [insectaAntennaPitch, setInsectaAntennaPitch] = useState(30);
  const [insectaAntennaRoot, setInsectaAntennaRoot] = useState(0);
  const [insectaMandibleLength, setInsectaMandibleLength] = useState(0);
  const [insectaMandibleSpread, setInsectaMandibleSpread] = useState(0);
  const [insectaMandibleForward, setInsectaMandibleForward] = useState(0);
  const [insectaWingShape, setInsectaWingShape] = useState(85);
  const [insectaShowWingFore, setInsectaShowWingFore] = useState(true);
  const [insectaWingForeLength, setInsectaWingForeLength] = useState(12);
  const [insectaWingForeWidth, setInsectaWingForeWidth] = useState(3);
  const [insectaWingForeSpread, setInsectaWingForeSpread] = useState(15);
  const [insectaWingForePitch, setInsectaWingForePitch] = useState(0);
  const [insectaWingForeOffset, setInsectaWingForeOffset] = useState(0);
  const [insectaWingForeForwardCant, setInsectaWingForeForwardCant] = useState(0);
  const [insectaShowWingHind, setInsectaShowWingHind] = useState(false);
  const [insectaWingHindLength, setInsectaWingHindLength] = useState(8);
  const [insectaWingHindWidth, setInsectaWingHindWidth] = useState(2);
  const [insectaWingHindSpread, setInsectaWingHindSpread] = useState(15);
  const [insectaWingHindPitch, setInsectaWingHindPitch] = useState(0);
  const [insectaWingHindOffset, setInsectaWingHindOffset] = useState(0);

  const insectaPhase = useStrokePhase<{ nx: number; ny: number }>({
    phases: ["settings"],
    onCancel: () => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      void invoke("voxel_stroke_preview_reset").catch(() => {});
    },
    onCommit: (snap) => {
      void invoke("unlock_generator_preview_camera").catch(() => {});
      const { nx, ny } = snap.data;
      void invoke("generator_insecta_at_screen", {
        args: {
          nx,
          ny,
          species: insectaSpecies,
          totalLength: insectaTotalLength,
          headRatio: insectaHeadRatio,
          thoraxRatio: insectaThoraxRatio,
          abdomenRatio: insectaAbdomenRatio,
          bodyHalfWidth: insectaBodyHalfWidth,
          bodyHalfHeight: insectaBodyHalfHeight,
          abdomenTaper: insectaAbdomenTaper,
          headShape: insectaHeadShape,
          anchorOffsetU: insectaAnchorU,
          anchorOffsetV: insectaAnchorV,
          bodyYaw: insectaBodyYawDeg * (Math.PI / 180),
          bodyArch: insectaBodyArch,
          antennaLength: insectaAntennaLength,
          antennaSpread: insectaAntennaSpread,
          antennaPitch: insectaAntennaPitch,
          antennaRoot: insectaAntennaRoot,
          mandibleLength: insectaMandibleLength,
          mandibleSpread: insectaMandibleSpread,
          mandibleForward: insectaMandibleForward,
          wingShape: insectaWingShape,
          showWingFore: insectaShowWingFore,
          wingForeLength: insectaWingForeLength,
          wingForeWidth: insectaWingForeWidth,
          wingForeSpread: insectaWingForeSpread,
          wingForePitch: insectaWingForePitch,
          wingForeOffset: insectaWingForeOffset,
          wingForeForwardCant: insectaWingForeForwardCant,
          showWingHind: insectaShowWingHind,
          wingHindLength: insectaWingHindLength,
          wingHindWidth: insectaWingHindWidth,
          wingHindSpread: insectaWingHindSpread,
          wingHindPitch: insectaWingHindPitch,
          wingHindOffset: insectaWingHindOffset,
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
        },
      }).catch(() => {});
    },
  });

  return {
    insectaSpecies, setInsectaSpecies,
    insectaTotalLength, setInsectaTotalLength,
    insectaHeadRatio, setInsectaHeadRatio,
    insectaThoraxRatio, setInsectaThoraxRatio,
    insectaAbdomenRatio, setInsectaAbdomenRatio,
    insectaBodyHalfWidth, setInsectaBodyHalfWidth,
    insectaBodyHalfHeight, setInsectaBodyHalfHeight,
    insectaAbdomenTaper, setInsectaAbdomenTaper,
    insectaHeadShape, setInsectaHeadShape,
    insectaBodyYawDeg, setInsectaBodyYawDeg,
    insectaBodyArch, setInsectaBodyArch,
    insectaAnchorU, setInsectaAnchorU,
    insectaAnchorV, setInsectaAnchorV,
    insectaAntennaLength, setInsectaAntennaLength,
    insectaAntennaSpread, setInsectaAntennaSpread,
    insectaAntennaPitch, setInsectaAntennaPitch,
    insectaAntennaRoot, setInsectaAntennaRoot,
    insectaMandibleLength, setInsectaMandibleLength,
    insectaMandibleSpread, setInsectaMandibleSpread,
    insectaMandibleForward, setInsectaMandibleForward,
    insectaWingShape, setInsectaWingShape,
    insectaShowWingFore, setInsectaShowWingFore,
    insectaWingForeLength, setInsectaWingForeLength,
    insectaWingForeWidth, setInsectaWingForeWidth,
    insectaWingForeSpread, setInsectaWingForeSpread,
    insectaWingForePitch, setInsectaWingForePitch,
    insectaWingForeOffset, setInsectaWingForeOffset,
    insectaWingForeForwardCant, setInsectaWingForeForwardCant,
    insectaShowWingHind, setInsectaShowWingHind,
    insectaWingHindLength, setInsectaWingHindLength,
    insectaWingHindWidth, setInsectaWingHindWidth,
    insectaWingHindSpread, setInsectaWingHindSpread,
    insectaWingHindPitch, setInsectaWingHindPitch,
    insectaWingHindOffset, setInsectaWingHindOffset,
    insectaPhase,
  };
}
