// ── Generator tool-options panel ──���───────────────────────────────────
// Extracted from App.tsx to reduce file size (~1 600 lines of JSX).

import { invoke } from "@tauri-apps/api/core";
import type { GeneratorKindId, ClothGravityDirectionId } from "../types";
import {
  INSECTA_SPECIES_PRESETS,
  PISCINA_SPECIES_PRESETS,
  FAUNA_STANCE_PRESETS,
  FLORA_PRESETS,
} from "../generatorPresets";
import { SCULPT_BRUSH_MAX_INDEX } from "../constants";

// ── Props ────────────────────────────────────────────────────────────

export interface GeneratorToolOptionsProps {
  loading: boolean;
  workBusy: boolean;
  generatorKind: GeneratorKindId;

  // Rocks
  generatorSphereRadius: number;
  setGeneratorSphereRadius: (n: number) => void;
  rockRoughness: number;
  setRockRoughness: (n: number) => void;
  rockCount: number;
  setRockCount: (n: number) => void;
  rockClusterRadius: number;
  setRockClusterRadius: (n: number) => void;
  rockSinkDirection: "none" | "under" | "over";
  setRockSinkDirection: (d: "none" | "under" | "over") => void;
  rockSinkAmount: number;
  setRockSinkAmount: (n: number) => void;

  // Grass
  grassDensity: number;
  setGrassDensity: (n: number) => void;
  grassMaxHeight: number;
  setGrassMaxHeight: (n: number) => void;

  // Rope & Cloth shared
  clothGravityDirection: ClothGravityDirectionId;
  setClothGravityDirection: (d: ClothGravityDirectionId) => void;
  ropeBrushShapeUi: "sphere" | "cube";
  setRopeBrushShapeUi: (s: "sphere" | "cube") => void;
  ropeBrushRadiusIndex: number;
  setRopeBrushRadiusIndex: (n: number) => void;

  // Cloth-specific
  clothSimGravityPct: number;
  setClothSimGravityPct: (n: number) => void;
  clothSimStiffnessPct: number;
  setClothSimStiffnessPct: (n: number) => void;
  clothSimIterations: number;
  setClothSimIterations: (n: number) => void;
  clothSimConstraintPasses: number;
  setClothSimConstraintPasses: (n: number) => void;

  // Ashlar
  ashlarThickness: number;
  setAshlarThickness: (n: number) => void;

  // Flora
  floraPreset: string;
  setFloraPreset: (p: string) => void;
  floraHeight: number;
  setFloraHeight: (n: number) => void;
  floraGirth: number;
  setFloraGirth: (n: number) => void;
  floraWobble: number;
  setFloraWobble: (n: number) => void;
  floraTaper: number;
  setFloraTaper: (n: number) => void;
  floraStemCount: number;
  setFloraStemCount: (n: number) => void;
  floraClusterRadius: number;
  setFloraClusterRadius: (n: number) => void;
  floraBranchCount: number;
  setFloraBranchCount: (n: number) => void;
  floraBranchDepth: number;
  setFloraBranchDepth: (n: number) => void;
  floraBranchStart: number;
  setFloraBranchStart: (n: number) => void;
  floraBranchSpread: number;
  setFloraBranchSpread: (n: number) => void;
  floraBraidStrands: number;
  setFloraBraidStrands: (n: number) => void;
  floraBraidTwist: number;
  setFloraBraidTwist: (n: number) => void;
  floraCanopy: number;
  setFloraCanopy: (n: number) => void;

  // Roof
  roofAreaShape: "polygon" | "square" | "circle";
  setRoofAreaShape: (s: "polygon" | "square" | "circle") => void;
  roofPins: [number, number, number][];
  setRoofPins: (p: [number, number, number][]) => void;
  roofPinsRef: React.MutableRefObject<[number, number, number][]>;
  roofFirstClickRef: React.MutableRefObject<[number, number, number] | null>;
  setRoofFirstClick: (v: [number, number, number] | null) => void;
  roofStyle: string;
  setRoofStyle: (s: string) => void;
  roofHeight: number;
  setRoofHeight: (n: number) => void;
  roofHollow: boolean;
  setRoofHollow: (v: boolean) => void;

  // Piscina
  piscinaSpecies: string;
  setPiscinaSpecies: (s: string) => void;
  piscinaLength: number;
  setPiscinaLength: (n: number) => void;
  piscinaWidth: number;
  setPiscinaWidth: (n: number) => void;
  piscinaThickness: number;
  setPiscinaThickness: (n: number) => void;
  piscinaSpineBend: number;
  setPiscinaSpineBend: (n: number) => void;
  piscinaSpineSCurve: number;
  setPiscinaSpineSCurve: (n: number) => void;
  piscinaAnchorU: number;
  setPiscinaAnchorU: (n: number) => void;
  piscinaAnchorV: number;
  setPiscinaAnchorV: (n: number) => void;
  piscinaShowFinDorsal: boolean;
  setPiscinaShowFinDorsal: (v: boolean) => void;
  piscinaFinDorsal: number;
  setPiscinaFinDorsal: (n: number) => void;
  piscinaShowFinAnal: boolean;
  setPiscinaShowFinAnal: (v: boolean) => void;
  piscinaFinAnal: number;
  setPiscinaFinAnal: (n: number) => void;
  piscinaShowFinCaudal: boolean;
  setPiscinaShowFinCaudal: (v: boolean) => void;
  piscinaFinCaudal: number;
  setPiscinaFinCaudal: (n: number) => void;
  piscinaShowFinPectoral: boolean;
  setPiscinaShowFinPectoral: (v: boolean) => void;
  piscinaFinPectoral: number;
  setPiscinaFinPectoral: (n: number) => void;
  piscinaShowFinPelvic: boolean;
  setPiscinaShowFinPelvic: (v: boolean) => void;
  piscinaFinPelvic: number;
  setPiscinaFinPelvic: (n: number) => void;
  piscinaShowFinAdipose: boolean;
  setPiscinaShowFinAdipose: (v: boolean) => void;
  piscinaFinAdipose: number;
  setPiscinaFinAdipose: (n: number) => void;

  // Insecta
  insectaSpecies: string;
  setInsectaSpecies: (s: string) => void;
  insectaTotalLength: number;
  setInsectaTotalLength: (n: number) => void;
  insectaHeadRatio: number;
  setInsectaHeadRatio: (n: number) => void;
  insectaThoraxRatio: number;
  setInsectaThoraxRatio: (n: number) => void;
  insectaAbdomenRatio: number;
  setInsectaAbdomenRatio: (n: number) => void;
  insectaBodyHalfWidth: number;
  setInsectaBodyHalfWidth: (n: number) => void;
  insectaBodyHalfHeight: number;
  setInsectaBodyHalfHeight: (n: number) => void;
  insectaAbdomenTaper: number;
  setInsectaAbdomenTaper: (n: number) => void;
  insectaHeadShape: number;
  setInsectaHeadShape: (n: number) => void;
  insectaBodyYawDeg: number;
  setInsectaBodyYawDeg: (n: number) => void;
  insectaBodyArch: number;
  setInsectaBodyArch: (n: number) => void;
  insectaAnchorU: number;
  setInsectaAnchorU: (n: number) => void;
  insectaAnchorV: number;
  setInsectaAnchorV: (n: number) => void;
  insectaAntennaLength: number;
  setInsectaAntennaLength: (n: number) => void;
  insectaAntennaSpread: number;
  setInsectaAntennaSpread: (n: number) => void;
  insectaAntennaPitch: number;
  setInsectaAntennaPitch: (n: number) => void;
  insectaAntennaRoot: number;
  setInsectaAntennaRoot: (n: number) => void;
  insectaMandibleLength: number;
  setInsectaMandibleLength: (n: number) => void;
  insectaMandibleSpread: number;
  setInsectaMandibleSpread: (n: number) => void;
  insectaMandibleForward: number;
  setInsectaMandibleForward: (n: number) => void;
  insectaWingShape: number;
  setInsectaWingShape: (n: number) => void;
  insectaShowWingFore: boolean;
  setInsectaShowWingFore: (v: boolean) => void;
  insectaWingForeLength: number;
  setInsectaWingForeLength: (n: number) => void;
  insectaWingForeWidth: number;
  setInsectaWingForeWidth: (n: number) => void;
  insectaWingForeSpread: number;
  setInsectaWingForeSpread: (n: number) => void;
  insectaWingForePitch: number;
  setInsectaWingForePitch: (n: number) => void;
  insectaWingForeOffset: number;
  setInsectaWingForeOffset: (n: number) => void;
  insectaWingForeForwardCant: number;
  setInsectaWingForeForwardCant: (n: number) => void;
  insectaShowWingHind: boolean;
  setInsectaShowWingHind: (v: boolean) => void;
  insectaWingHindLength: number;
  setInsectaWingHindLength: (n: number) => void;
  insectaWingHindWidth: number;
  setInsectaWingHindWidth: (n: number) => void;
  insectaWingHindSpread: number;
  setInsectaWingHindSpread: (n: number) => void;
  insectaWingHindPitch: number;
  setInsectaWingHindPitch: (n: number) => void;
  insectaWingHindOffset: number;
  setInsectaWingHindOffset: (n: number) => void;

  // Fauna
  faunaStance: string;
  setFaunaStance: (s: string) => void;
  faunaArchetype: string;
  setFaunaArchetype: (s: string) => void;
  faunaBodyYawDeg: number;
  setFaunaBodyYawDeg: (n: number) => void;
  faunaBodyArch: number;
  setFaunaBodyArch: (n: number) => void;
  faunaSpineSegments: number;
  setFaunaSpineSegments: (n: number) => void;
  faunaBodyLength: number;
  setFaunaBodyLength: (n: number) => void;
  faunaBodyHalfWidth: number;
  setFaunaBodyHalfWidth: (n: number) => void;
  faunaBodyHalfHeight: number;
  setFaunaBodyHalfHeight: (n: number) => void;
  faunaNeckLength: number;
  setFaunaNeckLength: (n: number) => void;
  faunaNeckHalfWidth: number;
  setFaunaNeckHalfWidth: (n: number) => void;
  faunaNeckHalfHeight: number;
  setFaunaNeckHalfHeight: (n: number) => void;
  faunaHeadLength: number;
  setFaunaHeadLength: (n: number) => void;
  faunaHeadHalfWidth: number;
  setFaunaHeadHalfWidth: (n: number) => void;
  faunaHeadHalfHeight: number;
  setFaunaHeadHalfHeight: (n: number) => void;
  faunaTailLength: number;
  setFaunaTailLength: (n: number) => void;
  faunaShoulderOffsetForward: number;
  setFaunaShoulderOffsetForward: (n: number) => void;
  faunaHipOffsetForward: number;
  setFaunaHipOffsetForward: (n: number) => void;
  faunaFrontUpperLength: number;
  setFaunaFrontUpperLength: (n: number) => void;
  faunaFrontLowerLength: number;
  setFaunaFrontLowerLength: (n: number) => void;
  faunaHindUpperLength: number;
  setFaunaHindUpperLength: (n: number) => void;
  faunaHindLowerLength: number;
  setFaunaHindLowerLength: (n: number) => void;
  faunaAnchorU: number;
  setFaunaAnchorU: (n: number) => void;
  faunaAnchorV: number;
  setFaunaAnchorV: (n: number) => void;
  faunaAutoFootPlacement: boolean;
  setFaunaAutoFootPlacement: (v: boolean) => void;
}

// ── Component ────────────────────────────────────────────────────────

export function GeneratorToolOptions(props: GeneratorToolOptionsProps) {
  const {
    loading,
    workBusy,
    generatorKind,
    generatorSphereRadius,
    setGeneratorSphereRadius,
    rockRoughness,
    setRockRoughness,
    rockCount,
    setRockCount,
    rockClusterRadius,
    setRockClusterRadius,
    rockSinkDirection,
    setRockSinkDirection,
    rockSinkAmount,
    setRockSinkAmount,
    grassDensity,
    setGrassDensity,
    grassMaxHeight,
    setGrassMaxHeight,
    clothGravityDirection,
    setClothGravityDirection,
    ropeBrushShapeUi,
    setRopeBrushShapeUi,
    ropeBrushRadiusIndex,
    setRopeBrushRadiusIndex,
    clothSimGravityPct,
    setClothSimGravityPct,
    clothSimStiffnessPct,
    setClothSimStiffnessPct,
    clothSimIterations,
    setClothSimIterations,
    clothSimConstraintPasses,
    setClothSimConstraintPasses,
    ashlarThickness,
    setAshlarThickness,
    floraPreset,
    setFloraPreset,
    floraHeight,
    setFloraHeight,
    floraGirth,
    setFloraGirth,
    floraWobble,
    setFloraWobble,
    floraTaper,
    setFloraTaper,
    floraStemCount,
    setFloraStemCount,
    floraClusterRadius,
    setFloraClusterRadius,
    floraBranchCount,
    setFloraBranchCount,
    floraBranchDepth,
    setFloraBranchDepth,
    floraBranchStart,
    setFloraBranchStart,
    floraBranchSpread,
    setFloraBranchSpread,
    floraBraidStrands,
    setFloraBraidStrands,
    floraBraidTwist,
    setFloraBraidTwist,
    floraCanopy,
    setFloraCanopy,
    roofAreaShape,
    setRoofAreaShape,
    roofPins,
    setRoofPins,
    roofPinsRef,
    roofFirstClickRef,
    setRoofFirstClick,
    roofStyle,
    setRoofStyle,
    roofHeight,
    setRoofHeight,
    roofHollow,
    setRoofHollow,
    piscinaSpecies,
    setPiscinaSpecies,
    piscinaLength,
    setPiscinaLength,
    piscinaWidth,
    setPiscinaWidth,
    piscinaThickness,
    setPiscinaThickness,
    piscinaSpineBend,
    setPiscinaSpineBend,
    piscinaSpineSCurve,
    setPiscinaSpineSCurve,
    piscinaAnchorU,
    setPiscinaAnchorU,
    piscinaAnchorV,
    setPiscinaAnchorV,
    piscinaShowFinDorsal,
    setPiscinaShowFinDorsal,
    piscinaFinDorsal,
    setPiscinaFinDorsal,
    piscinaShowFinAnal,
    setPiscinaShowFinAnal,
    piscinaFinAnal,
    setPiscinaFinAnal,
    piscinaShowFinCaudal,
    setPiscinaShowFinCaudal,
    piscinaFinCaudal,
    setPiscinaFinCaudal,
    piscinaShowFinPectoral,
    setPiscinaShowFinPectoral,
    piscinaFinPectoral,
    setPiscinaFinPectoral,
    piscinaShowFinPelvic,
    setPiscinaShowFinPelvic,
    piscinaFinPelvic,
    setPiscinaFinPelvic,
    piscinaShowFinAdipose,
    setPiscinaShowFinAdipose,
    piscinaFinAdipose,
    setPiscinaFinAdipose,
    insectaSpecies,
    setInsectaSpecies,
    insectaTotalLength,
    setInsectaTotalLength,
    insectaHeadRatio,
    setInsectaHeadRatio,
    insectaThoraxRatio,
    setInsectaThoraxRatio,
    insectaAbdomenRatio,
    setInsectaAbdomenRatio,
    insectaBodyHalfWidth,
    setInsectaBodyHalfWidth,
    insectaBodyHalfHeight,
    setInsectaBodyHalfHeight,
    insectaAbdomenTaper,
    setInsectaAbdomenTaper,
    insectaHeadShape,
    setInsectaHeadShape,
    insectaBodyYawDeg,
    setInsectaBodyYawDeg,
    insectaBodyArch,
    setInsectaBodyArch,
    insectaAnchorU,
    setInsectaAnchorU,
    insectaAnchorV,
    setInsectaAnchorV,
    insectaAntennaLength,
    setInsectaAntennaLength,
    insectaAntennaSpread,
    setInsectaAntennaSpread,
    insectaAntennaPitch,
    setInsectaAntennaPitch,
    insectaAntennaRoot,
    setInsectaAntennaRoot,
    insectaMandibleLength,
    setInsectaMandibleLength,
    insectaMandibleSpread,
    setInsectaMandibleSpread,
    insectaMandibleForward,
    setInsectaMandibleForward,
    insectaWingShape,
    setInsectaWingShape,
    insectaShowWingFore,
    setInsectaShowWingFore,
    insectaWingForeLength,
    setInsectaWingForeLength,
    insectaWingForeWidth,
    setInsectaWingForeWidth,
    insectaWingForeSpread,
    setInsectaWingForeSpread,
    insectaWingForePitch,
    setInsectaWingForePitch,
    insectaWingForeOffset,
    setInsectaWingForeOffset,
    insectaWingForeForwardCant,
    setInsectaWingForeForwardCant,
    insectaShowWingHind,
    setInsectaShowWingHind,
    insectaWingHindLength,
    setInsectaWingHindLength,
    insectaWingHindWidth,
    setInsectaWingHindWidth,
    insectaWingHindSpread,
    setInsectaWingHindSpread,
    insectaWingHindPitch,
    setInsectaWingHindPitch,
    insectaWingHindOffset,
    setInsectaWingHindOffset,
    faunaStance,
    setFaunaStance,
    faunaArchetype,
    setFaunaArchetype,
    faunaBodyYawDeg,
    setFaunaBodyYawDeg,
    faunaBodyArch,
    setFaunaBodyArch,
    faunaSpineSegments,
    setFaunaSpineSegments,
    faunaBodyLength,
    setFaunaBodyLength,
    faunaBodyHalfWidth,
    setFaunaBodyHalfWidth,
    faunaBodyHalfHeight,
    setFaunaBodyHalfHeight,
    faunaNeckLength,
    setFaunaNeckLength,
    faunaNeckHalfWidth,
    setFaunaNeckHalfWidth,
    faunaNeckHalfHeight,
    setFaunaNeckHalfHeight,
    faunaHeadLength,
    setFaunaHeadLength,
    faunaHeadHalfWidth,
    setFaunaHeadHalfWidth,
    faunaHeadHalfHeight,
    setFaunaHeadHalfHeight,
    faunaTailLength,
    setFaunaTailLength,
    faunaShoulderOffsetForward,
    setFaunaShoulderOffsetForward,
    faunaHipOffsetForward,
    setFaunaHipOffsetForward,
    faunaFrontUpperLength,
    setFaunaFrontUpperLength,
    faunaFrontLowerLength,
    setFaunaFrontLowerLength,
    faunaHindUpperLength,
    setFaunaHindUpperLength,
    faunaHindLowerLength,
    setFaunaHindLowerLength,
    faunaAnchorU,
    setFaunaAnchorU,
    faunaAnchorV,
    setFaunaAnchorV,
    faunaAutoFootPlacement,
    setFaunaAutoFootPlacement,
  } = props;

  const disabled = loading || workBusy;

  return (
    <div className="tool-options-section">
      <div className="tool-options-heading">Generator</div>
      {generatorKind === "rocks" ? (
        <>
          <label className="tool-options-range-label">
            <span>Size</span>
            <input
              type="range"
              min={1}
              max={20}
              value={Math.min(20, generatorSphereRadius)}
              onChange={(ev) => setGeneratorSphereRadius(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
          <label className="tool-options-range-label">
            <span>Roughness</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.02}
              value={rockRoughness}
              onChange={(ev) => setRockRoughness(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
          <label className="tool-options-range-label">
            <span>Count</span>
            <input
              type="range"
              min={1}
              max={5}
              value={rockCount}
              onChange={(ev) => setRockCount(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
          {rockCount > 1 ? (
            <label className="tool-options-range-label">
              <span>Spread</span>
              <input
                type="range"
                min={0}
                max={3}
                value={rockClusterRadius}
                onChange={(ev) => setRockClusterRadius(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          ) : null}
          <div className="tool-options-range-label">
            <span>Sink</span>
            <div className="stroke-mode-buttons" style={{ display: "flex", gap: 2 }}>
              {(["over", "none", "under"] as const).map((dir) => (
                <button
                  key={dir}
                  type="button"
                  className={rockSinkDirection === dir ? "active" : ""}
                  onClick={() => setRockSinkDirection(dir)}
                  disabled={disabled}
                  style={{
                    flex: 1,
                    textTransform: "capitalize",
                    fontSize: 11,
                  }}
                >
                  {dir}
                </button>
              ))}
            </div>
          </div>
          {rockSinkDirection !== "none" ? (
            <label className="tool-options-range-label">
              <span>Layers</span>
              <input
                type="range"
                min={1}
                max={5}
                value={rockSinkAmount}
                onChange={(ev) => setRockSinkAmount(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          ) : null}
        </>
      ) : null}
      {generatorKind === "grass" ? (
        <>
          <label className="tool-options-range-label">
            <span>Radius</span>
            <input
              type="range"
              min={2}
              max={20}
              step={1}
              value={Math.min(20, Math.max(2, generatorSphereRadius))}
              onChange={(ev) => setGeneratorSphereRadius(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
          <label className="tool-options-range-label">
            <span>Density</span>
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={Math.round(grassDensity * 100)}
              onChange={(ev) => setGrassDensity(Number(ev.target.value) / 100)}
              disabled={disabled}
            />
          </label>
          <label className="tool-options-range-label">
            <span>Height</span>
            <input
              type="range"
              min={1}
              max={40}
              value={grassMaxHeight}
              onChange={(ev) => setGrassMaxHeight(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
        </>
      ) : null}
      {generatorKind === "rope" ? (
        <>
          <label className="tool-options-range-label" style={{ marginTop: "0.35rem" }}>
            <span>Gravity</span>
            <select
              aria-label="Rope gravity direction"
              value={clothGravityDirection}
              onChange={(ev) =>
                setClothGravityDirection(ev.target.value as ClothGravityDirectionId)
              }
              disabled={disabled}
            >
              <option value="down">Down (-Y)</option>
              <option value="up">Up (+Y)</option>
              <option value="left">Left (-X)</option>
              <option value="right">Right (+X)</option>
              <option value="forward">Forward (-Z)</option>
              <option value="back">Back (+Z)</option>
            </select>
          </label>
          <div
            className="tool-options-shape-row"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "0.25rem",
              marginTop: "0.35rem",
            }}
            role="group"
            aria-label="Shape"
          >
            <button
              type="button"
              className={
                ropeBrushShapeUi === "sphere"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={disabled}
              onClick={() => setRopeBrushShapeUi("sphere")}
            >
              Sphere
            </button>
            <button
              type="button"
              className={
                ropeBrushShapeUi === "cube"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={disabled}
              onClick={() => setRopeBrushShapeUi("cube")}
            >
              Cube
            </button>
          </div>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Size</span>
            <input
              type="range"
              min={0}
              max={SCULPT_BRUSH_MAX_INDEX}
              value={ropeBrushRadiusIndex}
              onChange={(ev) => setRopeBrushRadiusIndex(Number(ev.target.value))}
              disabled={disabled}
            />
            <span className="tool-options-range-value">{ropeBrushRadiusIndex + 1}</span>
          </label>
        </>
      ) : null}
      {generatorKind === "cloth" ? (
        <>
          <label className="tool-options-range-label" style={{ marginTop: "0.35rem" }}>
            <span>Gravity</span>
            <select
              aria-label="Cloth gravity direction"
              value={clothGravityDirection}
              onChange={(ev) =>
                setClothGravityDirection(ev.target.value as ClothGravityDirectionId)
              }
              disabled={disabled}
            >
              <option value="down">Down (-Y)</option>
              <option value="up">Up (+Y)</option>
              <option value="left">Left (-X)</option>
              <option value="right">Right (+X)</option>
              <option value="forward">Forward (-Z)</option>
              <option value="back">Back (+Z)</option>
            </select>
          </label>
          <div
            className="tool-options-shape-row"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "0.25rem",
              marginTop: "0.35rem",
            }}
            role="group"
            aria-label="Shape"
          >
            <button
              type="button"
              className={
                ropeBrushShapeUi === "sphere"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={disabled}
              onClick={() => setRopeBrushShapeUi("sphere")}
            >
              Sphere
            </button>
            <button
              type="button"
              className={
                ropeBrushShapeUi === "cube"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={disabled}
              onClick={() => setRopeBrushShapeUi("cube")}
            >
              Cube
            </button>
          </div>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Size</span>
            <input
              type="range"
              min={0}
              max={SCULPT_BRUSH_MAX_INDEX}
              value={ropeBrushRadiusIndex}
              onChange={(ev) => setRopeBrushRadiusIndex(Number(ev.target.value))}
              disabled={disabled}
            />
            <span className="tool-options-range-value">{ropeBrushRadiusIndex + 1}</span>
          </label>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Sim gravity</span>
            <input
              type="range"
              min={50}
              max={200}
              step={5}
              value={clothSimGravityPct}
              onChange={(ev) => setClothSimGravityPct(Number(ev.target.value))}
              disabled={disabled}
              title="PBD gravity step scale"
            />
            <span className="tool-options-range-value">{clothSimGravityPct}%</span>
          </label>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Stiffness</span>
            <input
              type="range"
              min={50}
              max={150}
              step={5}
              value={clothSimStiffnessPct}
              onChange={(ev) => setClothSimStiffnessPct(Number(ev.target.value))}
              disabled={disabled}
            />
            <span className="tool-options-range-value">{clothSimStiffnessPct}%</span>
          </label>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Iterations</span>
            <input
              type="range"
              min={0}
              max={64}
              step={1}
              value={clothSimIterations}
              onChange={(ev) => setClothSimIterations(Number(ev.target.value))}
              disabled={disabled}
              title="0 = automatic from tension"
            />
            <span className="tool-options-range-value">
              {clothSimIterations === 0 ? "Auto" : String(clothSimIterations)}
            </span>
          </label>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Passes</span>
            <input
              type="range"
              min={1}
              max={6}
              step={1}
              value={clothSimConstraintPasses}
              onChange={(ev) => setClothSimConstraintPasses(Number(ev.target.value))}
              disabled={disabled}
            />
            <span className="tool-options-range-value">{clothSimConstraintPasses}</span>
          </label>
        </>
      ) : null}
      {generatorKind === "ashlar" ? (
        <>
          <label className="tool-options-range-label">
            <span>Size</span>
            <input
              type="range"
              min={1}
              max={20}
              value={Math.min(20, generatorSphereRadius)}
              onChange={(ev) => setGeneratorSphereRadius(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
          <label className="tool-options-range-label">
            <span>Roughness</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.02}
              value={rockRoughness}
              onChange={(ev) => setRockRoughness(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
          <label className="tool-options-range-label">
            <span>Thickness</span>
            <input
              type="range"
              min={1}
              max={20}
              step={1}
              value={ashlarThickness}
              onChange={(ev) => setAshlarThickness(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
        </>
      ) : null}
      {generatorKind === "flora" ? (
        <div className="gen-wide-grid">
          <div className="gen-card gen-card-full">
            <div className="gen-card-title">Preset</div>
            <select
              value={floraPreset}
              onChange={(ev) => {
                const name = ev.target.value;
                setFloraPreset(name);
                const p = FLORA_PRESETS[name];
                if (p) {
                  setFloraHeight(p.height);
                  setFloraGirth(p.girth);
                  setFloraWobble(p.wobble);
                  setFloraTaper(p.taper);
                  setFloraStemCount(p.stemCount);
                  setFloraClusterRadius(p.clusterRadius);
                  setFloraBranchCount(p.branchCount);
                  setFloraBranchDepth(p.branchDepth);
                  setFloraBranchStart(p.branchStart);
                  setFloraBranchSpread(p.branchSpread);
                  setFloraBraidStrands(p.braidStrands);
                  setFloraBraidTwist(p.braidTwist);
                  setFloraCanopy(p.canopy);
                }
              }}
              disabled={disabled}
            >
              <option value="stalk">Stalk</option>
              <option value="trunk">Trunk</option>
              <option value="contorted">Contorted</option>
              <option value="multi_stem">Multi-stem</option>
              <option value="branched">Branched</option>
              <option value="braided">Braided</option>
              <option value="tuft">Tuft</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Stem</div>
            <label className="tool-options-range-label">
              <span>Height</span>
              <input
                type="range"
                min={1}
                max={96}
                value={floraHeight}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraHeight(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Girth</span>
              <input
                type="range"
                min={0}
                max={20}
                value={floraGirth}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraGirth(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Wobble</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={floraWobble}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraWobble(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Taper</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={floraTaper}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraTaper(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Stems</div>
            <label className="tool-options-range-label">
              <span>Count</span>
              <input
                type="range"
                min={1}
                max={8}
                value={floraStemCount}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraStemCount(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Cluster r</span>
              <input
                type="range"
                min={0}
                max={4}
                value={floraClusterRadius}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraClusterRadius(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Branches</div>
            <label className="tool-options-range-label">
              <span>Count</span>
              <input
                type="range"
                min={0}
                max={6}
                value={floraBranchCount}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraBranchCount(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Depth</span>
              <input
                type="range"
                min={1}
                max={2}
                value={floraBranchDepth}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraBranchDepth(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Start</span>
              <input
                type="range"
                min={0}
                max={0.9}
                step={0.05}
                value={floraBranchStart}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraBranchStart(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Spread</span>
              <input
                type="range"
                min={0}
                max={3}
                step={0.1}
                value={floraBranchSpread}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraBranchSpread(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Braid</div>
            <label className="tool-options-range-label">
              <span>Strands</span>
              <input
                type="range"
                min={1}
                max={5}
                value={floraBraidStrands}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraBraidStrands(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Twist</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={floraBraidTwist}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraBraidTwist(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Canopy</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.02}
                value={floraCanopy}
                onChange={(ev) => {
                  setFloraPreset("custom");
                  setFloraCanopy(Number(ev.target.value));
                }}
                disabled={disabled}
              />
            </label>
          </div>
        </div>
      ) : null}
      {generatorKind === "roof" ? (
        <>
          <div className="tool-options-range-label" style={{ marginTop: "0.35rem" }}>
            <span>Area</span>
            <div
              className="tool-options-shape-row"
              role="group"
              aria-label="Roof area shape"
            >
              {(["polygon", "square", "circle"] as const).map((shape) => (
                <button
                  key={shape}
                  type="button"
                  className={
                    roofAreaShape === shape
                      ? "tool-options-shape-btn is-active"
                      : "tool-options-shape-btn"
                  }
                  onClick={() => {
                    setRoofAreaShape(shape);
                    setRoofPins([]);
                    roofPinsRef.current = [];
                    roofFirstClickRef.current = null;
                    setRoofFirstClick(null);
                    void invoke("voxel_stroke_preview_reset").catch(() => {});
                  }}
                  disabled={disabled}
                  style={{ textTransform: "capitalize" }}
                >
                  {shape}
                </button>
              ))}
            </div>
          </div>
          <label className="tool-options-range-label">
            <span>Style</span>
            <select
              value={roofStyle}
              onChange={(ev) => setRoofStyle(ev.target.value)}
              disabled={disabled}
            >
              <option value="flat">Flat</option>
              <option value="flat_parapet">Flat Parapet</option>
              <option value="pyramid">Pyramid</option>
              <option value="cone">Cone</option>
              <option value="shed">Shed</option>
              <option value="gable">Gable</option>
              <option value="saltbox">Saltbox</option>
              <option value="hip">Hip</option>
              <option value="barrel">Barrel</option>
              <option value="mansard">Mansard</option>
              <option value="gambrel">Gambrel</option>
              <option value="pavilion">Pavilion</option>
              <option value="dutch_gable">Dutch Gable</option>
            </select>
          </label>
          <label className="tool-options-range-label">
            <span>Height</span>
            <input
              type="range"
              min={1}
              max={32}
              value={roofHeight}
              onChange={(ev) => setRoofHeight(Number(ev.target.value))}
              disabled={disabled}
            />
          </label>
          <label className="tool-options-checkbox-row">
            <input
              type="checkbox"
              checked={roofHollow}
              onChange={(ev) => setRoofHollow(ev.target.checked)}
              disabled={disabled}
            />
            <span>Hollow</span>
          </label>
          <p className="sidebar-pane-hint" style={{ marginTop: "0.25rem" }}>
            {roofAreaShape === "polygon"
              ? `Click surface to add pins (${roofPins.length} placed).`
              : roofAreaShape === "square"
                ? "Drag on a face to define the rectangle."
                : "Drag from center to set radius."}
          </p>
        </>
      ) : null}
      {generatorKind === "piscina" ? (
        <div className="gen-wide-grid">
          <div className="gen-card">
            <div className="gen-card-title">Body</div>
            <label className="tool-options-range-label">
              <span>Species</span>
              <select
                value={piscinaSpecies}
                onChange={(ev) => {
                  const sp = ev.target.value;
                  setPiscinaSpecies(sp);
                  const p = PISCINA_SPECIES_PRESETS[sp];
                  if (p) {
                    setPiscinaLength(p.length);
                    setPiscinaWidth(p.width);
                    setPiscinaThickness(p.thickness);
                    setPiscinaFinDorsal(p.finDorsal);
                    setPiscinaFinAnal(p.finAnal);
                    setPiscinaFinCaudal(p.finCaudal);
                    setPiscinaFinPectoral(p.finPectoral);
                    setPiscinaFinPelvic(p.finPelvic);
                    setPiscinaFinAdipose(p.finAdipose);
                  }
                }}
                disabled={disabled}
              >
                <option value="trout">Trout</option>
                <option value="bass">Bass</option>
                <option value="goldfish">Goldfish</option>
                <option value="tuna">Tuna</option>
                <option value="eel">Eel</option>
              </select>
            </label>
            <label className="tool-options-range-label">
              <span>Length</span>
              <input
                type="range"
                min={4}
                max={72}
                value={piscinaLength}
                onChange={(ev) => setPiscinaLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Width</span>
              <input
                type="range"
                min={2}
                max={48}
                value={piscinaWidth}
                onChange={(ev) => setPiscinaWidth(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Thickness</span>
              <input
                type="range"
                min={1}
                max={36}
                value={piscinaThickness}
                onChange={(ev) => setPiscinaThickness(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Pose</div>
            <label className="tool-options-range-label">
              <span>Bend</span>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.05}
                value={piscinaSpineBend}
                onChange={(ev) => setPiscinaSpineBend(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>S-curve</span>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.05}
                value={piscinaSpineSCurve}
                onChange={(ev) => setPiscinaSpineSCurve(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Anchor U</span>
              <input
                type="range"
                min={-24}
                max={24}
                value={piscinaAnchorU}
                onChange={(ev) => setPiscinaAnchorU(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Anchor V</span>
              <input
                type="range"
                min={-24}
                max={24}
                value={piscinaAnchorV}
                onChange={(ev) => setPiscinaAnchorV(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card gen-card-full">
            <div className="gen-card-title">Fins</div>
            <div className="gen-fin-grid">
              {(
                [
                  [
                    "Dorsal",
                    piscinaShowFinDorsal,
                    setPiscinaShowFinDorsal,
                    piscinaFinDorsal,
                    setPiscinaFinDorsal,
                  ],
                  [
                    "Anal",
                    piscinaShowFinAnal,
                    setPiscinaShowFinAnal,
                    piscinaFinAnal,
                    setPiscinaFinAnal,
                  ],
                  [
                    "Caudal",
                    piscinaShowFinCaudal,
                    setPiscinaShowFinCaudal,
                    piscinaFinCaudal,
                    setPiscinaFinCaudal,
                  ],
                  [
                    "Pectoral",
                    piscinaShowFinPectoral,
                    setPiscinaShowFinPectoral,
                    piscinaFinPectoral,
                    setPiscinaFinPectoral,
                  ],
                  [
                    "Pelvic",
                    piscinaShowFinPelvic,
                    setPiscinaShowFinPelvic,
                    piscinaFinPelvic,
                    setPiscinaFinPelvic,
                  ],
                  [
                    "Adipose",
                    piscinaShowFinAdipose,
                    setPiscinaShowFinAdipose,
                    piscinaFinAdipose,
                    setPiscinaFinAdipose,
                  ],
                ] as const
              ).map(([name, show, setShow, scale, setScale]) => (
                <div key={name} className="gen-card">
                  <label className="tool-options-checkbox-row">
                    <input
                      type="checkbox"
                      checked={show}
                      onChange={(ev) => setShow(ev.target.checked)}
                      disabled={disabled}
                    />
                    <span>{name}</span>
                  </label>
                  <label className="tool-options-range-label">
                    <span>Scale</span>
                    <input
                      type="range"
                      min={1}
                      max={8}
                      value={scale}
                      onChange={(ev) => setScale(Number(ev.target.value))}
                      disabled={disabled || !show}
                    />
                  </label>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
      {generatorKind === "insecta" ? (
        <div className="gen-wide-grid">
          <div className="gen-card">
            <div className="gen-card-title">Body</div>
            <label className="tool-options-range-label">
              <span>Species</span>
              <select
                value={insectaSpecies}
                onChange={(ev) => {
                  const sp = ev.target.value;
                  setInsectaSpecies(sp);
                  const p = INSECTA_SPECIES_PRESETS[sp];
                  if (p) {
                    setInsectaTotalLength(p.totalLength);
                    setInsectaHeadRatio(p.headRatio);
                    setInsectaThoraxRatio(p.thoraxRatio);
                    setInsectaAbdomenRatio(p.abdomenRatio);
                    setInsectaBodyHalfWidth(p.bodyHalfWidth);
                    setInsectaBodyHalfHeight(p.bodyHalfHeight);
                    setInsectaAbdomenTaper(p.abdomenTaper);
                    setInsectaHeadShape(p.headShape);
                    setInsectaBodyArch(p.bodyArch);
                    setInsectaAntennaLength(p.antennaLength);
                    setInsectaAntennaSpread(p.antennaSpread);
                    setInsectaAntennaPitch(p.antennaPitch);
                    setInsectaAntennaRoot(p.antennaRoot);
                    setInsectaMandibleLength(p.mandibleLength);
                    setInsectaMandibleSpread(p.mandibleSpread);
                    setInsectaMandibleForward(p.mandibleForward);
                    setInsectaWingShape(p.wingShape);
                    setInsectaShowWingFore(p.showWingFore);
                    setInsectaWingForeLength(p.wingForeLength);
                    setInsectaWingForeWidth(p.wingForeWidth);
                    setInsectaWingForeSpread(p.wingForeSpread);
                    setInsectaWingForePitch(p.wingForePitch);
                    setInsectaWingForeOffset(p.wingForeOffset);
                    setInsectaWingForeForwardCant(p.wingForeForwardCant);
                    setInsectaShowWingHind(p.showWingHind);
                    setInsectaWingHindLength(p.wingHindLength);
                    setInsectaWingHindWidth(p.wingHindWidth);
                    setInsectaWingHindSpread(p.wingHindSpread);
                    setInsectaWingHindPitch(p.wingHindPitch);
                    setInsectaWingHindOffset(p.wingHindOffset);
                  }
                }}
                disabled={disabled}
              >
                <option value="bee">Bee</option>
                <option value="dragonfly">Dragonfly</option>
                <option value="grasshopper">Grasshopper</option>
                <option value="fly">Fly</option>
                <option value="junebug">June Bug</option>
              </select>
            </label>
            <label className="tool-options-range-label">
              <span>Length</span>
              <input
                type="range"
                min={12}
                max={72}
                value={insectaTotalLength}
                onChange={(ev) => setInsectaTotalLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-width</span>
              <input
                type="range"
                min={1}
                max={12}
                value={insectaBodyHalfWidth}
                onChange={(ev) => setInsectaBodyHalfWidth(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-height</span>
              <input
                type="range"
                min={1}
                max={10}
                value={insectaBodyHalfHeight}
                onChange={(ev) => setInsectaBodyHalfHeight(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Yaw degrees</span>
              <input
                type="range"
                min={-45}
                max={45}
                value={insectaBodyYawDeg}
                onChange={(ev) => setInsectaBodyYawDeg(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Arch</span>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.05}
                value={insectaBodyArch}
                onChange={(ev) => setInsectaBodyArch(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Segments</div>
            <label className="tool-options-range-label">
              <span>Head</span>
              <input
                type="range"
                min={0.1}
                max={3}
                step={0.1}
                value={insectaHeadRatio}
                onChange={(ev) => setInsectaHeadRatio(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Thorax</span>
              <input
                type="range"
                min={0.1}
                max={3}
                step={0.1}
                value={insectaThoraxRatio}
                onChange={(ev) => setInsectaThoraxRatio(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Abdomen</span>
              <input
                type="range"
                min={0.1}
                max={4}
                step={0.1}
                value={insectaAbdomenRatio}
                onChange={(ev) => setInsectaAbdomenRatio(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Abd taper</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={insectaAbdomenTaper}
                onChange={(ev) => setInsectaAbdomenTaper(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Head shape</span>
              <input
                type="range"
                min={0}
                max={100}
                value={insectaHeadShape}
                onChange={(ev) => setInsectaHeadShape(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Antennae</div>
            <label className="tool-options-range-label">
              <span>Length</span>
              <input
                type="range"
                min={0}
                max={32}
                value={insectaAntennaLength}
                onChange={(ev) => setInsectaAntennaLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Spread degrees</span>
              <input
                type="range"
                min={0}
                max={45}
                value={insectaAntennaSpread}
                onChange={(ev) => setInsectaAntennaSpread(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Pitch degrees</span>
              <input
                type="range"
                min={0}
                max={80}
                value={insectaAntennaPitch}
                onChange={(ev) => setInsectaAntennaPitch(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Root</span>
              <input
                type="range"
                min={0}
                max={12}
                value={insectaAntennaRoot}
                onChange={(ev) => setInsectaAntennaRoot(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <div className="gen-card-title" style={{ marginTop: "0.25rem" }}>
              Mandibles
            </div>
            <label className="tool-options-range-label">
              <span>Length</span>
              <input
                type="range"
                min={0}
                max={8}
                value={insectaMandibleLength}
                onChange={(ev) => setInsectaMandibleLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Spread</span>
              <input
                type="range"
                min={0}
                max={25}
                value={insectaMandibleSpread}
                onChange={(ev) => setInsectaMandibleSpread(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Forward</span>
              <input
                type="range"
                min={0}
                max={6}
                value={insectaMandibleForward}
                onChange={(ev) => setInsectaMandibleForward(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Placement</div>
            <label className="tool-options-range-label">
              <span>Anchor U</span>
              <input
                type="range"
                min={-24}
                max={24}
                value={insectaAnchorU}
                onChange={(ev) => setInsectaAnchorU(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Anchor V</span>
              <input
                type="range"
                min={-24}
                max={24}
                value={insectaAnchorV}
                onChange={(ev) => setInsectaAnchorV(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card gen-card-full">
            <div className="gen-card-title">Wings</div>
            <div
              style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem" }}
            >
              <div className="gen-card">
                <label className="tool-options-checkbox-row">
                  <input
                    type="checkbox"
                    checked={insectaShowWingFore}
                    onChange={(ev) => setInsectaShowWingFore(ev.target.checked)}
                    disabled={disabled}
                  />
                  <span>Fore wings</span>
                </label>
                <label className="tool-options-range-label">
                  <span>Shape</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={insectaWingShape}
                    onChange={(ev) => setInsectaWingShape(Number(ev.target.value))}
                    disabled={disabled}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Length</span>
                  <input
                    type="range"
                    min={0}
                    max={40}
                    value={insectaWingForeLength}
                    onChange={(ev) => setInsectaWingForeLength(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingFore}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Width</span>
                  <input
                    type="range"
                    min={0}
                    max={12}
                    value={insectaWingForeWidth}
                    onChange={(ev) => setInsectaWingForeWidth(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingFore}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Spread degrees</span>
                  <input
                    type="range"
                    min={0}
                    max={90}
                    value={insectaWingForeSpread}
                    onChange={(ev) => setInsectaWingForeSpread(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingFore}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Pitch degrees</span>
                  <input
                    type="range"
                    min={0}
                    max={45}
                    value={insectaWingForePitch}
                    onChange={(ev) => setInsectaWingForePitch(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingFore}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Offset</span>
                  <input
                    type="range"
                    min={-8}
                    max={8}
                    value={insectaWingForeOffset}
                    onChange={(ev) => setInsectaWingForeOffset(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingFore}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Fwd cant degrees</span>
                  <input
                    type="range"
                    min={0}
                    max={35}
                    value={insectaWingForeForwardCant}
                    onChange={(ev) =>
                      setInsectaWingForeForwardCant(Number(ev.target.value))
                    }
                    disabled={disabled || !insectaShowWingFore}
                  />
                </label>
              </div>
              <div className="gen-card">
                <label className="tool-options-checkbox-row">
                  <input
                    type="checkbox"
                    checked={insectaShowWingHind}
                    onChange={(ev) => setInsectaShowWingHind(ev.target.checked)}
                    disabled={disabled}
                  />
                  <span>Hind wings</span>
                </label>
                <label className="tool-options-range-label">
                  <span>Length</span>
                  <input
                    type="range"
                    min={0}
                    max={40}
                    value={insectaWingHindLength}
                    onChange={(ev) => setInsectaWingHindLength(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingHind}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Width</span>
                  <input
                    type="range"
                    min={0}
                    max={12}
                    value={insectaWingHindWidth}
                    onChange={(ev) => setInsectaWingHindWidth(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingHind}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Spread degrees</span>
                  <input
                    type="range"
                    min={0}
                    max={90}
                    value={insectaWingHindSpread}
                    onChange={(ev) => setInsectaWingHindSpread(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingHind}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Pitch degrees</span>
                  <input
                    type="range"
                    min={0}
                    max={45}
                    value={insectaWingHindPitch}
                    onChange={(ev) => setInsectaWingHindPitch(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingHind}
                  />
                </label>
                <label className="tool-options-range-label">
                  <span>Offset</span>
                  <input
                    type="range"
                    min={-8}
                    max={8}
                    value={insectaWingHindOffset}
                    onChange={(ev) => setInsectaWingHindOffset(Number(ev.target.value))}
                    disabled={disabled || !insectaShowWingHind}
                  />
                </label>
              </div>
            </div>
          </div>
        </div>
      ) : null}
      {generatorKind === "fauna" ? (
        <div className="gen-wide-grid">
          <div className="gen-card">
            <div className="gen-card-title">Type</div>
            <label className="tool-options-range-label">
              <span>Stance</span>
              <select
                value={faunaStance}
                onChange={(ev) => {
                  const st = ev.target.value;
                  setFaunaStance(st);
                  const p = FAUNA_STANCE_PRESETS[st];
                  if (p) {
                    setFaunaArchetype(p.archetype);
                    setFaunaBodyArch(p.bodyArch);
                    setFaunaSpineSegments(p.spineSegments);
                    setFaunaBodyLength(p.bodyLength);
                    setFaunaBodyHalfWidth(p.bodyHalfWidth);
                    setFaunaBodyHalfHeight(p.bodyHalfHeight);
                    setFaunaNeckLength(p.neckLength);
                    setFaunaNeckHalfWidth(p.neckHalfWidth);
                    setFaunaNeckHalfHeight(p.neckHalfHeight);
                    setFaunaHeadLength(p.headLength);
                    setFaunaHeadHalfWidth(p.headHalfWidth);
                    setFaunaHeadHalfHeight(p.headHalfHeight);
                    setFaunaTailLength(p.tailLength);
                    setFaunaShoulderOffsetForward(p.shoulderOffsetForward);
                    setFaunaHipOffsetForward(p.hipOffsetForward);
                    setFaunaFrontUpperLength(p.frontUpperLength);
                    setFaunaFrontLowerLength(p.frontLowerLength);
                    setFaunaHindUpperLength(p.hindUpperLength);
                    setFaunaHindLowerLength(p.hindLowerLength);
                  }
                }}
                disabled={disabled}
              >
                <option value="quadruped">Quadruped</option>
                <option value="biped">Biped</option>
              </select>
            </label>
            <label className="tool-options-range-label">
              <span>Archetype</span>
              <select
                value={faunaArchetype}
                onChange={(ev) => setFaunaArchetype(ev.target.value)}
                disabled={disabled}
              >
                <option value="plantigrade">Plantigrade</option>
                <option value="digitigrade">Digitigrade</option>
                <option value="ungulate">Ungulate</option>
              </select>
            </label>
            <label className="tool-options-range-label">
              <span>Yaw degrees</span>
              <input
                type="range"
                min={-180}
                max={180}
                value={faunaBodyYawDeg}
                onChange={(ev) => setFaunaBodyYawDeg(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Arch</span>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.02}
                value={faunaBodyArch}
                onChange={(ev) => setFaunaBodyArch(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-checkbox-row">
              <input
                type="checkbox"
                checked={faunaAutoFootPlacement}
                onChange={(ev) => setFaunaAutoFootPlacement(ev.target.checked)}
                disabled={disabled}
              />
              <span>Auto feet</span>
            </label>
            <label className="tool-options-range-label">
              <span>Anchor U</span>
              <input
                type="range"
                min={-24}
                max={24}
                value={faunaAnchorU}
                onChange={(ev) => setFaunaAnchorU(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Anchor V</span>
              <input
                type="range"
                min={-24}
                max={24}
                value={faunaAnchorV}
                onChange={(ev) => setFaunaAnchorV(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Trunk</div>
            <label className="tool-options-range-label">
              <span>Length</span>
              <input
                type="range"
                min={4}
                max={60}
                value={faunaBodyLength}
                onChange={(ev) => setFaunaBodyLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-width</span>
              <input
                type="range"
                min={1}
                max={12}
                value={faunaBodyHalfWidth}
                onChange={(ev) => setFaunaBodyHalfWidth(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-height</span>
              <input
                type="range"
                min={1}
                max={12}
                value={faunaBodyHalfHeight}
                onChange={(ev) => setFaunaBodyHalfHeight(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Spine segs</span>
              <input
                type="range"
                min={2}
                max={20}
                value={faunaSpineSegments}
                onChange={(ev) => setFaunaSpineSegments(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Tail</span>
              <input
                type="range"
                min={0}
                max={20}
                value={faunaTailLength}
                onChange={(ev) => setFaunaTailLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Neck</div>
            <label className="tool-options-range-label">
              <span>Length</span>
              <input
                type="range"
                min={0}
                max={24}
                value={faunaNeckLength}
                onChange={(ev) => setFaunaNeckLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-width</span>
              <input
                type="range"
                min={1}
                max={8}
                value={faunaNeckHalfWidth}
                onChange={(ev) => setFaunaNeckHalfWidth(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-height</span>
              <input
                type="range"
                min={1}
                max={8}
                value={faunaNeckHalfHeight}
                onChange={(ev) => setFaunaNeckHalfHeight(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <div className="gen-card-title" style={{ marginTop: "0.25rem" }}>
              Head
            </div>
            <label className="tool-options-range-label">
              <span>Length</span>
              <input
                type="range"
                min={2}
                max={20}
                value={faunaHeadLength}
                onChange={(ev) => setFaunaHeadLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-width</span>
              <input
                type="range"
                min={1}
                max={8}
                value={faunaHeadHalfWidth}
                onChange={(ev) => setFaunaHeadHalfWidth(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Half-height</span>
              <input
                type="range"
                min={1}
                max={8}
                value={faunaHeadHalfHeight}
                onChange={(ev) => setFaunaHeadHalfHeight(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
          <div className="gen-card">
            <div className="gen-card-title">Limbs</div>
            <label className="tool-options-range-label">
              <span>Shoulder fwd</span>
              <input
                type="range"
                min={-8}
                max={8}
                value={faunaShoulderOffsetForward}
                onChange={(ev) =>
                  setFaunaShoulderOffsetForward(Number(ev.target.value))
                }
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Hip fwd</span>
              <input
                type="range"
                min={-8}
                max={8}
                value={faunaHipOffsetForward}
                onChange={(ev) => setFaunaHipOffsetForward(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Front upper</span>
              <input
                type="range"
                min={2}
                max={20}
                value={faunaFrontUpperLength}
                onChange={(ev) => setFaunaFrontUpperLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Front lower</span>
              <input
                type="range"
                min={2}
                max={20}
                value={faunaFrontLowerLength}
                onChange={(ev) => setFaunaFrontLowerLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Hind upper</span>
              <input
                type="range"
                min={2}
                max={20}
                value={faunaHindUpperLength}
                onChange={(ev) => setFaunaHindUpperLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
            <label className="tool-options-range-label">
              <span>Hind lower</span>
              <input
                type="range"
                min={2}
                max={20}
                value={faunaHindLowerLength}
                onChange={(ev) => setFaunaHindLowerLength(Number(ev.target.value))}
                disabled={disabled}
              />
            </label>
          </div>
        </div>
      ) : null}
    </div>
  );
}
