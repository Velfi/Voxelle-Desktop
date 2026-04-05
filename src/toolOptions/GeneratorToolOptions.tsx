// ── Generator tool-options panel ──���───────────────────────────────────
// Extracted from App.tsx to reduce file size (~1 600 lines of JSX).

import { invoke } from "@tauri-apps/api/core";
import type { GeneratorKindId, ClothGravityDirectionId, StartShapeId } from "../types";
import { FLORA_PRESETS } from "../generatorPresets";
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

  // Shape
  shapeKind: StartShapeId;
  setShapeKind: (s: StartShapeId) => void;
  shapeSize: number;
  setShapeSize: (n: number) => void;
  shapeOverwrite: boolean;
  setShapeOverwrite: (v: boolean) => void;
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
    shapeKind,
    setShapeKind,
    shapeSize,
    setShapeSize,
    shapeOverwrite,
    setShapeOverwrite,
  } = props;

  const disabled = loading || workBusy;

  return (
    <div className="tool-options-section">
      <div className="tool-options-heading">Generator</div>
      {generatorKind === "rocks" ? (
        <>
          <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
          <div className="tool-options-range-label tool-options-range-with-value">
            <span>Sink</span>
            <div
              className="stroke-mode-buttons"
              style={{ display: "flex", gap: 2, flex: "1 1 auto" }}
            >
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Gravity</span>
            <select
              aria-label="Rope gravity direction"
              value={clothGravityDirection}
              onChange={(ev) =>
                setClothGravityDirection(ev.target.value as ClothGravityDirectionId)
              }
              disabled={disabled}
              style={{ flex: "1 1 auto", minWidth: 0 }}
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
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Gravity</span>
            <select
              aria-label="Cloth gravity direction"
              value={clothGravityDirection}
              onChange={(ev) =>
                setClothGravityDirection(ev.target.value as ClothGravityDirectionId)
              }
              disabled={disabled}
              style={{ flex: "1 1 auto", minWidth: 0 }}
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
            <label className="tool-options-range-label tool-options-range-with-value">
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
          <div className="tool-options-range-label tool-options-range-with-value">
            <span>Area</span>
            <div
              className="tool-options-shape-row"
              role="group"
              aria-label="Roof area shape"
              style={{ flex: "1 1 auto" }}
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
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Style</span>
            <select
              value={roofStyle}
              onChange={(ev) => setRoofStyle(ev.target.value)}
              disabled={disabled}
              style={{ flex: "1 1 auto", minWidth: 0 }}
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
          <label className="tool-options-range-label tool-options-range-with-value">
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
          {roofPins.length >= 2 ? (
            <div className="tool-options-range-label">
              <button
                type="button"
                className="tool-options-shape-btn"
                disabled={disabled}
                onClick={() => {
                  const flipped = [...roofPins].reverse();
                  setRoofPins(flipped);
                  roofPinsRef.current = flipped;
                }}
              >
                Flip
              </button>
            </div>
          ) : null}
          <p className="sidebar-pane-hint" style={{ marginTop: "0.25rem" }}>
            {roofAreaShape === "polygon"
              ? `Click surface to add pins (${roofPins.length} placed).`
              : roofAreaShape === "square"
                ? "Drag on a face to define the rectangle."
                : "Drag from center to set radius."}
          </p>
        </>
      ) : null}
      {generatorKind === "shape" ? (
        <>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Shape</span>
            <select
              value={shapeKind}
              onChange={(ev) => setShapeKind(ev.target.value as StartShapeId)}
              disabled={disabled}
              style={{ flex: "1 1 auto", minWidth: 0 }}
            >
              <option value="cube">Cube</option>
              <option value="orb">Orb</option>
              <option value="cylinder">Cylinder</option>
              <option value="hollowCube">Hollow Cube</option>
              <option value="plane">Plane</option>
              <option value="circle">Circle</option>
            </select>
          </label>
          <label className="tool-options-range-label tool-options-range-with-value">
            <span>Size</span>
            <input
              type="range"
              min={1}
              max={256}
              value={shapeSize}
              onChange={(ev) => setShapeSize(Number(ev.target.value))}
              disabled={disabled}
            />
            <span className="tool-options-range-value">{shapeSize}</span>
          </label>
          <label className="tool-options-checkbox-row">
            <input
              type="checkbox"
              checked={shapeOverwrite}
              onChange={(ev) => setShapeOverwrite(ev.target.checked)}
              disabled={disabled}
            />
            <span>Overwrite existing</span>
          </label>
          <p className="sidebar-pane-hint" style={{ marginTop: "0.25rem" }}>
            Click a surface to place the shape.
          </p>
        </>
      ) : null}
    </div>
  );
}
