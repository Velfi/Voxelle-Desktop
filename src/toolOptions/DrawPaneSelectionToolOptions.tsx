import { AreaShapeControls } from "./AreaShapeControls";
import type {
  DrawStrokeModeApi,
  PlaneAxisApi,
  SelectionMethod,
  StrokeDrawStyle,
  StrokeFamilyVariant,
} from "../drawToolModel";

type BrushShape = "sphere" | "cube" | "pyramid" | "square" | "circle";

export type DrawPaneSelectionToolOptionsProps = {
  loading: boolean;
  workBusy: boolean;
  selectionMethod: SelectionMethod;
  drawStrokeMode: DrawStrokeModeApi;
  setDrawStrokeMode: (m: DrawStrokeModeApi) => void;
  strokeDrawStyle: StrokeDrawStyle;
  setStrokeDrawStyle: (s: StrokeDrawStyle) => void;
  strokeFamilyVariant: StrokeFamilyVariant;
  setStrokeFamilyVariant: (v: StrokeFamilyVariant) => void;
  planeAxis: PlaneAxisApi;
  setPlaneAxis: (a: PlaneAxisApi) => void;
  fillSelectDiagonals: boolean;
  setFillSelectDiagonals: (v: boolean) => void;
  fillRespectsColor: boolean;
  setFillRespectsColor: (v: boolean) => void;
  sprayDensity: number;
  setSprayDensity: (n: number) => void;
  brushShape: BrushShape;
  setBrushShape: (s: BrushShape) => void;
  /** Clip brush to the outward side of the face under the cursor (axis from ray hit). */
  brushClipBottomHalf: boolean;
  setBrushClipBottomHalf: (v: boolean) => void;
  brushRadius: number;
  setBrushRadius: (n: number) => void;
  selectionStrokeSnapToSurface: boolean;
  setSelectionStrokeSnapToSurface: (v: boolean) => void;
  selectionStrokeAxisAlign: boolean;
  setSelectionStrokeAxisAlign: (v: boolean) => void;
  surfacePlaneHollow: boolean;
  setSurfacePlaneHollow: (v: boolean) => void;
  sprayConstrainToPlane: boolean;
  setSprayConstrainToPlane: (v: boolean) => void;
  spraySizeRange: boolean;
  setSpraySizeRange: (v: boolean) => void;
  /** Scatter: random stamp offset in voxels (web `sprayScatter`). */
  sprayScatter: number;
  setSprayScatter: (n: number) => void;
  sprayRadiusMin: number;
  setSprayRadiusMin: (n: number) => void;
  sprayRadiusMax: number;
  setSprayRadiusMax: (n: number) => void;
  /** Separate brush shape for spray mode. */
  sprayBrushShape: BrushShape;
  setSprayBrushShape: (s: BrushShape) => void;
  /** Plane reference for constrain-to-plane. */
  sprayConstrainToPlaneRef: "auto" | "camera" | "x" | "y" | "z";
  setSprayConstrainToPlaneRef: (v: "auto" | "camera" | "x" | "y" | "z") => void;
  fillConstrainToPlane: boolean;
  setFillConstrainToPlane: (v: boolean) => void;
};

/**
 * Bottom tool panel for draw tools × selection method (add/remove/paint/select).
 * Stroke: Line/Precise, brush, snap/axis-align.
 * Surface: Plane/Circle/Polygon, brush, snap; plane mode adds axis + Hollow.
 * Solid: Cube/Cylinder/Polygon, brush, snap, plane axis (X/Y/Z/Auto), Hollow.
 * Spray: constrain + brush + size range + size + scatter + snap.
 * Fill: constrain + diagonals + respect color.
 * Other methods: Brush + stroke line/brush + full area shape dropdown.
 */
export function DrawPaneSelectionToolOptions(p: DrawPaneSelectionToolOptionsProps) {
  const narrowStroke = p.selectionMethod === "stroke";
  const narrowSurface = p.selectionMethod === "surface";
  const narrowSolid = p.selectionMethod === "solid";
  const narrowSpray = p.selectionMethod === "spray";
  const narrowFill = p.selectionMethod === "fill";

  const areaCommon = (
    <AreaShapeControls
      loading={p.loading}
      workBusy={p.workBusy}
      drawStrokeMode={p.drawStrokeMode}
      onDrawStrokeModeChange={p.setDrawStrokeMode}
      planeAxis={p.planeAxis}
      onPlaneAxisChange={p.setPlaneAxis}
      fillSelectDiagonals={p.fillSelectDiagonals}
      onFillSelectDiagonalsChange={p.setFillSelectDiagonals}
      fillRespectsColor={p.fillRespectsColor}
      onFillRespectsColorChange={p.setFillRespectsColor}
      sprayDensity={p.sprayDensity}
      onSprayDensityChange={p.setSprayDensity}
      spraySliderMode="selection"
      selectLabel="Area shape"
      fillOptionLabel="Fill (region)"
    />
  );

  if (narrowStroke) {
    return (
      <>
        <div className="tool-options-section">
          <div className="tool-options-heading">Area shape</div>
          <div
            className="tool-options-shape-row tool-options-shape-row-two"
            role="group"
            aria-label="Area shape"
          >
            <button
              type="button"
              className={
                p.drawStrokeMode === "line"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("line");
                p.setStrokeDrawStyle("line");
              }}
            >
              Line
            </button>
            <button
              type="button"
              className={
                p.drawStrokeMode === "precise"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("precise");
                p.setStrokeDrawStyle("line");
              }}
            >
              Precise
            </button>
          </div>
        </div>
        <div className="tool-options-section">
          <div className="tool-options-heading">Brush shape</div>
          <div className="tool-options-shape-row" role="group" aria-label="Brush shape">
            {(
              [
                ["cube", "Cube"],
                ["sphere", "Sphere"],
                ["pyramid", "Pyramid"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={
                  p.brushShape === id
                    ? "tool-options-shape-btn is-active"
                    : "tool-options-shape-btn"
                }
                disabled={p.loading || p.workBusy}
                onClick={() => p.setBrushShape(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.35rem" }}>
            <input
              type="checkbox"
              checked={p.brushClipBottomHalf}
              onChange={(ev) => p.setBrushClipBottomHalf(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span title="Uses the clicked face outward normal (world +Y if no solid hit)">
              Outer half (face)
            </span>
          </label>
          <div
            className="tool-options-heading tool-options-heading-mixed"
            style={{ marginTop: "0.5rem" }}
          >
            Brush diameter
          </div>
          <label className="tool-options-range-label tool-options-range-with-value">
            <input
              type="range"
              min={1}
              max={65}
              step={2}
              value={p.brushRadius * 2 + 1}
              onChange={(ev) => p.setBrushRadius((Number(ev.target.value) - 1) / 2)}
              disabled={p.loading || p.workBusy}
            />
            <span className="tool-options-range-value" aria-live="polite">
              {p.brushRadius * 2 + 1}
            </span>
          </label>
        </div>
        <div className="tool-options-section">
          <label className="tool-options-checkbox-row">
            <input
              type="checkbox"
              checked={p.selectionStrokeSnapToSurface}
              onChange={(ev) => p.setSelectionStrokeSnapToSurface(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Snap to surface</span>
          </label>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.35rem" }}>
            <input
              type="checkbox"
              checked={p.selectionStrokeAxisAlign}
              onChange={(ev) => p.setSelectionStrokeAxisAlign(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Axis-align</span>
          </label>
        </div>
      </>
    );
  }

  if (narrowSurface) {
    return (
      <>
        <div className="tool-options-section">
          <div className="tool-options-heading">Area shape</div>
          <div
            className="tool-options-shape-row"
            role="group"
            aria-label="Area shape"
            style={{ flexWrap: "wrap" }}
          >
            <button
              type="button"
              className={
                p.drawStrokeMode === "plane"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("plane");
                p.setStrokeDrawStyle("brush");
              }}
            >
              Plane
            </button>
            <button
              type="button"
              className={
                p.drawStrokeMode === "circle"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("circle");
                p.setStrokeDrawStyle("brush");
              }}
            >
              Circle
            </button>
            <button
              type="button"
              className={
                p.drawStrokeMode === "polygonHull"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("polygonHull");
                p.setStrokeDrawStyle("brush");
              }}
            >
              Polygon
            </button>
          </div>
        </div>
        <div className="tool-options-section">
          <div className="tool-options-heading">Brush shape</div>
          <div className="tool-options-shape-row" role="group" aria-label="Brush shape">
            {(
              [
                ["cube", "Cube"],
                ["sphere", "Sphere"],
                ["pyramid", "Pyramid"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={
                  p.brushShape === id
                    ? "tool-options-shape-btn is-active"
                    : "tool-options-shape-btn"
                }
                disabled={p.loading || p.workBusy}
                onClick={() => p.setBrushShape(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.35rem" }}>
            <input
              type="checkbox"
              checked={p.brushClipBottomHalf}
              onChange={(ev) => p.setBrushClipBottomHalf(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span title="Uses the clicked face outward normal (world +Y if no solid hit)">
              Outer half (face)
            </span>
          </label>
          <div
            className="tool-options-heading tool-options-heading-mixed"
            style={{ marginTop: "0.5rem" }}
          >
            Brush diameter
          </div>
          <label className="tool-options-range-label tool-options-range-with-value">
            <input
              type="range"
              min={1}
              max={65}
              step={2}
              value={p.brushRadius * 2 + 1}
              onChange={(ev) => p.setBrushRadius((Number(ev.target.value) - 1) / 2)}
              disabled={p.loading || p.workBusy}
            />
            <span className="tool-options-range-value" aria-live="polite">
              {p.brushRadius * 2 + 1}
            </span>
          </label>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.5rem" }}>
            <input
              type="checkbox"
              checked={p.selectionStrokeSnapToSurface}
              onChange={(ev) => p.setSelectionStrokeSnapToSurface(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Snap to surface</span>
          </label>
        </div>
        {p.drawStrokeMode === "plane" ? (
          <div className="tool-options-section">
            <div className="tool-options-heading">Plane options</div>
            <div
              className="tool-options-shape-row"
              role="group"
              aria-label="Plane axis"
              style={{ flexWrap: "wrap" }}
            >
              {(
                [
                  ["x", "X"],
                  ["y", "Y"],
                  ["z", "Z"],
                  ["auto", "Auto"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={
                    p.planeAxis === id
                      ? "tool-options-shape-btn is-active"
                      : "tool-options-shape-btn"
                  }
                  disabled={p.loading || p.workBusy}
                  onClick={() => p.setPlaneAxis(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            <label className="tool-options-checkbox-row" style={{ marginTop: "0.5rem" }}>
              <input
                type="checkbox"
                checked={p.surfacePlaneHollow}
                onChange={(ev) => p.setSurfacePlaneHollow(ev.target.checked)}
                disabled={p.loading || p.workBusy}
              />
              <span>Hollow</span>
            </label>
          </div>
        ) : null}
        {p.drawStrokeMode === "circle" ? (
          <div className="tool-options-section">
            <p
              className="tool-options-hint"
              style={{ margin: 0, fontSize: "0.85rem", opacity: 0.9 }}
            >
              Circle: first click center, second click edge.
            </p>
          </div>
        ) : null}
      </>
    );
  }

  if (narrowSolid) {
    return (
      <>
        <div className="tool-options-section">
          <div className="tool-options-heading">Area shape</div>
          <div
            className="tool-options-shape-row"
            role="group"
            aria-label="Area shape"
            style={{ flexWrap: "wrap" }}
          >
            <button
              type="button"
              className={
                p.drawStrokeMode === "cuboid"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("cuboid");
                p.setStrokeDrawStyle("line");
                p.setStrokeFamilyVariant("solid");
              }}
            >
              Cube
            </button>
            <button
              type="button"
              className={
                p.drawStrokeMode === "cylinder"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("cylinder");
                p.setStrokeDrawStyle("line");
                p.setStrokeFamilyVariant("solid");
              }}
            >
              Cylinder
            </button>
            <button
              type="button"
              className={
                p.drawStrokeMode === "polygon" || p.drawStrokeMode === "polygonHull"
                  ? "tool-options-shape-btn is-active"
                  : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => {
                p.setDrawStrokeMode("polygon");
                p.setStrokeDrawStyle("line");
                p.setStrokeFamilyVariant("solid");
              }}
            >
              Polygon
            </button>
          </div>
        </div>
        <div className="tool-options-section">
          <div className="tool-options-heading">Brush shape</div>
          <div className="tool-options-shape-row" role="group" aria-label="Brush shape">
            {(
              [
                ["cube", "Cube"],
                ["sphere", "Sphere"],
                ["pyramid", "Pyramid"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={
                  p.brushShape === id
                    ? "tool-options-shape-btn is-active"
                    : "tool-options-shape-btn"
                }
                disabled={p.loading || p.workBusy}
                onClick={() => p.setBrushShape(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.35rem" }}>
            <input
              type="checkbox"
              checked={p.brushClipBottomHalf}
              onChange={(ev) => p.setBrushClipBottomHalf(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span title="Uses the clicked face outward normal (world +Y if no solid hit)">
              Outer half (face)
            </span>
          </label>
          <div
            className="tool-options-heading tool-options-heading-mixed"
            style={{ marginTop: "0.5rem" }}
          >
            Brush diameter
          </div>
          <label className="tool-options-range-label tool-options-range-with-value">
            <input
              type="range"
              min={1}
              max={65}
              step={2}
              value={p.brushRadius * 2 + 1}
              onChange={(ev) => p.setBrushRadius((Number(ev.target.value) - 1) / 2)}
              disabled={p.loading || p.workBusy}
            />
            <span className="tool-options-range-value" aria-live="polite">
              {p.brushRadius * 2 + 1}
            </span>
          </label>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.5rem" }}>
            <input
              type="checkbox"
              checked={p.selectionStrokeSnapToSurface}
              onChange={(ev) => p.setSelectionStrokeSnapToSurface(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Snap to surface</span>
          </label>
        </div>
        <div className="tool-options-section">
          <div className="tool-options-heading">Plane options</div>
          <div
            className="tool-options-shape-row"
            role="group"
            aria-label="Plane axis"
            style={{ flexWrap: "wrap" }}
          >
            {(
              [
                ["x", "X"],
                ["y", "Y"],
                ["z", "Z"],
                ["auto", "Auto"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={
                  p.planeAxis === id ? "tool-options-shape-btn is-active" : "tool-options-shape-btn"
                }
                disabled={p.loading || p.workBusy}
                onClick={() => p.setPlaneAxis(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="tool-options-section">
          <label className="tool-options-checkbox-row">
            <input
              type="checkbox"
              checked={p.surfacePlaneHollow}
              onChange={(ev) => p.setSurfacePlaneHollow(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Hollow</span>
          </label>
        </div>
        {p.drawStrokeMode === "cuboid" ? (
          <div className="tool-options-section">
            <p
              className="tool-options-hint"
              style={{ margin: 0, fontSize: "0.85rem", opacity: 0.9 }}
            >
              Cuboid: Click and drag, set a depth, and then click "Done" when you're ready to
              commit.
            </p>
          </div>
        ) : null}
        {p.drawStrokeMode === "cylinder" ? (
          <div className="tool-options-section">
            <p
              className="tool-options-hint"
              style={{ margin: 0, fontSize: "0.85rem", opacity: 0.9 }}
            >
              Cylinder: Click and drag, set a depth, and then click "Done" when you're ready to
              commit.
            </p>
          </div>
        ) : null}
      </>
    );
  }

  if (narrowFill) {
    return (
      <>
        <div className="tool-options-section">
          <label className="tool-options-checkbox-row">
            <input
              type="checkbox"
              checked={p.fillConstrainToPlane}
              onChange={(ev) => p.setFillConstrainToPlane(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Constrain to plane</span>
          </label>
          {p.fillConstrainToPlane ? (
            <label
              className="tool-options-range-label"
              style={{ marginTop: "0.5rem", display: "block" }}
            >
              <span>Plane</span>
              <select
                value={p.planeAxis}
                onChange={(ev) => p.setPlaneAxis(ev.target.value as PlaneAxisApi)}
                disabled={p.loading || p.workBusy}
              >
                <option value="auto">Auto (face)</option>
                <option value="x">X</option>
                <option value="y">Y</option>
                <option value="z">Z</option>
                <option value="camera">Camera (view)</option>
              </select>
            </label>
          ) : null}
        </div>
        <hr
          style={{
            border: "none",
            borderTop: "1px solid rgba(255, 255, 255, 0.12)",
            margin: "0.25rem 0 0.35rem",
          }}
        />
        <div className="tool-options-section">
          <label className="tool-options-checkbox-row">
            <input
              type="checkbox"
              checked={p.fillSelectDiagonals}
              onChange={(ev) => p.setFillSelectDiagonals(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Include diagonals</span>
          </label>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.5rem" }}>
            <input
              type="checkbox"
              checked={p.fillRespectsColor}
              onChange={(ev) => p.setFillRespectsColor(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Respect color</span>
          </label>
        </div>
      </>
    );
  }

  if (narrowSpray) {
    const MAX_BRUSH_SIZE = 33;
    return (
      <>
        <div className="tool-options-section">
          <label className="tool-options-checkbox-row">
            <input
              type="checkbox"
              checked={p.sprayConstrainToPlane}
              onChange={(ev) => p.setSprayConstrainToPlane(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Constrain to plane</span>
          </label>
          {p.sprayConstrainToPlane && (
            <div
              className="tool-options-shape-row"
              role="group"
              aria-label="Plane reference"
              style={{ marginTop: "0.35rem" }}
            >
              {(
                [
                  ["auto", "Auto"],
                  ["camera", "Camera"],
                  ["x", "X"],
                  ["y", "Y"],
                  ["z", "Z"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={
                    p.sprayConstrainToPlaneRef === id
                      ? "tool-options-shape-btn is-active"
                      : "tool-options-shape-btn"
                  }
                  disabled={p.loading || p.workBusy}
                  onClick={() => p.setSprayConstrainToPlaneRef(id)}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
        </div>
        <hr
          style={{
            border: "none",
            borderTop: "1px solid rgba(255, 255, 255, 0.12)",
            margin: "0.25rem 0 0.35rem",
          }}
        />
        <div className="tool-options-section">
          <div className="tool-options-heading">Brush shape</div>
          <div className="tool-options-shape-row" role="group" aria-label="Spray brush shape">
            {(
              [
                ["cube", "Cube"],
                ["sphere", "Sphere"],
                ["pyramid", "Pyramid"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={
                  p.sprayBrushShape === id
                    ? "tool-options-shape-btn is-active"
                    : "tool-options-shape-btn"
                }
                disabled={p.loading || p.workBusy}
                onClick={() => p.setSprayBrushShape(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.35rem" }}>
            <input
              type="checkbox"
              checked={p.brushClipBottomHalf}
              onChange={(ev) => p.setBrushClipBottomHalf(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span title="Uses the clicked face outward normal (world +Y if no solid hit)">
              Outer half (face)
            </span>
          </label>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.5rem" }}>
            <input
              type="checkbox"
              checked={p.spraySizeRange}
              onChange={(ev) => p.setSpraySizeRange(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Size range</span>
          </label>
          {p.spraySizeRange ? (
            <>
              <div
                className="tool-options-heading tool-options-heading-mixed"
                style={{ marginTop: "0.5rem" }}
              >
                Min
              </div>
              <label className="tool-options-range-label tool-options-range-with-value">
                <input
                  type="range"
                  min={0}
                  max={MAX_BRUSH_SIZE - 1}
                  step={1}
                  value={p.sprayRadiusMin}
                  onChange={(ev) => {
                    const v = Number(ev.target.value);
                    p.setSprayRadiusMin(v);
                    if (v > p.sprayRadiusMax) p.setSprayRadiusMax(v);
                  }}
                  disabled={p.loading || p.workBusy}
                />
                <span className="tool-options-range-value" aria-live="polite">
                  {p.sprayRadiusMin + 1}
                </span>
              </label>
              <div
                className="tool-options-heading tool-options-heading-mixed"
                style={{ marginTop: "0.5rem" }}
              >
                Max
              </div>
              <label className="tool-options-range-label tool-options-range-with-value">
                <input
                  type="range"
                  min={0}
                  max={MAX_BRUSH_SIZE - 1}
                  step={1}
                  value={p.sprayRadiusMax}
                  onChange={(ev) => {
                    const v = Number(ev.target.value);
                    p.setSprayRadiusMax(v);
                    if (v < p.sprayRadiusMin) p.setSprayRadiusMin(v);
                  }}
                  disabled={p.loading || p.workBusy}
                />
                <span className="tool-options-range-value" aria-live="polite">
                  {p.sprayRadiusMax + 1}
                </span>
              </label>
            </>
          ) : (
            <>
              <div
                className="tool-options-heading tool-options-heading-mixed"
                style={{ marginTop: "0.5rem" }}
              >
                Size
              </div>
              <label className="tool-options-range-label tool-options-range-with-value">
                <input
                  type="range"
                  min={0}
                  max={MAX_BRUSH_SIZE - 1}
                  step={1}
                  value={p.brushRadius}
                  onChange={(ev) => p.setBrushRadius(Number(ev.target.value))}
                  disabled={p.loading || p.workBusy}
                />
                <span className="tool-options-range-value" aria-live="polite">
                  {p.brushRadius + 1}
                </span>
              </label>
            </>
          )}
          <div
            className="tool-options-heading tool-options-heading-mixed"
            style={{ marginTop: "0.5rem" }}
          >
            Scatter
          </div>
          <label className="tool-options-range-label tool-options-range-with-value">
            <input
              type="range"
              min={0}
              max={MAX_BRUSH_SIZE - 1}
              step={1}
              value={p.sprayScatter}
              onChange={(ev) => p.setSprayScatter(Number(ev.target.value))}
              disabled={p.loading || p.workBusy}
            />
            <span className="tool-options-range-value" aria-live="polite">
              {p.sprayScatter}
            </span>
          </label>
          <label className="tool-options-checkbox-row" style={{ marginTop: "0.5rem" }}>
            <input
              type="checkbox"
              checked={p.selectionStrokeSnapToSurface}
              onChange={(ev) => p.setSelectionStrokeSnapToSurface(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Snap to surface</span>
          </label>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="tool-options-section">
        <div className="tool-options-heading">Brush</div>
        <div className="tool-options-shape-row" role="group" aria-label="Brush shape">
          {(
            [
              ["sphere", "Sphere"],
              ["cube", "Cube"],
              ["pyramid", "Pyr"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={
                p.brushShape === id ? "tool-options-shape-btn is-active" : "tool-options-shape-btn"
              }
              disabled={p.loading || p.workBusy}
              onClick={() => p.setBrushShape(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <label className="tool-options-checkbox-row" style={{ marginTop: "0.35rem" }}>
          <input
            type="checkbox"
            checked={p.brushClipBottomHalf}
            onChange={(ev) => p.setBrushClipBottomHalf(ev.target.checked)}
            disabled={p.loading || p.workBusy}
          />
          <span title="Uses the clicked face outward normal (world +Y if no solid hit)">
            Outer half (face)
          </span>
        </label>
        <label className="tool-options-range-label">
          <span>Diameter</span>
          <input
            type="range"
            min={1}
            max={65}
            step={2}
            value={p.brushRadius * 2 + 1}
            onChange={(ev) => p.setBrushRadius((Number(ev.target.value) - 1) / 2)}
            disabled={p.loading || p.workBusy}
          />
        </label>
      </div>
      <div className="tool-options-section">
        <div className="tool-options-heading">Stroke</div>
        <div className="tool-options-shape-row" role="group" aria-label="Stroke style">
          <button
            type="button"
            className={
              p.strokeDrawStyle === "line"
                ? "tool-options-shape-btn is-active"
                : "tool-options-shape-btn"
            }
            disabled={p.loading || p.workBusy}
            onClick={() => p.setStrokeDrawStyle("line")}
          >
            Line
          </button>
          <button
            type="button"
            className={
              p.strokeDrawStyle === "brush"
                ? "tool-options-shape-btn is-active"
                : "tool-options-shape-btn"
            }
            disabled={p.loading || p.workBusy}
            onClick={() => p.setStrokeDrawStyle("brush")}
          >
            Brush
          </button>
        </div>
        {areaCommon}
      </div>
    </>
  );
}
