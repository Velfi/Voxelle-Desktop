import type { DrawStrokeModeApi, PlaneAxisApi } from "../drawToolModel";

export type AreaShapeControlsProps = {
  loading: boolean;
  workBusy: boolean;
  drawStrokeMode: DrawStrokeModeApi;
  onDrawStrokeModeChange: (m: DrawStrokeModeApi) => void;
  planeAxis: PlaneAxisApi;
  onPlaneAxisChange: (a: PlaneAxisApi) => void;
  fillSelectDiagonals: boolean;
  onFillSelectDiagonalsChange: (v: boolean) => void;
  fillRespectsColor: boolean;
  onFillRespectsColorChange: (v: boolean) => void;
  sprayDensity: number;
  onSprayDensityChange: (v: number) => void;
  /** Selection paths: show spray slider when stroke mode is `spray` or density &gt; 0. Sculpt: density only. */
  spraySliderMode: "selection" | "sculpt";
  selectLabel: string;
  fillOptionLabel: string;
};

export function AreaShapeControls(p: AreaShapeControlsProps) {
  const showSpraySlider =
    p.spraySliderMode === "sculpt"
      ? p.sprayDensity > 0
      : p.drawStrokeMode === "spray" || p.sprayDensity > 0;

  return (
    <>
      <label
        className="tool-options-range-label"
        style={{ marginTop: "0.35rem" }}
      >
        <span>{p.selectLabel}</span>
        <select
          value={p.drawStrokeMode}
          onChange={(ev) =>
            p.onDrawStrokeModeChange(ev.target.value as DrawStrokeModeApi)
          }
          disabled={p.loading || p.workBusy}
        >
          <option value="line">Line</option>
          <option value="spray">Spray</option>
          <option value="plane">Plane</option>
          <option value="precise">Precise</option>
          <option value="circle">Circle</option>
          <option value="cuboid">Cuboid</option>
          <option value="cylinder">Cylinder</option>
          <option value="polygonHull">Polygon hull</option>
          <option value="polygon">Polygon</option>
          <option value="fill">{p.fillOptionLabel}</option>
        </select>
      </label>
      {p.drawStrokeMode === "plane" ? (
        <label
          className="tool-options-range-label"
          style={{ marginTop: "0.25rem" }}
        >
          <span>Plane axis</span>
          <select
            value={p.planeAxis}
            onChange={(ev) =>
              p.onPlaneAxisChange(ev.target.value as PlaneAxisApi)
            }
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
      {p.drawStrokeMode === "fill" ? (
        <div style={{ marginTop: "0.35rem" }}>
          <label
            className="tool-options-range-label"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
            }}
          >
            <input
              type="checkbox"
              checked={p.fillSelectDiagonals}
              onChange={(ev) =>
                p.onFillSelectDiagonalsChange(ev.target.checked)
              }
              disabled={p.loading || p.workBusy}
            />
            <span>Include diagonals (26-connectivity)</span>
          </label>
          <label
            className="tool-options-range-label"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
            }}
          >
            <input
              type="checkbox"
              checked={p.fillRespectsColor}
              onChange={(ev) => p.onFillRespectsColorChange(ev.target.checked)}
              disabled={p.loading || p.workBusy}
            />
            <span>Respect color (stop at different voxel color)</span>
          </label>
        </div>
      ) : null}
      {p.drawStrokeMode === "circle" ? (
        <p
          style={{
            margin: "0.35rem 0 0",
            fontSize: "0.85rem",
            opacity: 0.9,
          }}
        >
          Circle: first click center, second click edge.
        </p>
      ) : null}
      {p.drawStrokeMode === "cuboid" ? (
        <p
          style={{
            margin: "0.35rem 0 0",
            fontSize: "0.85rem",
            opacity: 0.9,
          }}
        >
          Cuboid: drag on a face to set the rectangle, then adjust depth and tap
          Done.
        </p>
      ) : null}
      {p.drawStrokeMode === "cylinder" ? (
        <p
          style={{
            margin: "0.35rem 0 0",
            fontSize: "0.85rem",
            opacity: 0.9,
          }}
        >
          Cylinder: drag on a face to set the disk, then adjust depth and tap
          Done.
        </p>
      ) : null}
      {showSpraySlider ? (
        <label className="tool-options-range-label">
          <span>Spray density</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.02}
            value={p.sprayDensity}
            onChange={(ev) => p.onSprayDensityChange(Number(ev.target.value))}
            disabled={p.loading || p.workBusy}
          />
        </label>
      ) : null}
    </>
  );
}
