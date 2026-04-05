import { type MutableRefObject } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { StrokePhaseHandle } from "./useStrokePhase";
import type { DepthPhaseData } from "./types";

/** Minimal phase-handle interface for components that only need active/cancel/commit/advance/retreat. */
interface SimplePhase {
  active: boolean;
  cancel(): void;
  commit(): void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  enter(phase: string, data: any): void;
  snapshot: { phase: string } | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  advance(patch?: any): void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  retreat(patch?: any): void;
}

interface ViewportTopCenterHud {
  label: string;
  pct: number;
  showFillCancel: boolean;
}

interface Props {
  // Work-phase chip
  viewportTopCenterHud: ViewportTopCenterHud | null;

  // Cuboid / cylinder / polygon depth bar
  cuboidPhase: StrokePhaseHandle<DepthPhaseData>;
  cylinderPhase: StrokePhaseHandle<DepthPhaseData>;
  polygonPhase: StrokePhaseHandle<{ endNorm: { nx: number; ny: number } }>;
  cuboidDepthUi: number;
  setCuboidDepthUi: (n: number) => void;
  cuboidDepthRef: MutableRefObject<number>;
  cylinderDepthUi: number;
  setCylinderDepthUi: (n: number) => void;
  cylinderDepthRef: MutableRefObject<number>;
  polygonDepthUi: number;
  setPolygonDepthUi: (n: number) => void;
  polygonDepthRef: MutableRefObject<number>;
  extrusionDepthEditing: boolean;
  setExtrusionDepthEditing: (v: boolean) => void;
  extrusionDepthDraft: string;
  setExtrusionDepthDraft: (v: string) => void;
  commitCuboidSolidAtScreen: () => void;
  commitCylinderSolidAtScreen: () => void;
  commitPolygonSolid: () => void;

  // Extrude phase bar
  extrudePhase: SimplePhase;
  extrudeProfile: "cube" | "cylinder";
  extrudeTaper: boolean;

  // Roof placement bar
  generatorKind: string;
  roofPins: [number, number, number][];
  setRoofPins: (pins: [number, number, number][]) => void;
  roofPinsRef: MutableRefObject<[number, number, number][]>;
  roofFirstClick: unknown;
  setRoofFirstClick: (v: null) => void;
  roofFirstClickRef: MutableRefObject<unknown>;
  roofAreaShape: string;
  roofStyle: string;
  roofHeight: number;
  roofHollow: boolean;
  activeColor: number;
  activeMaterialRef: MutableRefObject<string>;
  loading: boolean;
  workBusy: boolean;

  // Cloth placement bar
  clothPins: [number, number, number][];
  setClothPins: (pins: [number, number, number][]) => void;
  clothPinsRef: MutableRefObject<[number, number, number][]>;
  clothPhase: SimplePhase;

  // Cloth settings bar
  clothTension: number;
  setClothTension: (v: number) => void;

  // Rope settings bar
  ropePhase: SimplePhase;
  ropeTension: number;
  setRopeTension: (v: number) => void;

  // Generator "adjust in sidebar" bars
  rocksPhase: SimplePhase;
  grassPhase: SimplePhase;
  ashlarPhase: SimplePhase;
  floraPhase: SimplePhase;
  shapePhase: SimplePhase;

  // Squishy session bar
  squishyPhase: SimplePhase;

  // Bone session bar
  bonePhase: SimplePhase;
  setBoneMode: (mode: "add" | "edit") => void;

  // Wall sculpt polygon HUD
  showWallSculptPolygonHud: boolean;
  wallSculptPolygonVerts: [number, number, number][];
  setWallSculptPolygonVerts: (verts: [number, number, number][]) => void;
  commitWallSculptPolygonStroke: () => void;

  // Polygon phase HUD
  showPolygonPhaseHud: boolean;
  strokePolygonVerts: [number, number, number][];
  setStrokePolygonVerts: (verts: [number, number, number][]) => void;
  strokePolygonLastScreenRef: MutableRefObject<{ nx: number; ny: number } | null>;
  applyPolygonStrokeFill: () => void;
}

export function ViewportHUD({
  viewportTopCenterHud,
  cuboidPhase,
  cylinderPhase,
  polygonPhase,
  cuboidDepthUi,
  setCuboidDepthUi,
  cuboidDepthRef,
  cylinderDepthUi,
  setCylinderDepthUi,
  cylinderDepthRef,
  polygonDepthUi,
  setPolygonDepthUi,
  polygonDepthRef,
  extrusionDepthEditing,
  setExtrusionDepthEditing,
  extrusionDepthDraft,
  setExtrusionDepthDraft,
  commitCuboidSolidAtScreen,
  commitCylinderSolidAtScreen,
  commitPolygonSolid,
  extrudePhase,
  extrudeProfile,
  extrudeTaper,
  generatorKind,
  roofPins,
  setRoofPins,
  roofPinsRef,
  roofFirstClick,
  setRoofFirstClick,
  roofFirstClickRef,
  roofAreaShape,
  roofStyle,
  roofHeight,
  roofHollow,
  activeColor,
  activeMaterialRef,
  loading,
  workBusy,
  clothPins,
  setClothPins,
  clothPinsRef,
  clothPhase,
  clothTension,
  setClothTension,
  ropePhase,
  ropeTension,
  setRopeTension,
  rocksPhase,
  grassPhase,
  ashlarPhase,
  floraPhase,
  shapePhase,
  squishyPhase,
  bonePhase,
  setBoneMode,
  showWallSculptPolygonHud,
  wallSculptPolygonVerts,
  setWallSculptPolygonVerts,
  commitWallSculptPolygonStroke,
  showPolygonPhaseHud,
  strokePolygonVerts,
  setStrokePolygonVerts,
  strokePolygonLastScreenRef,
  applyPolygonStrokeFill,
}: Props) {
  return (
    <div
      className="viewport-top-center-hud"
      onPointerDown={(e) => e.stopPropagation()}
      onPointerMove={(e) => e.stopPropagation()}
    >
      {viewportTopCenterHud ? (
        <div className="viewport-work-phase-chip" role="status" aria-live="polite">
          <span className="viewport-work-phase-text">{viewportTopCenterHud.label}</span>
          <span className="viewport-work-phase-pct">{viewportTopCenterHud.pct}%</span>
          {viewportTopCenterHud.showFillCancel ? (
            <button
              type="button"
              className="viewport-work-phase-cancel"
              onClick={() => void invoke("voxel_fill_cancel").catch(() => {})}
            >
              Cancel
            </button>
          ) : null}
        </div>
      ) : null}

      {cuboidPhase.active || cylinderPhase.active || polygonPhase.active ? (
        <div
          className="viewport-cuboid-depth-bar"
          role="dialog"
          aria-label={
            cuboidPhase.active
              ? "Cuboid extrusion depth"
              : polygonPhase.active
                ? "Polygon extrusion depth"
                : "Cylinder extrusion depth"
          }
        >
          <span>Depth</span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => {
              if (cuboidPhase.active) {
                const n = Math.max(-256, cuboidDepthUi - 1);
                cuboidDepthRef.current = n;
                setCuboidDepthUi(n);
                if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
              } else if (polygonPhase.active) {
                const n = Math.max(-256, polygonDepthUi - 1);
                polygonDepthRef.current = n;
                setPolygonDepthUi(n);
                if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
              } else {
                const n = Math.max(-256, cylinderDepthUi - 1);
                cylinderDepthRef.current = n;
                setCylinderDepthUi(n);
                if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
              }
            }}
          >
            −
          </button>
          <input
            type="text"
            inputMode="numeric"
            className="viewport-cuboid-depth-input"
            aria-label="Extrusion depth"
            autoComplete="off"
            value={
              extrusionDepthEditing
                ? extrusionDepthDraft
                : String(
                    cuboidPhase.active
                      ? cuboidDepthUi
                      : polygonPhase.active
                        ? polygonDepthUi
                        : cylinderDepthUi,
                  )
            }
            onChange={(e) => {
              const v = e.target.value;
              if (v === "" || /^-?\d*$/.test(v)) {
                setExtrusionDepthDraft(v);
              }
            }}
            onFocus={(e) => {
              const cur = cuboidPhase.active
                ? cuboidDepthUi
                : polygonPhase.active
                  ? polygonDepthUi
                  : cylinderDepthUi;
              setExtrusionDepthEditing(true);
              setExtrusionDepthDraft(String(cur));
              e.target.select();
            }}
            onBlur={() => {
              const current = cuboidPhase.active
                ? cuboidDepthUi
                : polygonPhase.active
                  ? polygonDepthUi
                  : cylinderDepthUi;
              let n = parseInt(extrusionDepthDraft, 10);
              if (Number.isNaN(n)) n = current;
              n = Math.max(-256, Math.min(256, n));
              if (cuboidPhase.active) {
                setCuboidDepthUi(n);
                cuboidDepthRef.current = n;
              } else if (polygonPhase.active) {
                setPolygonDepthUi(n);
                polygonDepthRef.current = n;
              } else {
                setCylinderDepthUi(n);
                cylinderDepthRef.current = n;
              }
              setExtrusionDepthEditing(false);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                (e.target as HTMLInputElement).blur();
              }
            }}
          />
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => {
              if (cuboidPhase.active) {
                const n = Math.min(256, cuboidDepthUi + 1);
                cuboidDepthRef.current = n;
                setCuboidDepthUi(n);
                if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
              } else if (polygonPhase.active) {
                const n = Math.min(256, polygonDepthUi + 1);
                polygonDepthRef.current = n;
                setPolygonDepthUi(n);
                if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
              } else {
                const n = Math.min(256, cylinderDepthUi + 1);
                cylinderDepthRef.current = n;
                setCylinderDepthUi(n);
                if (extrusionDepthEditing) setExtrusionDepthDraft(String(n));
              }
            }}
          >
            +
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => {
              if (cuboidPhase.active) void commitCuboidSolidAtScreen();
              else if (polygonPhase.active) void commitPolygonSolid();
              else void commitCylinderSolidAtScreen();
            }}
          >
            Done
          </button>
        </div>
      ) : null}

      {extrudePhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Extrude settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Extrude</span>
          <span
            style={{
              fontSize: "0.78rem",
              color: "var(--app-text-muted)",
            }}
          >
            {extrudeProfile === "cylinder" ? "Cylinder" : "Cube"}
            {extrudeTaper ? " tapered" : ""}
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => extrudePhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => extrudePhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {/* Roof placing phase: pin/anchor count + Cancel */}
      {generatorKind === "roof" && (roofPins.length > 0 || roofFirstClick !== null) ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Roof placement">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Roof</span>
          <span
            style={{
              fontSize: "0.78rem",
              color: "var(--app-text-muted)",
            }}
          >
            {roofFirstClick !== null
              ? roofAreaShape === "circle"
                ? "click edge"
                : "click opposite corner"
              : `${roofPins.length} pin${roofPins.length !== 1 ? "s" : ""}`}
          </span>
          <div className="viewport-polygon-phase-actions">
            <button
              type="button"
              className="tool-options-shape-btn"
              disabled={loading || workBusy || roofPins.length < 3}
              onClick={() => {
                void invoke("generator_roof_from_pins_cmd", {
                  args: {
                    pins: roofPins,
                    style: roofStyle,
                    height: roofHeight,
                    thickness: 1,
                    shedEdgeIndex: 0,
                    gableOrientation: 0,
                    breakRatio: 0.5,
                    wallHeight: 3,
                    parapetHeight: 2,
                    saltSkew: 0,
                    hollow: roofHollow,
                    color: activeColor,
                    material: activeMaterialRef.current,
                  },
                })
                  .then(() => {
                    setRoofPins([]);
                    roofPinsRef.current = [];
                    void invoke("voxel_stroke_preview_reset").catch(() => {});
                  })
                  .catch(() => {});
              }}
            >
              Done
            </button>
            <button
              type="button"
              className="tool-options-shape-btn"
              onClick={() => {
                setRoofPins([]);
                roofPinsRef.current = [];
                roofFirstClickRef.current = null;
                setRoofFirstClick(null);
                void invoke("voxel_stroke_preview_reset").catch(() => {});
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {/* Cloth placing phase: pin count + Done/Cancel */}
      {generatorKind === "cloth" && !clothPhase.active && clothPins.length > 0 ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Cloth pin placement">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Cloth</span>
          <span
            style={{
              fontSize: "0.78rem",
              color: "var(--app-text-muted)",
            }}
          >
            {clothPins.length} pin{clothPins.length !== 1 ? "s" : ""}
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => {
              setClothPins([]);
              clothPinsRef.current = [];
              void invoke("voxel_stroke_preview_reset").catch(() => {});
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            disabled={clothPins.length < 3}
            onClick={() => clothPhase.enter("settings", {})}
          >
            Done
          </button>
        </div>
      ) : null}

      {/* Cloth settings phase: tension slider + Done/Cancel */}
      {clothPhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Cloth settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Cloth</span>
          <span
            style={{
              fontSize: "0.78rem",
              color: "var(--app-text-muted)",
            }}
          >
            Tension
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.02}
            value={clothTension}
            onChange={(ev) => setClothTension(Number(ev.target.value))}
            style={{ width: 80 }}
            title="0 = loose drape, 1 = stiff"
          />
          <span
            style={{
              fontSize: "0.78rem",
              minWidth: "2.2em",
              textAlign: "right",
            }}
          >
            {clothTension.toFixed(2)}
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => clothPhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => clothPhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {/* Rope settings phase: tension + sag sliders + Done/Cancel */}
      {ropePhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Rope settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Rope</span>
          <span
            style={{
              fontSize: "0.78rem",
              color: "var(--app-text-muted)",
            }}
          >
            Tension
          </span>
          <input
            type="range"
            min={-1}
            max={1}
            step={0.02}
            value={ropeTension}
            onChange={(ev) => setRopeTension(Number(ev.target.value))}
            style={{ width: 60 }}
            title="-1 = very loose, 0 = loose, 1 = taut"
          />
          <span
            style={{
              fontSize: "0.78rem",
              minWidth: "2.2em",
              textAlign: "right",
            }}
          >
            {ropeTension.toFixed(2)}
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => ropePhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => ropePhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {rocksPhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Rocks settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Rocks</span>
          <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
            Adjust in sidebar — Enter to place
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => rocksPhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => rocksPhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {grassPhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Grass settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Grass</span>
          <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
            Adjust in sidebar — Enter to place
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => grassPhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => grassPhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {ashlarPhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Ashlar settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Ashlar</span>
          <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
            Adjust in sidebar — Enter to place
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => ashlarPhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => ashlarPhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {floraPhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Flora settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Flora</span>
          <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
            Adjust in sidebar — Enter to place
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => floraPhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => floraPhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {shapePhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Shape settings">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Shape</span>
          <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
            Adjust in sidebar — Enter to place
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => shapePhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => shapePhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {squishyPhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Squishy session">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Squishy</span>
          <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
            Adjust in sidebar — Enter to commit, Esc to cancel
          </span>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => squishyPhase.cancel()}
          >
            Cancel
          </button>
          <button
            type="button"
            className="tool-options-shape-btn"
            onClick={() => squishyPhase.commit()}
          >
            Done
          </button>
        </div>
      ) : null}

      {bonePhase.active ? (
        <div className="viewport-cuboid-depth-bar" role="dialog" aria-label="Bone session">
          <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Bone</span>
          <span style={{ fontSize: "0.78rem", color: "var(--app-text-muted)" }}>
            {bonePhase.snapshot?.phase === "build"
              ? "Place joints — click Next to pose"
              : "Pose joints — Enter to commit, Esc to cancel"}
          </span>
          {bonePhase.snapshot?.phase === "build" ? (
            <>
              <button onClick={() => bonePhase.cancel()}>Cancel</button>
              <button
                onClick={() => {
                  setBoneMode("edit");
                  bonePhase.advance();
                }}
              >
                Next
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => {
                  setBoneMode("add");
                  bonePhase.retreat();
                }}
              >
                Back
              </button>
              <button onClick={() => bonePhase.commit()}>Done</button>
            </>
          )}
        </div>
      ) : null}

      {showWallSculptPolygonHud ? (
        <div className="viewport-polygon-phase-bar" role="dialog" aria-label="Wall polygon outline">
          <p className="viewport-polygon-phase-hint">
            Wall outline: {wallSculptPolygonVerts.length} corners. Click the surface to add; Done
            applies (min 2 corners).
          </p>
          <div className="viewport-polygon-phase-actions">
            <button
              type="button"
              className="tool-options-shape-btn"
              disabled={loading || workBusy}
              onClick={() => setWallSculptPolygonVerts([])}
            >
              Clear
            </button>
            <button
              type="button"
              className="tool-options-shape-btn"
              disabled={loading || workBusy || wallSculptPolygonVerts.length < 2}
              onClick={() => commitWallSculptPolygonStroke()}
            >
              Done
            </button>
          </div>
        </div>
      ) : null}

      {showPolygonPhaseHud && !polygonPhase.active ? (
        <div className="viewport-polygon-phase-bar" role="dialog" aria-label="Polygon area">
          <p className="viewport-polygon-phase-hint">
            Vertices: {strokePolygonVerts.length}. Click to add corners; Apply with three or more.
          </p>
          <div className="viewport-polygon-phase-actions">
            <button
              type="button"
              className="tool-options-shape-btn"
              disabled={loading || workBusy}
              onClick={() => {
                setStrokePolygonVerts([]);
                strokePolygonLastScreenRef.current = null;
              }}
            >
              Clear
            </button>
            <button
              type="button"
              className="tool-options-shape-btn"
              disabled={loading || workBusy || strokePolygonVerts.length < 3}
              onClick={() => {
                applyPolygonStrokeFill();
              }}
            >
              Apply
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
