/**
 * Sculpt tool: all brush / terrain / wall / smooth UI state and refs in one place.
 */

import { useRef, useState } from "react";
import { useLatestRef } from "./useLatestRef";
import type {
  SculptStrokeModeApi,
  SculptBrushShapeUi,
  SculptSmoothVariantApi,
  TerrainSculptOpApi,
  WallAreaShapeApi,
  SprayDirectionApi,
} from "../types";

export interface SculptToolState {
  sculptStrokeMode: SculptStrokeModeApi;
  setSculptStrokeMode: React.Dispatch<React.SetStateAction<SculptStrokeModeApi>>;
  terrainSculptOp: TerrainSculptOpApi;
  setTerrainSculptOp: React.Dispatch<React.SetStateAction<TerrainSculptOpApi>>;
  terrainBaseY: number;
  setTerrainBaseY: React.Dispatch<React.SetStateAction<number>>;
  terrainSmoothRadius: number;
  setTerrainSmoothRadius: React.Dispatch<React.SetStateAction<number>>;
  terrainFlattenUseBaseY: boolean;
  setTerrainFlattenUseBaseY: React.Dispatch<React.SetStateAction<boolean>>;
  terrainSubVoxel: boolean;
  setTerrainSubVoxel: React.Dispatch<React.SetStateAction<boolean>>;
  terrainHoverY: number | null;
  setTerrainHoverY: React.Dispatch<React.SetStateAction<number | null>>;
  sculptSmoothPasses: number;
  setSculptSmoothPasses: React.Dispatch<React.SetStateAction<number>>;
  sculptBrushRadius: number;
  setSculptBrushRadius: React.Dispatch<React.SetStateAction<number>>;
  sculptBrushStrength: number;
  setSculptBrushStrength: React.Dispatch<React.SetStateAction<number>>;
  sculptBrushFalloff: number;
  setSculptBrushFalloff: React.Dispatch<React.SetStateAction<number>>;
  sculptBrushShapeUi: SculptBrushShapeUi;
  setSculptBrushShapeUi: React.Dispatch<React.SetStateAction<SculptBrushShapeUi>>;
  extrudeDirectionRef: "camera" | "auto" | "x" | "y" | "z";
  setExtrudeDirectionRef: React.Dispatch<React.SetStateAction<"camera" | "auto" | "x" | "y" | "z">>;
  extrudeProfile: "cube" | "cylinder";
  setExtrudeProfile: React.Dispatch<React.SetStateAction<"cube" | "cylinder">>;
  extrudeEndCap: "flat" | "rounded" | "pointed";
  setExtrudeEndCap: React.Dispatch<React.SetStateAction<"flat" | "rounded" | "pointed">>;
  extrudeTaper: boolean;
  setExtrudeTaper: React.Dispatch<React.SetStateAction<boolean>>;
  extrudeTaperStart: number;
  setExtrudeTaperStart: React.Dispatch<React.SetStateAction<number>>;
  extrudeTaperEnd: number;
  setExtrudeTaperEnd: React.Dispatch<React.SetStateAction<number>>;
  sculptExtrudeAutoCommitOnMouseUp: boolean;
  setSculptExtrudeAutoCommitOnMouseUp: React.Dispatch<React.SetStateAction<boolean>>;
  wallAreaShape: WallAreaShapeApi;
  setWallAreaShape: React.Dispatch<React.SetStateAction<WallAreaShapeApi>>;
  sprayDirection: SprayDirectionApi;
  setSprayDirection: React.Dispatch<React.SetStateAction<SprayDirectionApi>>;
  wallWidthIndex: number;
  setWallWidthIndex: React.Dispatch<React.SetStateAction<number>>;
  wallHeightVox: number;
  setWallHeightVox: React.Dispatch<React.SetStateAction<number>>;
  wallLockStartHeight: boolean;
  setWallLockStartHeight: React.Dispatch<React.SetStateAction<boolean>>;
  wallAxisAlign: boolean;
  setWallAxisAlign: React.Dispatch<React.SetStateAction<boolean>>;
  sculptSmoothVariant: SculptSmoothVariantApi;
  setSculptSmoothVariant: React.Dispatch<React.SetStateAction<SculptSmoothVariantApi>>;
  smoothNeighborRadius: number;
  setSmoothNeighborRadius: React.Dispatch<React.SetStateAction<number>>;
  smoothAggressiveness: number;
  setSmoothAggressiveness: React.Dispatch<React.SetStateAction<number>>;
  smoothLaplacianIterations: number;
  setSmoothLaplacianIterations: React.Dispatch<React.SetStateAction<number>>;
  smoothLaplacianRelaxPct: number;
  setSmoothLaplacianRelaxPct: React.Dispatch<React.SetStateAction<number>>;
  wallSculptPolygonVerts: [number, number, number][];
  setWallSculptPolygonVerts: React.Dispatch<React.SetStateAction<[number, number, number][]>>;

  extrudeDirectionRefRef: React.MutableRefObject<"camera" | "auto" | "x" | "y" | "z">;
  extrudeProfileRef: React.MutableRefObject<"cube" | "cylinder">;
  extrudeEndCapRef: React.MutableRefObject<"flat" | "rounded" | "pointed">;
  extrudeTaperRef: React.MutableRefObject<boolean>;
  extrudeTaperStartRef: React.MutableRefObject<number>;
  extrudeTaperEndRef: React.MutableRefObject<number>;
  sculptExtrudeAutoCommitOnMouseUpRef: React.MutableRefObject<boolean>;
  terrainSculptOpRef: React.MutableRefObject<TerrainSculptOpApi>;
  terrainBaseYRef: React.MutableRefObject<number>;
  terrainSmoothRadiusRef: React.MutableRefObject<number>;
  terrainFlattenUseBaseYRef: React.MutableRefObject<boolean>;
  terrainSubVoxelRef: React.MutableRefObject<boolean>;
  lastTerrainHoverMsRef: React.MutableRefObject<number>;
  sculptSmoothPassesRef: React.MutableRefObject<number>;
  sculptBrushRadiusRef: React.MutableRefObject<number>;
  sculptBrushStrengthRef: React.MutableRefObject<number>;
  sculptBrushFalloffRef: React.MutableRefObject<number>;
  sculptBrushShapeUiRef: React.MutableRefObject<SculptBrushShapeUi>;
  wallAreaShapeRef: React.MutableRefObject<WallAreaShapeApi>;
  sprayDirectionRef: React.MutableRefObject<SprayDirectionApi>;
  wallWidthIndexRef: React.MutableRefObject<number>;
  wallHeightVoxRef: React.MutableRefObject<number>;
  wallLockStartHeightRef: React.MutableRefObject<boolean>;
  wallAxisAlignRef: React.MutableRefObject<boolean>;
  sculptSmoothVariantRef: React.MutableRefObject<SculptSmoothVariantApi>;
  smoothNeighborRadiusRef: React.MutableRefObject<number>;
  smoothAggressivenessRef: React.MutableRefObject<number>;
  smoothLaplacianIterationsRef: React.MutableRefObject<number>;
  smoothLaplacianRelaxPctRef: React.MutableRefObject<number>;
  wallSculptPolygonVertsRef: React.MutableRefObject<[number, number, number][]>;
}

export function useSculptToolState(): SculptToolState {
  const [sculptStrokeMode, setSculptStrokeMode] = useState<SculptStrokeModeApi>("draw");
  const [terrainSculptOp, setTerrainSculptOp] = useState<TerrainSculptOpApi>("raise");
  const [terrainBaseY, setTerrainBaseY] = useState(0);
  const [terrainSmoothRadius, setTerrainSmoothRadius] = useState(2);
  const [terrainFlattenUseBaseY, setTerrainFlattenUseBaseY] = useState(false);
  const [terrainSubVoxel, setTerrainSubVoxel] = useState(false);
  const [terrainHoverY, setTerrainHoverY] = useState<number | null>(null);
  const [sculptSmoothPasses, setSculptSmoothPasses] = useState(1);
  const [sculptBrushRadius, setSculptBrushRadius] = useState(2);
  const [sculptBrushStrength, setSculptBrushStrength] = useState(100);
  const [sculptBrushFalloff, setSculptBrushFalloff] = useState(0);
  const [sculptBrushShapeUi, setSculptBrushShapeUi] = useState<SculptBrushShapeUi>("circle");
  const [extrudeDirectionRef, setExtrudeDirectionRef] = useState<
    "camera" | "auto" | "x" | "y" | "z"
  >("camera");
  const [extrudeProfile, setExtrudeProfile] = useState<"cube" | "cylinder">("cube");
  const [extrudeEndCap, setExtrudeEndCap] = useState<"flat" | "rounded" | "pointed">("flat");
  const [extrudeTaper, setExtrudeTaper] = useState(false);
  const [extrudeTaperStart, setExtrudeTaperStart] = useState(3);
  const [extrudeTaperEnd, setExtrudeTaperEnd] = useState(0);
  const [sculptExtrudeAutoCommitOnMouseUp, setSculptExtrudeAutoCommitOnMouseUp] = useState(true);
  const [wallAreaShape, setWallAreaShape] = useState<WallAreaShapeApi>("brush");
  const [sprayDirection, setSprayDirection] = useState<SprayDirectionApi>("auto");
  const [wallWidthIndex, setWallWidthIndex] = useState(0);
  const [wallHeightVox, setWallHeightVox] = useState(2);
  const [wallLockStartHeight, setWallLockStartHeight] = useState(false);
  const [wallAxisAlign, setWallAxisAlign] = useState(false);
  const [sculptSmoothVariant, setSculptSmoothVariant] =
    useState<SculptSmoothVariantApi>("majority");
  const [smoothNeighborRadius, setSmoothNeighborRadius] = useState(0);
  const [smoothAggressiveness, setSmoothAggressiveness] = useState(100);
  const [smoothLaplacianIterations, setSmoothLaplacianIterations] = useState(4);
  const [smoothLaplacianRelaxPct, setSmoothLaplacianRelaxPct] = useState(50);
  const [wallSculptPolygonVerts, setWallSculptPolygonVerts] = useState<[number, number, number][]>(
    [],
  );

  const extrudeDirectionRefRef = useLatestRef(extrudeDirectionRef);
  const extrudeProfileRef = useLatestRef(extrudeProfile);
  const extrudeEndCapRef = useLatestRef(extrudeEndCap);
  const extrudeTaperRef = useLatestRef(extrudeTaper);
  const extrudeTaperStartRef = useLatestRef(extrudeTaperStart);
  const extrudeTaperEndRef = useLatestRef(extrudeTaperEnd);
  const sculptExtrudeAutoCommitOnMouseUpRef = useLatestRef(sculptExtrudeAutoCommitOnMouseUp);
  const terrainSculptOpRef = useLatestRef(terrainSculptOp);
  const terrainBaseYRef = useLatestRef(terrainBaseY);
  const terrainSmoothRadiusRef = useLatestRef(terrainSmoothRadius);
  const terrainFlattenUseBaseYRef = useLatestRef(terrainFlattenUseBaseY);
  const terrainSubVoxelRef = useLatestRef(terrainSubVoxel);
  const lastTerrainHoverMsRef = useRef(0);
  const sculptSmoothPassesRef = useLatestRef(sculptSmoothPasses);
  const sculptBrushRadiusRef = useLatestRef(sculptBrushRadius);
  const sculptBrushStrengthRef = useLatestRef(sculptBrushStrength);
  const sculptBrushFalloffRef = useLatestRef(sculptBrushFalloff);
  const sculptBrushShapeUiRef = useLatestRef(sculptBrushShapeUi);
  const wallAreaShapeRef = useLatestRef(wallAreaShape);
  const sprayDirectionRef = useLatestRef(sprayDirection);
  const wallWidthIndexRef = useLatestRef(wallWidthIndex);
  const wallHeightVoxRef = useLatestRef(wallHeightVox);
  const wallLockStartHeightRef = useLatestRef(wallLockStartHeight);
  const wallAxisAlignRef = useLatestRef(wallAxisAlign);
  const sculptSmoothVariantRef = useLatestRef(sculptSmoothVariant);
  const smoothNeighborRadiusRef = useLatestRef(smoothNeighborRadius);
  const smoothAggressivenessRef = useLatestRef(smoothAggressiveness);
  const smoothLaplacianIterationsRef = useLatestRef(smoothLaplacianIterations);
  const smoothLaplacianRelaxPctRef = useLatestRef(smoothLaplacianRelaxPct);
  const wallSculptPolygonVertsRef = useLatestRef(wallSculptPolygonVerts);

  return {
    sculptStrokeMode,
    setSculptStrokeMode,
    terrainSculptOp,
    setTerrainSculptOp,
    terrainBaseY,
    setTerrainBaseY,
    terrainSmoothRadius,
    setTerrainSmoothRadius,
    terrainFlattenUseBaseY,
    setTerrainFlattenUseBaseY,
    terrainSubVoxel,
    setTerrainSubVoxel,
    terrainHoverY,
    setTerrainHoverY,
    sculptSmoothPasses,
    setSculptSmoothPasses,
    sculptBrushRadius,
    setSculptBrushRadius,
    sculptBrushStrength,
    setSculptBrushStrength,
    sculptBrushFalloff,
    setSculptBrushFalloff,
    sculptBrushShapeUi,
    setSculptBrushShapeUi,
    extrudeDirectionRef,
    setExtrudeDirectionRef,
    extrudeProfile,
    setExtrudeProfile,
    extrudeEndCap,
    setExtrudeEndCap,
    extrudeTaper,
    setExtrudeTaper,
    extrudeTaperStart,
    setExtrudeTaperStart,
    extrudeTaperEnd,
    setExtrudeTaperEnd,
    sculptExtrudeAutoCommitOnMouseUp,
    setSculptExtrudeAutoCommitOnMouseUp,
    wallAreaShape,
    setWallAreaShape,
    sprayDirection,
    setSprayDirection,
    wallWidthIndex,
    setWallWidthIndex,
    wallHeightVox,
    setWallHeightVox,
    wallLockStartHeight,
    setWallLockStartHeight,
    wallAxisAlign,
    setWallAxisAlign,
    sculptSmoothVariant,
    setSculptSmoothVariant,
    smoothNeighborRadius,
    setSmoothNeighborRadius,
    smoothAggressiveness,
    setSmoothAggressiveness,
    smoothLaplacianIterations,
    setSmoothLaplacianIterations,
    smoothLaplacianRelaxPct,
    setSmoothLaplacianRelaxPct,
    wallSculptPolygonVerts,
    setWallSculptPolygonVerts,
    extrudeDirectionRefRef,
    extrudeProfileRef,
    extrudeEndCapRef,
    extrudeTaperRef,
    extrudeTaperStartRef,
    extrudeTaperEndRef,
    sculptExtrudeAutoCommitOnMouseUpRef,
    terrainSculptOpRef,
    terrainBaseYRef,
    terrainSmoothRadiusRef,
    terrainFlattenUseBaseYRef,
    terrainSubVoxelRef,
    lastTerrainHoverMsRef,
    sculptSmoothPassesRef,
    sculptBrushRadiusRef,
    sculptBrushStrengthRef,
    sculptBrushFalloffRef,
    sculptBrushShapeUiRef,
    wallAreaShapeRef,
    sprayDirectionRef,
    wallWidthIndexRef,
    wallHeightVoxRef,
    wallLockStartHeightRef,
    wallAxisAlignRef,
    sculptSmoothVariantRef,
    smoothNeighborRadiusRef,
    smoothAggressivenessRef,
    smoothLaplacianIterationsRef,
    smoothLaplacianRelaxPctRef,
    wallSculptPolygonVertsRef,
  };
}
