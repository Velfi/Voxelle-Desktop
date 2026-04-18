/**
 * Viewport pointer-event handlers, mechanically extracted from App.tsx.
 *
 * The hook accepts a ref whose `.current` is populated each render with
 * all component-level locals the handlers need. The handler bodies are
 * verbatim copies from App.tsx; they close over the destructured locals
 * which are refreshed on every render (same closure semantics as before).
 *
 * **Pipeline A vs B:** Manipulator/gizmo paths (gestureRef) vs scene tools
 * (strokes, preview, camera). See docs/agents/viewport-pointer-pipeline.md
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getStrokeDispatch, strokeModeSkipsDrag } from "../drawToolModel";
import { layoutViewportCssSize, sculptBrushShapeToRust } from "../constants";
import type {
  CuboidPlaneGeo,
  ViewportCursorDebugPayload,
  ViewportCursorDebugScreen,
} from "../types";
import { useLatestRef } from "./useLatestRef";
import { getViewportSceneBehavior } from "./viewportPointer/sceneBehavior";

/** Opaque ref bag populated by App every render. */
export type ViewportPointerLocals = any;

export function useViewportPointer(localsRef: React.MutableRefObject<ViewportPointerLocals>) {
  // Stable ref owned by the hook
  const planeStrokeDebugEnabledRef = useRef(true);
  const bonePendingJointRef = useRef<number | null>(null);
  const boneAddDragJointRef = useRef<number | null>(null);
  const squishyAddDragBallIdRef = useRef<number | null>(null);
  // Track the last (nx, ny, mode) sent to sync_preview_input so we can skip
  // identical calls when the cursor hasn't moved between RAF frames.
  const lastSyncPreviewNxRef = useRef<number>(NaN);
  const lastSyncPreviewNyRef = useRef<number>(NaN);
  const lastSyncPreviewModeRef = useRef<string>("");

  // Destructure App locals from ref bag
  // (refreshed each render; handlers close over the latest snapshot)
  const {
    activePointerIdRef,
    ashlarPreviewSeedRef,
    capturedPointerIdRef,
    currentStrokeSeedRef,
    dragDidEditRef,
    extrudeDirectionRefRef,
    extrudeEndCapRef,
    extrudeGizmoRef,
    extrudeProfileRef,
    extrudeRedragRef,
    extrudeStartNormRef,
    extrudeTaperEndRef,
    extrudeTaperRef,
    extrudeTaperStartRef,
    sculptExtrudeAutoCommitOnMouseUpRef,
    eyedropperReturnModeRef,
    floraPreviewSeedRef,
    flyMouseLookActiveRef,
    gestureRef,
    grassAutoCommitOnMouseUpRef,
    gizmoHoverRef,
    gizmoRef,
    grassPreviewSeedRef,
    interactionBlockedRef,
    lastCursorScreenRef,
    lastRef,
    lastStrokeEditMsRef,
    lastStrokeNormRef,
    lastTerrainHoverMsRef,
    lastViewportPickNormRef,
    lastWallHoverMsRef,
    maxPointerMoveRef,
    onPointerUpRef,
    pendingPointerUpRef,
    pointerStartRef,
    probingRef,
    rockPreviewSeedRef,
    rocksAutoCommitOnMouseUpRef,
    roofAreaShapeRef,
    roofFirstClickRef,
    roofPinsRef,
    sculptBrushFalloffRef,
    sculptBrushRadiusRef,
    sculptBrushShapeUiRef,
    sculptBrushStrengthRef,
    sculptSmoothPassesRef,
    sculptSmoothVariantRef,
    selectionStrokeBegunRef,
    selectionStrokeSnapToSurfaceRef,
    smoothAggressivenessRef,
    smoothLaplacianIterationsRef,
    smoothLaplacianRelaxPctRef,
    smoothNeighborRadiusRef,
    sprayDirectionRef,
    squishyModeRef,
    stampOriginXRef,
    stampOriginZRef,
    stampRotXRef,
    stampRotYRef,
    stampRotZRef,
    startScreenLogoLoadedRef,
    strokeClickRef,
    strokePolygonLastScreenRef,
    strokePolygonVertsRef,
    strokeShiftKeyRef,
    strokeViewportStartRef,
    surfacePhysRef,
    terrainBaseYRef,
    terrainFlattenUseBaseYRef,
    terrainSculptOpRef,
    terrainSmoothRadiusRef,
    terrainSubVoxelRef,
    viewportCursorDebugRafRef,
    viewportCursorDebugScreenRef,
    viewportPhysRef,
    viewportRef,
    wallAreaShapeRef,
    wallAxisAlignRef,
    wallHeightVoxRef,
    wallLockStartHeightRef,
    wallSculptPolygonVertsRef,
    wallWidthIndexRef,
    cuboidDepthRef,
    cylinderDepthRef,
    setCuboidDepthUi,
    setCylinderDepthUi,
    setRoofFirstClick,
    setRoofPins,
    setRopeFirstScreen,
    setRopeFirstVoxel,
    setShapeSize,
    setShapeRotX,
    setShapeRotY,
    setShapeRotZ,
    setSelectionCount,
    setSquishyBallCount,
    setStrokePolygonVerts,
    setTerrainHoverY,
    setViewportCursorDebugJs,
    setViewportCursorDebugRust,
    setViewportCursorDebugScreen,
    setWallSculptPolygonVerts,
    ashlarPhase,
    ashlarAutoCommitOnMouseUpRef,
    clothPhase,
    cuboidPhase,
    cylinderPhase,
    extrudePhase,
    floraPhase,
    grassPhase,
    rocksPhase,
    ropePhase,
    shapePhase,
    shapeGizmoPosRef,
    squishyPhase,
    floraAutoCommitOnMouseUpRef,
    bonePhase,
    boneModeRef,
    boneDefaultRadiusRef,
    ikEnabledRef: _ikEnabledRef,
    setBoneJointCount,
    setBoneBoneCount,
    loading,
    workBusy,
    fillOperationPending,
    viewportCursorDebugEnabled,
    ropeFirstScreen,
    mergedStrokeAux,
    syncPreviewInput,
    placeAshlarAtScreen,
    placeFloraAtScreen,
    placeGrassAtScreen,
    placeRocksAtScreen,
    previewModeForSync,
    runStrokeAtScreen,
    clearPreview,
    handleClothPinClick,
    activateFlyMouseLook,
    releaseFlyMouseLook,
    // Tool state values (passed from App alongside context)
    interactionMode: _interactionMode,
    activeColor: _activeColor,
    activeMaterial: _activeMaterial,
    brushRadius: _brushRadius,
    brushShape: _brushShape,
    brushClipBottomHalf: _brushClipBottomHalf,
    mirrorX: _mirrorX,
    mirrorY: _mirrorY,
    mirrorZ: _mirrorZ,
    strokeDrawStyle: _strokeDrawStyle,
    drawStrokeMode: _drawStrokeMode,
    planeAxis: _planeAxis,
    sprayDensity: _sprayDensity,
    selectedColors: _selectedColors,
    paintColorDistrib: _paintColorDistrib,
    matchMaterialSelectColor: _matchMaterialSelectColor,
    generatorKind: _generatorKind,
    sculptStrokeMode: _sculptStrokeMode,
    setInteractionMode,
    setActiveColor,
    setActiveMaterial,
  } = localsRef.current;

  // Stable refs for tool state values (needed inside async callbacks).
  const interactionModeRef = useLatestRef(_interactionMode);
  const activeColorRef = useLatestRef(_activeColor);
  const activeMaterialRef = useLatestRef(_activeMaterial);
  const brushRadiusRef = useLatestRef(_brushRadius);
  const brushShapeRef = useLatestRef(_brushShape);
  const brushClipBottomHalfRef = useLatestRef(_brushClipBottomHalf);
  const mirrorXRef = useLatestRef(_mirrorX);
  const mirrorYRef = useLatestRef(_mirrorY);
  const mirrorZRef = useLatestRef(_mirrorZ);
  const strokeDrawStyleRef = useLatestRef(_strokeDrawStyle);
  const drawStrokeModeRef = useLatestRef(_drawStrokeMode);
  const planeAxisRef = useLatestRef(_planeAxis);
  const sprayDensityRef = useLatestRef(_sprayDensity);
  const selectedColorsRef = useLatestRef(_selectedColors);
  const paintColorDistribRef = useLatestRef(_paintColorDistrib);
  const matchMaterialSelectColorRef = useLatestRef(_matchMaterialSelectColor);
  const generatorKindRef = useLatestRef(_generatorKind);
  const sculptStrokeModeRef = useLatestRef(_sculptStrokeMode);

  // ---- Private helper: clientToViewportNormalized ----
  function clientToViewportNormalized(e: React.PointerEvent) {
    const el = viewportRef.current;
    if (!el) return { nx: 0.5, ny: 0.5 };
    const rect = el.getBoundingClientRect();
    const rw = rect.width;
    const rh = rect.height;
    if (rw <= 0 || rh <= 0) return { nx: 0.5, ny: 0.5 };

    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;
    return {
      nx: Math.min(1, Math.max(0, relX / rw)),
      ny: Math.min(1, Math.max(0, relY / rh)),
    };
  }

  // ---- Private helper: logPlaneStrokeDebug ----
  function logPlaneStrokeDebug(
    _phase: string,
    _e: React.PointerEvent,
    _extra?: Record<string, unknown>,
  ) {
    if (!planeStrokeDebugEnabledRef.current) return;
    const mode = interactionModeRef.current;
    const sm = drawStrokeModeRef.current;
    if (!(sm === "plane" && (mode === "add" || mode === "remove" || mode === "paint"))) {
      return;
    }
    void gestureRef.current;
  }

  function generatorPreviewPhaseActive() {
    return (
      rocksPhase.active ||
      grassPhase.active ||
      ropePhase.active ||
      clothPhase.active ||
      ashlarPhase.active ||
      floraPhase.active ||
      shapePhase.active
    );
  }

  // ---- Private helper: resetPointerGesture ----
  function resetPointerGesture(reason: string, e?: React.PointerEvent) {
    if (e) {
      logPlaneStrokeDebug(`gesture:reset:${reason}`, e);
    }
    gestureRef.current = null;
    activePointerIdRef.current = null;
    pointerStartRef.current = null;
    maxPointerMoveRef.current = 0;
    pendingPointerUpRef.current = null;
  }

  function restoreBonePosePreviewAtEvent(e: React.PointerEvent) {
    if (interactionModeRef.current !== "bone" || bonePhase.ref.current?.phase !== "pose") {
      return;
    }
    const { nx, ny } = clientToViewportNormalized(e);
    const mode = previewModeForSync(interactionModeRef.current);
    // Invalidate the hover-sync cache first so later move events are not
    // blocked if this immediate restore races with capture/leave handling.
    lastSyncPreviewNxRef.current = NaN;
    lastSyncPreviewNyRef.current = NaN;
    lastSyncPreviewModeRef.current = "";
    syncPreviewInput(nx, ny, mode, true);
  }

  // ---- Private helper: strokeViewportLineStartNorm ----
  function strokeViewportLineStartNorm(): { nx: number; ny: number } | null {
    const start = strokeViewportStartRef.current;
    if (!start) return null;
    if (strokeDrawStyleRef.current === "line") return start;
    if (strokeDrawStyleRef.current === "brush" && drawStrokeModeRef.current !== "spray") {
      return start;
    }
    return null;
  }

  // ---- Private helper: invokeSelectionSpecialClick ----
  function invokeSelectionSpecialClick(interaction: string, nx: number, ny: number) {
    const cmd =
      interaction === "selectByColor"
        ? "selection_add_by_color_at_screen"
        : interaction === "selectCoplanar"
          ? "selection_add_coplanar_at_screen"
          : interaction === "selectCoplanarEmpty"
            ? "selection_add_coplanar_empty_at_screen"
            : null;
    if (!cmd) return;
    const args: Record<string, unknown> = { nx, ny };
    if (interaction === "selectByColor") {
      args.matchMaterial = matchMaterialSelectColorRef.current;
    }
    if (strokeShiftKeyRef.current) {
      args.combineModeOverride = "add";
    }
    void invoke<number>(cmd, { args })
      .then((n) => {
        if (n > 0) {
          void invoke<number>("selection_get_count").then((c) => setSelectionCount(c));
        }
      })
      .catch(() => {});
  }

  // ---- Private helper: handleStrokeAnchorClick ----
  async function handleStrokeAnchorClick(nx: number, ny: number) {
    const dispatch = getStrokeDispatch(interactionModeRef.current);
    if (!dispatch) return;
    const tool = dispatch.kind === "edit" ? dispatch.tool : "remove";
    const c = await invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
      args: {
        nx,
        ny,
        tool,
        strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
      },
    });
    if (!c) return;
    const sm = drawStrokeModeRef.current;
    if (sm === "fill") {
      runStrokeAtScreen(nx, ny, {});
      return;
    }
    if (sm === "polygon" || sm === "polygonHull") {
      setStrokePolygonVerts((v: any) => {
        const idx = v.findIndex((p: any) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2]);
        const next = idx >= 0 ? v.filter((_: any, i: number) => i !== idx) : [...v, c];
        strokePolygonVertsRef.current = next;
        return next;
      });
      strokePolygonLastScreenRef.current = { nx, ny };
      queueMicrotask(() => {
        syncPreviewInput(nx, ny, previewModeForSync(interactionModeRef.current));
      });
      return;
    }
    if (sm === "circle") {
      const r = strokeClickRef.current;
      if (!r.circleCenter) {
        r.circleCenter = c;
      } else {
        runStrokeAtScreen(nx, ny, {
          circleCenter: r.circleCenter,
          circleEdge: c,
        });
        r.circleCenter = null;
      }
      return;
    }
  }

  // ---- Private helper: handleWallSculptPolygonClick ----
  async function handleWallSculptPolygonClick(nx: number, ny: number) {
    const c = await invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
      args: {
        nx,
        ny,
        tool: "add",
        strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
      },
    });
    if (!c) return;
    setWallSculptPolygonVerts((v: any) => {
      const idx = v.findIndex((p: any) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2]);
      const next = idx >= 0 ? v.filter((_: any, i: number) => i !== idx) : [...v, c];
      wallSculptPolygonVertsRef.current = next;
      return next;
    });
  }

  // ---- Pipeline A: manipulator pointer-down (ordered; see viewport-pointer-pipeline.md) ----

  async function pipelineASquishyAddPointerDown(
    e: React.PointerEvent,
    nx: number,
    ny: number,
    pointerId: number,
    logoSplashPointer: boolean,
  ): Promise<boolean> {
    const mode = interactionModeRef.current;
    if (
      mode !== "squishy" ||
      squishyModeRef.current !== "add" ||
      e.button !== 0 ||
      loading ||
      workBusy ||
      logoSplashPointer
    ) {
      return false;
    }
    try {
      await invoke("squishy_session_set_mode", { args: { mode: "add" } });
      const id = await invoke<number | null>("squishy_metaball_add_at_screen", {
        args: { nx, ny, radius: 4 },
      });
      if (id != null) {
        if (!squishyPhase.active) squishyPhase.enter("settings", {});
        await invoke("squishy_metaball_select", { args: { id } });
        squishyAddDragBallIdRef.current = id;
        probingRef.current = false;
        gestureRef.current = { pointerId, mode: "squishyAddDrag" };
        lastRef.current = { x: e.clientX, y: e.clientY };
        return true;
      }
    } catch {
      /* fall through to camera */
    }
    return false;
  }

  async function pipelineASquishyEditGizmoPointerDown(
    e: React.PointerEvent,
    nx: number,
    ny: number,
    pointerId: number,
    logoSplashPointer: boolean,
  ): Promise<boolean> {
    const mode = interactionModeRef.current;
    if (
      mode !== "squishy" ||
      squishyModeRef.current !== "edit" ||
      e.button !== 0 ||
      loading ||
      workBusy ||
      logoSplashPointer
    ) {
      return false;
    }
    try {
      const consumed = await invoke<boolean>("squishy_gizmo_pointer_down", {
        args: { nx, ny },
      });
      if (consumed) {
        probingRef.current = false;
        gestureRef.current = { pointerId, mode: "squishyGizmo" };
        lastRef.current = { x: e.clientX, y: e.clientY };
        return true;
      }
    } catch {
      /* fall through to pick / camera */
    }
    return false;
  }

  async function pipelineABoneGizmoPointerDown(
    e: React.PointerEvent,
    nx: number,
    ny: number,
    pointerId: number,
    logoSplashPointer: boolean,
  ): Promise<boolean> {
    const mode = interactionModeRef.current;
    if (
      mode !== "bone" ||
      boneModeRef.current !== "edit" ||
      bonePhase.ref.current?.phase !== "pose" ||
      e.button !== 0 ||
      loading ||
      workBusy ||
      logoSplashPointer
    ) {
      return false;
    }
    try {
      const consumed = await invoke<boolean>("bone_gizmo_pointer_down", {
        args: { nx, ny },
      });
      if (consumed) {
        probingRef.current = false;
        gestureRef.current = { pointerId, mode: "boneGizmo" };
        lastRef.current = { x: e.clientX, y: e.clientY };
        return true;
      }
    } catch {
      /* fall through */
    }
    return false;
  }

  async function pipelineAExtrudeGizmoPointerDown(
    e: React.PointerEvent,
    pointerId: number,
    logoSplashPointer: boolean,
    navigate: boolean,
    forceCamera: boolean,
  ): Promise<boolean> {
    const mode = interactionModeRef.current;
    if (
      e.button !== 0 ||
      loading ||
      workBusy ||
      logoSplashPointer ||
      navigate ||
      forceCamera ||
      mode !== "selectExtrude"
    ) {
      return false;
    }
    try {
      const hit = await extrudeGizmoRef.current?.startDragIfHit(e.clientX, e.clientY);
      if (hit) {
        probingRef.current = false;
        gestureRef.current = { pointerId, mode: "extrudeGizmo" };
        lastRef.current = { x: e.clientX, y: e.clientY };
        return true;
      }
    } catch {
      /* fall through */
    }
    if (extrudePhase.ref.current) {
      extrudePhase.cancel();
    }
    return false;
  }

  async function pipelineASelectionGizmoPointerDown(
    e: React.PointerEvent,
    pointerId: number,
    logoSplashPointer: boolean,
    navigate: boolean,
    forceCamera: boolean,
  ): Promise<boolean> {
    const mode = interactionModeRef.current;
    if (
      e.button !== 0 ||
      loading ||
      workBusy ||
      logoSplashPointer ||
      navigate ||
      forceCamera ||
      mode === "selectExtrude"
    ) {
      return false;
    }
    try {
      const hit = await gizmoRef.current?.startDragIfHit(e.clientX, e.clientY);
      if (hit) {
        probingRef.current = false;
        gestureRef.current = { pointerId, mode: "selectionGizmo" };
        lastRef.current = { x: e.clientX, y: e.clientY };
        return true;
      }
    } catch {
      /* fall through */
    }
    return false;
  }

  /** Pipeline A gesture moves: returns true if this event was handled here. */
  function dispatchPipelineAPointerMove(e: React.PointerEvent, px: number, py: number): boolean {
    if (gestureRef.current?.mode === "boneGizmo" && gestureRef.current.pointerId === e.pointerId) {
      void invoke("bone_gizmo_pointer_move", {
        args: { nx: px, ny: py },
      }).catch(() => {});
      return true;
    }
    if (
      gestureRef.current?.mode === "boneAddDrag" &&
      gestureRef.current.pointerId === e.pointerId &&
      boneAddDragJointRef.current != null
    ) {
      void invoke("bone_add_drag_resize", {
        args: { jointId: boneAddDragJointRef.current, nx: px, ny: py },
      }).catch(() => {});
      return true;
    }
    if (
      gestureRef.current?.mode === "squishyGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      void invoke("squishy_gizmo_pointer_move", {
        args: { nx: px, ny: py },
      }).catch(() => {});
      return true;
    }
    if (
      gestureRef.current?.mode === "squishyAddDrag" &&
      gestureRef.current.pointerId === e.pointerId &&
      squishyAddDragBallIdRef.current != null
    ) {
      void invoke("squishy_add_drag_resize", {
        args: { ballId: squishyAddDragBallIdRef.current, nx: px, ny: py },
      }).catch(() => {});
      return true;
    }
    if (
      gestureRef.current?.mode === "selectionGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      const last = lastRef.current;
      const cx = e.clientX;
      const cy = e.clientY;
      gizmoRef.current?.pointerMove(cx, cy, last.x, last.y);
      lastRef.current = { x: cx, y: cy };
      return true;
    }
    if (
      gestureRef.current?.mode === "extrudeGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      const last = lastRef.current;
      const cx = e.clientX;
      const cy = e.clientY;
      extrudeGizmoRef.current?.pointerMove(
        cx,
        cy,
        last.x,
        last.y,
        activeColorRef.current,
        activeMaterialRef.current,
      );
      dragDidEditRef.current = true;
      lastRef.current = { x: cx, y: cy };
      return true;
    }
    return false;
  }

  /** Pipeline A pointer-up handlers that run before the main voxel/up path. Returns true if handled. */
  function dispatchPipelineAPointerUpEarly(e: React.PointerEvent): boolean {
    if (gestureRef.current?.mode === "boneGizmo" && gestureRef.current.pointerId === e.pointerId) {
      void invoke("bone_gizmo_pointer_up").catch(() => {});
      resetPointerGesture("bone-gizmo-up", e);
      return true;
    }

    if (
      gestureRef.current?.mode === "boneAddDrag" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      const placedId = boneAddDragJointRef.current;
      boneAddDragJointRef.current = null;
      if (placedId != null) {
        const prev = bonePendingJointRef.current;
        if (prev != null) {
          void invoke("bone_connect_joints", {
            args: { jointA: prev, jointB: placedId },
          }).catch(() => {});
        }
        bonePendingJointRef.current = placedId;
        void invoke<any>("bone_session_get")
          .then((s: any) => {
            setBoneJointCount(s.joints?.length ?? 0);
            setBoneBoneCount(s.bones?.length ?? 0);
          })
          .catch(() => {});
      }
      resetPointerGesture("bone-add-drag-up", e);
      return true;
    }

    if (
      gestureRef.current?.mode === "squishyAddDrag" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      squishyAddDragBallIdRef.current = null;
      void invoke<{ balls: { id: number }[] }>("squishy_session_get")
        .then((s) => setSquishyBallCount(s.balls?.length ?? 0))
        .catch(() => {});
      resetPointerGesture("squishy-add-drag-up", e);
      return true;
    }

    if (
      gestureRef.current?.mode === "squishyGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      void invoke("squishy_gizmo_pointer_up").catch(() => {});
      resetPointerGesture("squishy-gizmo-up", e);
      return true;
    }

    if (
      gestureRef.current?.mode === "selectionGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      gizmoRef.current?.pointerUp();
      resetPointerGesture("selection-gizmo-up", e);
      return true;
    }
    if (
      gestureRef.current?.mode === "extrudeGizmo" &&
      gestureRef.current.pointerId === e.pointerId
    ) {
      extrudeGizmoRef.current?.pointerUp();
      syncPreviewInput(-1, 0, "selectExtrude");
      if (dragDidEditRef.current) {
        extrudePhase.enter("settings", {} as Record<string, never>);
        lastStrokeNormRef.current = null;
      } else {
        void invoke("voxel_stroke_preview_reset").catch(() => {});
        lastStrokeNormRef.current = null;
      }
      resetPointerGesture("extrude-gizmo-up", e);
      return true;
    }
    return false;
  }

  // ---- Event handlers (verbatim from App.tsx) ----

  const onPointerDown = async (e: React.PointerEvent) => {
    logPlaneStrokeDebug("down:received", e);
    const modeEarly = interactionModeRef.current;
    if ((modeEarly === "fly" || modeEarly === "walk") && (e.button === 0 || e.button === 2)) {
      e.preventDefault();
      if (flyMouseLookActiveRef.current) {
        void releaseFlyMouseLook();
      } else {
        void activateFlyMouseLook(e.pointerId);
      }
      probingRef.current = false;
      resetPointerGesture("fly-toggle", e);
      return;
    }

    const captureEl = e.currentTarget as HTMLElement;
    try {
      captureEl.setPointerCapture(e.pointerId);
      capturedPointerIdRef.current = e.pointerId;
    } catch (err) {
      capturedPointerIdRef.current = null;
      console.warn("[voxelle][plane-stroke] setPointerCapture failed", err);
    }
    activePointerIdRef.current = e.pointerId;
    pointerStartRef.current = { x: e.clientX, y: e.clientY };
    maxPointerMoveRef.current = 0;
    probingRef.current = true;
    gestureRef.current = null;

    // Selection extrude settings phase: re-drag repositions the endpoint.
    if (
      extrudePhase.ref.current &&
      interactionModeRef.current === "selectExtrude" &&
      e.button === 0
    ) {
      extrudeRedragRef.current = true;
      probingRef.current = false;
      return;
    }
    // Keep sculpt/select extrude settings alive on click so the gizmo can be used
    // and drag-orbit doesn't discard the current endpoint.
    if (
      extrudePhase.ref.current &&
      e.button === 0 &&
      interactionModeRef.current !== "selectExtrude" &&
      !(interactionModeRef.current === "sculpt" && sculptStrokeModeRef.current === "extrude")
    ) {
      extrudePhase.cancel();
    }

    const { nx, ny } = clientToViewportNormalized(e);
    const pointerId = e.pointerId;
    const shiftKey = e.shiftKey;
    const middleButton = e.button === 1;
    const mode = interactionModeRef.current;
    const navigate = mode === "navigate" || mode === "fly" || mode === "walk";
    const forceCamera =
      middleButton ||
      (mode === "add" && e.button !== 0) ||
      (mode === "remove" && e.button !== 0) ||
      (mode === "paint" && e.button !== 0) ||
      (mode === "eyedropper" && e.button !== 0) ||
      (mode === "select" && e.button !== 0) ||
      (mode === "selectByColor" && e.button !== 0) ||
      (mode === "selectCoplanar" && e.button !== 0) ||
      (mode === "selectCoplanarEmpty" && e.button !== 0) ||
      (mode === "stamp" && e.button !== 0 && e.button !== 2) ||
      (mode === "punch" && e.button !== 0 && e.button !== 2) ||
      (mode === "sculpt" && e.button !== 0) ||
      (mode === "generator" && e.button !== 0 && e.button !== 2) ||
      (mode === "squishy" && e.button !== 0) ||
      (mode === "bone" && e.button !== 0);

    const logoSplashPointer =
      startScreenLogoLoadedRef.current &&
      !loading &&
      !workBusy &&
      e.button === 0 &&
      mode !== "bone";

    if (await pipelineASquishyAddPointerDown(e, nx, ny, pointerId, logoSplashPointer)) return;
    if (await pipelineASquishyEditGizmoPointerDown(e, nx, ny, pointerId, logoSplashPointer)) return;
    if (await pipelineABoneGizmoPointerDown(e, nx, ny, pointerId, logoSplashPointer)) return;
    if (
      await pipelineAExtrudeGizmoPointerDown(e, pointerId, logoSplashPointer, navigate, forceCamera)
    )
      return;
    if (
      await pipelineASelectionGizmoPointerDown(
        e,
        pointerId,
        logoSplashPointer,
        navigate,
        forceCamera,
      )
    )
      return;

    // Stamp/punch right-click is handled in onContextMenu (fires reliably on all platforms).
    // Still return early here to avoid setting a camera gesture if pointerdown does fire.
    if ((mode === "stamp" || mode === "punch") && e.button === 2) {
      probingRef.current = false;
      resetPointerGesture("stamp-rotate-passthrough", e);
      return;
    }

    // Generator right-click: reseed preview (web parity)

    if (mode === "generator" && e.button === 2 && !loading && !workBusy) {
      e.preventDefault();
      rockPreviewSeedRef.current = (Math.random() * 1e9) | 0;
      ashlarPreviewSeedRef.current = (Math.random() * 1e9) | 0;
      floraPreviewSeedRef.current = (Math.random() * 1e9) | 0;
      // Trigger a preview sync so the new seed is sent to Rust
      const p = lastViewportPickNormRef.current ?? { nx: 0, ny: 0 };
      syncPreviewInput(p.nx, p.ny, previewModeForSync(mode));
      probingRef.current = false;
      resetPointerGesture("generator-reseed", e);
      return;
    }

    let hitSolid = false;
    const isDrawOrSelect =
      !logoSplashPointer &&
      !loading &&
      !workBusy &&
      !forceCamera &&
      !navigate &&
      (mode === "add" ||
        mode === "remove" ||
        mode === "paint" ||
        mode === "eyedropper" ||
        mode === "select" ||
        mode === "selectByColor" ||
        mode === "selectCoplanar" ||
        mode === "selectCoplanarEmpty" ||
        mode === "stamp" ||
        mode === "punch" ||
        mode === "sculpt" ||
        mode === "generator" ||
        mode === "squishy" ||
        mode === "bone") &&
      e.button === 0;
    // All draw/select modes run the async pick probe. Pointer-up events that
    // arrive while probing are deferred and replayed after the probe resolves
    // (see pendingPointerUpRef), which fixes both the "click twice" race and
    // the "can't orbit on empty space" bug for all tools including fill and
    // selection modes.
    if (isDrawOrSelect) {
      try {
        hitSolid = await invoke<boolean>("voxel_pick_probe", {
          args: { nx, ny },
        });
      } catch {
        hitSolid = false;
      }
    }

    probingRef.current = false;

    if (activePointerIdRef.current !== pointerId) {
      return;
    }

    let gestureMode: string = forceCamera || navigate || !hitSolid ? "camera" : "voxel";

    // Bone pose phase: always allow camera orbit on drag. Selection/delete
    // happens on pointer-up with a click threshold, so we force camera gesture
    // here and handle bone interactions in the pointer-up path.
    if (
      mode === "bone" &&
      bonePhase.ref.current?.phase === "pose" &&
      boneModeRef.current !== "add"
    ) {
      gestureMode = "camera";
    }
    if (
      mode === "sculpt" &&
      sculptStrokeModeRef.current === "extrude" &&
      extrudePhase.ref.current
    ) {
      gestureMode = "camera";
    }

    // Bone build+add: place joint immediately on pointer-down and drag to size it.
    if (
      mode === "bone" &&
      boneModeRef.current === "add" &&
      (!bonePhase.ref.current || bonePhase.ref.current.phase === "build") &&
      hitSolid &&
      e.button === 0 &&
      !forceCamera &&
      !navigate &&
      !loading &&
      !workBusy
    ) {
      if (!bonePhase.active) {
        bonePhase.enter("build", {});
      }
      try {
        const newId = await invoke<number | null>("bone_add_joint_at_screen", {
          args: { nx, ny, radius: boneDefaultRadiusRef.current },
        });
        if (newId != null) {
          gestureMode = "boneAddDrag";
          boneAddDragJointRef.current = newId;
        }
      } catch {
        /* fall through to camera */
      }
    }

    gestureRef.current = {
      pointerId,
      mode: gestureMode,
    };
    if (gestureMode === "camera") {
      // Session-style phased previews keep their locked placement visible during orbit.
      const bonePoseOrbit =
        interactionModeRef.current === "bone" && bonePhase.ref.current?.phase === "pose";
      const squishyOrbit = interactionModeRef.current === "squishy";
      if (bonePoseOrbit || squishyOrbit || generatorPreviewPhaseActive()) {
        // Just invalidate the cache so it recomputes fresh on the first hover
        // move after orbit ends, but don't send "navigate" to clear the preview.
        lastSyncPreviewNxRef.current = NaN;
        lastSyncPreviewNyRef.current = NaN;
        lastSyncPreviewModeRef.current = "";
      } else {
        // Clear preview immediately when orbit starts; invalidate cache so it
        // recomputes fresh on the first hover move after orbit ends.
        lastSyncPreviewNxRef.current = NaN;
        lastSyncPreviewNyRef.current = NaN;
        lastSyncPreviewModeRef.current = "";
        syncPreviewInput(-1, 0, "navigate");
      }
    }
    logPlaneStrokeDebug("down:gesture-assigned", e, {
      hitSolid,
      forceCamera,
      navigate,
      assignedGestureMode: gestureRef.current.mode,
    });
    lastRef.current = { x: e.clientX, y: e.clientY };

    if (gestureRef.current.mode === "voxel") {
      const dispatch = getStrokeDispatch(mode);
      const isSculptOrDispatch = dispatch || mode === "sculpt";
      if (isSculptOrDispatch) {
        if (drawStrokeModeRef.current === "cuboid" && cuboidPhase.ref.current) {
          cuboidPhase.cancel();
        }
        if (drawStrokeModeRef.current === "cylinder" && cylinderPhase.ref.current) {
          cylinderPhase.cancel();
        }
        dragDidEditRef.current = false;
        strokeViewportStartRef.current = { nx, ny };
        lastStrokeNormRef.current = { nx, ny };
        currentStrokeSeedRef.current = Math.floor(Math.random() * 0xffffffff) >>> 0;
        strokeShiftKeyRef.current = shiftKey;
        if (dispatch?.kind === "selection") {
          selectionStrokeBegunRef.current = true;
          void invoke("selection_stroke_begin").catch(() => {});
        } else {
          void invoke("voxel_stroke_begin").catch(() => {});
          // Immediately refresh the preview so the correct stroke seed and
          // line-start are reflected on click without requiring mouse movement.
          if (
            mode === "sculpt" &&
            sculptStrokeModeRef.current !== "extrude" &&
            !loading &&
            !workBusy
          ) {
            void invoke("voxel_sculpt_stroke_preview_at_screen", {
              args: buildSculptStrokeInvokeArgs(nx, ny, {
                strokeSegmentPrev: { nx, ny },
              }),
            }).catch(() => {});
          }
        }
      }
      // Roof square/circle: resolve first anchor for drag-to-define.
      if (
        mode === "generator" &&
        generatorKindRef.current === "roof" &&
        (roofAreaShapeRef.current === "square" || roofAreaShapeRef.current === "circle")
      ) {
        dragDidEditRef.current = false;
        void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
          args: {
            nx,
            ny,
            tool: "add",
            strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
          },
        })
          .then((c) => {
            if (c) {
              roofFirstClickRef.current = c;
              setRoofFirstClick(c);
            }
          })
          .catch(() => {});
      }
    }

    if (gestureRef.current.mode === "camera" && mode !== "fly") {
      void invoke("viewport_pointer", {
        ev: {
          kind: "down",
          nx,
          ny,
          dx: 0,
          dy: 0,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey,
        },
      });
    }

    // If pointer-up arrived while the probe was in-flight, replay it now that
    // the gesture is fully established.
    const pendingUp = pendingPointerUpRef.current;
    if (pendingUp && pendingUp.pointerId === pointerId) {
      pendingPointerUpRef.current = null;
      onPointerUpRef.current?.(pendingUp);
    }
  };

  /** Shared sculpt stroke payload for preview and apply (matches Rust `SculptStrokeAtScreenArgs`). */
  function buildSculptStrokeInvokeArgs(
    nx: number,
    ny: number,
    opts: {
      strokeSegmentPrev?: { nx: number; ny: number } | null;
      includeStrokeSeed?: boolean;
    } = {},
  ) {
    const sm = sculptStrokeModeRef.current;
    const includeStrokeSeed = opts.includeStrokeSeed !== false;
    const lineStart =
      sm === "wall" &&
      (wallAreaShapeRef.current === "circle" || wallAreaShapeRef.current === "brush") &&
      strokeViewportStartRef.current
        ? {
            strokeLineStartNx: strokeViewportStartRef.current.nx,
            strokeLineStartNy: strokeViewportStartRef.current.ny,
          }
        : {};
    const wallPoly =
      sm === "wall" &&
      wallAreaShapeRef.current === "polygon" &&
      wallSculptPolygonVertsRef.current.length >= 2
        ? {
            wallPolygonVertices: wallSculptPolygonVertsRef.current.map((v: any) => [
              v[0],
              v[1],
              v[2],
            ]),
          }
        : {};
    const seg = opts.strokeSegmentPrev
      ? {
          strokeSegmentPrevNx: opts.strokeSegmentPrev.nx,
          strokeSegmentPrevNy: opts.strokeSegmentPrev.ny,
        }
      : {};
    return {
      nx,
      ny,
      sculptMode: sm,
      color: activeColorRef.current,
      material: activeMaterialRef.current,
      brushRadius: sculptBrushRadiusRef.current,
      brushShape: sculptBrushShapeToRust(sculptBrushShapeUiRef.current),
      sprayDensity: 0,
      brushClipBottomHalf: brushClipBottomHalfRef.current,
      ...seg,
      ...(sm === "terrain"
        ? {
            terrainOp: terrainSculptOpRef.current,
            terrainBaseY: terrainBaseYRef.current,
            terrainStrength: sculptBrushStrengthRef.current,
            terrainSmoothRadius: terrainSmoothRadiusRef.current,
            terrainFlattenUseBaseY: terrainFlattenUseBaseYRef.current,
            terrainSubVoxel: terrainSubVoxelRef.current,
          }
        : {}),
      ...(sm === "smooth"
        ? {
            smoothNeighborPasses: sculptSmoothPassesRef.current,
          }
        : {}),
      brushStrength: sculptBrushStrengthRef.current,
      brushFalloff: sculptBrushFalloffRef.current,
      ...(includeStrokeSeed
        ? {
            strokeSeed: Math.floor(Math.random() * 0x1_0000_0000) >>> 0,
          }
        : {}),
      wallAreaShape: wallAreaShapeRef.current,
      sprayDirection: sprayDirectionRef.current,
      wallWidthIndex: wallWidthIndexRef.current,
      wallHeightVox: wallHeightVoxRef.current,
      wallLockStartHeight: wallLockStartHeightRef.current,
      wallAxisAlign: wallAxisAlignRef.current,
      sculptSmoothVariant: sculptSmoothVariantRef.current,
      smoothNeighborRadius: smoothNeighborRadiusRef.current,
      smoothAggressiveness: smoothAggressivenessRef.current,
      smoothLaplacianIterations: smoothLaplacianIterationsRef.current,
      smoothLaplacianRelaxPct: smoothLaplacianRelaxPctRef.current,
      extrudeProfile: extrudeProfileRef.current,
      extrudeEndCap: extrudeEndCapRef.current,
      extrudeTaper: extrudeTaperRef.current,
      extrudeTaperStart: extrudeTaperRef.current ? extrudeTaperStartRef.current : 0,
      extrudeTaperEnd: extrudeTaperRef.current ? extrudeTaperEndRef.current : 0,
      // For Draw mode, lock the face normal to the initial click so dragging over
      // different faces doesn't change the placement direction mid-stroke.
      ...(sm === "draw" && strokeViewportStartRef.current
        ? {
            drawNormalNx: strokeViewportStartRef.current.nx,
            drawNormalNy: strokeViewportStartRef.current.ny,
          }
        : {}),
      ...lineStart,
      ...wallPoly,
    };
  }

  function commitWallSculptPolygonStroke() {
    const verts = wallSculptPolygonVertsRef.current;
    if (verts.length < 2) return;
    const scr = lastViewportPickNormRef.current ?? { nx: 0, ny: 0 };
    void invoke("voxel_sculpt_stroke_at_screen", {
      args: buildSculptStrokeInvokeArgs(scr.nx, scr.ny, {
        includeStrokeSeed: true,
      }),
    }).catch(() => {});
    setWallSculptPolygonVerts([]);
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const { nx: px, ny: py } = clientToViewportNormalized(e);
    lastViewportPickNormRef.current = { nx: px, ny: py };
    lastCursorScreenRef.current = { x: e.clientX, y: e.clientY };
    if (viewportCursorDebugEnabled) {
      const el = viewportRef.current;
      const rect = el?.getBoundingClientRect();
      const lv = layoutViewportCssSize();
      const scr: ViewportCursorDebugScreen | null = rect
        ? {
            clientX: e.clientX,
            clientY: e.clientY,
            relX: e.clientX - rect.left,
            relY: e.clientY - rect.top,
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight,
            layoutWidth: lv.w,
            layoutHeight: lv.h,
            rectLeft: rect.left,
            rectTop: rect.top,
            rectWidth: rect.width,
            rectHeight: rect.height,
          }
        : null;
      viewportCursorDebugScreenRef.current = scr;
      setViewportCursorDebugScreen(scr);
      setViewportCursorDebugJs({ nx: px, ny: py });
      if (viewportCursorDebugRafRef.current == null) {
        viewportCursorDebugRafRef.current = requestAnimationFrame(() => {
          viewportCursorDebugRafRef.current = null;
          void invoke<ViewportCursorDebugPayload>("get_viewport_cursor_debug")
            .then((d) => {
              setViewportCursorDebugRust(d);
              // #region agent log
              const vel = viewportRef.current;
              const wrap = vel?.parentElement;
              const rV = vel?.getBoundingClientRect();
              const rW = wrap?.getBoundingClientRect();
              const phys = viewportPhysRef.current;
              const surf = surfacePhysRef.current;
              const scrSnap = viewportCursorDebugScreenRef.current;
              const iw = scrSnap?.layoutWidth ?? layoutViewportCssSize().w;
              const ih = scrSnap?.layoutHeight ?? layoutViewportCssSize().h;
              fetch("http://127.0.0.1:7756/ingest/93734617-b27b-4379-bb59-e5971936c3d4", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  "X-Debug-Session-Id": "0e537f",
                },
                body: JSON.stringify({
                  sessionId: "0e537f",
                  runId: "rect-mapping",
                  hypothesisId: "H_innerY_vs_relY",
                  location: "App.tsx:viewportDebugRaf",
                  message: "pick uses rel/rect; compare inner*surface−origin Y vs relY/rh",
                  data: (() => {
                    const scr = viewportCursorDebugScreenRef.current;
                    const pick = lastViewportPickNormRef.current;
                    let nxFromRel: number | null = null;
                    let nxWindow: number | null = null;
                    let deltaWinVsRel: number | null = null;
                    let nyFromRel: number | null = null;
                    let nyFromInner: number | null = null;
                    let deltaInnerVsRelNy: number | null = null;
                    if (
                      scr &&
                      rV &&
                      rV.width > 0 &&
                      rV.height > 0 &&
                      phys.w > 0 &&
                      phys.h > 0 &&
                      iw > 0 &&
                      ih > 0 &&
                      surf.w > 0 &&
                      surf.h > 0
                    ) {
                      nxFromRel = scr.relX / rV.width;
                      nyFromRel = scr.relY / rV.height;
                      const ox = Math.max(0, Math.round((rV.left / iw) * surf.w));
                      const oy = Math.max(0, Math.round((rV.top / ih) * surf.h));
                      nxWindow = ((scr.clientX / iw) * surf.w - ox) / phys.w;
                      deltaWinVsRel = nxWindow - nxFromRel;
                      nyFromInner = ((scr.clientY / ih) * surf.h - oy) / phys.h;
                      deltaInnerVsRelNy = nyFromInner - nyFromRel;
                    }
                    return {
                      viewportRw: rV?.width,
                      viewportRh: rV?.height,
                      wrapRw: rW?.width,
                      wrapRh: rW?.height,
                      rectDeltaW: rV && rW ? rV.width - rW.width : null,
                      rectDeltaH: rV && rW ? rV.height - rW.height : null,
                      aspectDom: rV && rV.height > 0 ? rV.width / rV.height : null,
                      physW: phys.w,
                      physH: phys.h,
                      aspectPhys: phys.h > 0 ? phys.w / phys.h : null,
                      rustW: d.viewportWidth,
                      rustH: d.viewportHeight,
                      aspectRust: d.viewportHeight > 0 ? d.viewportWidth / d.viewportHeight : null,
                      surfaceW: surf.w,
                      surfaceH: surf.h,
                      vwPerRw: rV && rV.width > 0 ? phys.w / rV.width : null,
                      swPerIw: iw > 0 ? surf.w / iw : null,
                      shPerIh: ih > 0 ? surf.h / ih : null,
                      nxFromRel,
                      nxWindow,
                      deltaWinVsRel,
                      nyFromRel,
                      nyFromInner,
                      deltaInnerVsRelNy,
                      nxPick: pick?.nx ?? null,
                      nyPick: pick?.ny ?? null,
                      deltaPickVsRelNx: pick && nxFromRel != null ? pick.nx - nxFromRel : null,
                      deltaPickVsRelNy: pick && nyFromRel != null ? pick.ny - nyFromRel : null,
                    };
                  })(),
                  timestamp: Date.now(),
                }),
              }).catch(() => {});
              // #endregion
            })
            .catch(() => setViewportCursorDebugRust(null));
        });
      }
    }
    if (dispatchPipelineAPointerMove(e, px, py)) return;
    if (!gestureRef.current) {
      if (
        interactionModeRef.current === "bone" &&
        boneModeRef.current === "edit" &&
        bonePhase.ref.current?.phase === "pose"
      ) {
        void invoke<boolean>("bone_gizmo_hit_test", {
          args: { nx: px, ny: py },
        })
          .then((h) => {
            gizmoHoverRef.current = h ?? false;
          })
          .catch(() => {
            gizmoHoverRef.current = false;
          });
      } else if (interactionModeRef.current === "selectExtrude") {
        extrudeGizmoRef.current
          ?.updateHover(e.clientX, e.clientY)
          .then((h: any) => {
            gizmoHoverRef.current = h ?? false;
          })
          .catch(() => {
            gizmoHoverRef.current = false;
          });
      } else {
        gizmoRef.current
          ?.updateHover(e.clientX, e.clientY)
          .then((h: any) => {
            gizmoHoverRef.current = h ?? false;
          })
          .catch(() => {
            gizmoHoverRef.current = false;
          });
      }
    } else {
      gizmoHoverRef.current = false;
    }
    const overGizmo = !gestureRef.current && gizmoHoverRef.current;
    const anyGenPhaseActive =
      rocksPhase.active ||
      grassPhase.active ||
      ashlarPhase.active ||
      floraPhase.active ||
      shapePhase.active ||
      bonePhase.active ||
      extrudePhase.active;
    const sceneBehavior = getViewportSceneBehavior(interactionModeRef.current);
    if (
      (!overGizmo || interactionModeRef.current === "squishy") &&
      gestureRef.current?.mode !== "camera" &&
      !probingRef.current &&
      sceneBehavior.allowsIdleHoverPreviewSync &&
      !interactionBlockedRef.current &&
      !anyGenPhaseActive
    ) {
      const m = previewModeForSync(interactionModeRef.current);
      if (
        px !== lastSyncPreviewNxRef.current ||
        py !== lastSyncPreviewNyRef.current ||
        m !== lastSyncPreviewModeRef.current
      ) {
        lastSyncPreviewNxRef.current = px;
        lastSyncPreviewNyRef.current = py;
        lastSyncPreviewModeRef.current = m;
        syncPreviewInput(px, py, m);
      }
    } else if (
      overGizmo &&
      !shapePhase.active &&
      !bonePhase.active &&
      !extrudePhase.active &&
      interactionModeRef.current !== "squishy"
    ) {
      // Preserve selectExtrude mode when hovering the extrude gizmo so the
      // GPU gizmo continues rendering in extrude style (balls, no rings).
      // Skip when the shape/bone phase is active — the gizmo is part of the
      // workflow, so hovering it should not clear the preview.
      const hoverMode =
        interactionModeRef.current === "selectExtrude" ? "selectExtrude" : "navigate";
      if (lastSyncPreviewNxRef.current !== -1 || lastSyncPreviewModeRef.current !== hoverMode) {
        lastSyncPreviewNxRef.current = -1;
        lastSyncPreviewNyRef.current = 0;
        lastSyncPreviewModeRef.current = hoverMode;
        syncPreviewInput(-1, 0, hoverMode);
      }
    }

    // Wall brush hover preview: show the full wall footprint under the cursor before any drag.
    // Pass strokeLineStart = current position so Rust uses a zero-length line anchor (single
    // surface voxel) and treats the union as non-accumulating, replacing each frame.
    // strokeSeed is fixed so the stochastic filter is stable and doesn't flicker on hover.
    if (
      !overGizmo &&
      e.buttons === 0 &&
      !probingRef.current &&
      !interactionBlockedRef.current &&
      !loading &&
      !workBusy &&
      interactionModeRef.current === "sculpt" &&
      sculptStrokeModeRef.current === "wall" &&
      wallAreaShapeRef.current === "brush"
    ) {
      const now = Date.now();
      if (now - lastWallHoverMsRef.current >= 40) {
        lastWallHoverMsRef.current = now;
        void invoke("voxel_sculpt_stroke_preview_at_screen", {
          args: {
            ...buildSculptStrokeInvokeArgs(px, py, {
              includeStrokeSeed: false,
            }),
            strokeLineStartNx: px,
            strokeLineStartNy: py,
            strokeSeed: 0,
          },
        }).catch(() => {});
      }
    }

    // Terrain hover: show surface Y under cursor when not stroking.
    if (
      e.buttons === 0 &&
      !probingRef.current &&
      !interactionBlockedRef.current &&
      !loading &&
      !workBusy &&
      interactionModeRef.current === "sculpt" &&
      sculptStrokeModeRef.current === "terrain"
    ) {
      const now = Date.now();
      if (now - lastTerrainHoverMsRef.current >= 80) {
        lastTerrainHoverMsRef.current = now;
        void invoke<number | null>("terrain_surface_y_at_screen", {
          nx: px,
          ny: py,
        })
          .then((r) => setTerrainHoverY(r))
          .catch(() => {});
      }
    }

    if (probingRef.current && activePointerIdRef.current === e.pointerId) {
      return;
    }
    // Extrude re-drag: reposition endpoint during settings phase.
    if (extrudeRedragRef.current && e.buttons && pointerStartRef.current) {
      const now = Date.now();
      if (now - lastStrokeEditMsRef.current >= 24) {
        lastStrokeEditMsRef.current = now;
        const startNorm = extrudeStartNormRef.current;
        if (startNorm) {
          const dpr = window.devicePixelRatio || 1;
          const screenDx = (e.clientX - pointerStartRef.current.x) * dpr;
          const screenDy = (pointerStartRef.current.y - e.clientY) * dpr;
          if (interactionModeRef.current === "selectExtrude") {
            void invoke("selection_extrude_preview", {
              args: {
                screenDx,
                screenDy,
                directionRef: "camera",
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch((err) => {
              console.error("[selection_extrude_preview re-drag]", err);
            });
          } else {
            void invoke("extrude_ray_preview", {
              args: {
                startNx: startNorm.nx,
                startNy: startNorm.ny,
                screenDx,
                screenDy,
                directionRef: extrudeDirectionRefRef.current,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
                brushRadius: sculptBrushRadiusRef.current,
                brushShape: sculptBrushShapeToRust(sculptBrushShapeUiRef.current),
                brushStrength: sculptBrushStrengthRef.current,
                brushFalloff: sculptBrushFalloffRef.current,
                strokeSeed: Math.floor(Math.random() * 0x1_0000_0000) >>> 0,
                extrudeProfile: extrudeProfileRef.current,
                extrudeEndCap: extrudeEndCapRef.current,
                extrudeTaper: extrudeTaperRef.current,
                extrudeTaperStart: extrudeTaperRef.current ? extrudeTaperStartRef.current : 0,
                extrudeTaperEnd: extrudeTaperRef.current ? extrudeTaperEndRef.current : 0,
              },
            }).catch((err) => {
              console.error("[extrude_ray_preview re-drag]", err);
            });
          }
        }
      }
      return;
    }
    if (
      gestureRef.current &&
      gestureRef.current.pointerId === e.pointerId &&
      gestureRef.current.mode === "voxel"
    ) {
      if (pointerStartRef.current) {
        const dx = e.clientX - pointerStartRef.current.x;
        const dy = e.clientY - pointerStartRef.current.y;
        maxPointerMoveRef.current = Math.max(maxPointerMoveRef.current, Math.hypot(dx, dy));
      }
      const m = interactionModeRef.current;
      {
        const dispatch = getStrokeDispatch(m);
        if (
          e.buttons &&
          dispatch &&
          !loading &&
          !workBusy &&
          !fillOperationPending &&
          !strokeModeSkipsDrag(drawStrokeModeRef.current) &&
          !(drawStrokeModeRef.current === "cuboid" && cuboidPhase.ref.current) &&
          !(drawStrokeModeRef.current === "cylinder" && cylinderPhase.ref.current)
        ) {
          const now = Date.now();
          if (now - lastStrokeEditMsRef.current >= 24) {
            lastStrokeEditMsRef.current = now;
            dragDidEditRef.current = true;
            const lineStart = strokeViewportLineStartNorm();
            const brushPrev =
              strokeDrawStyleRef.current === "brush" && lastStrokeNormRef.current
                ? lastStrokeNormRef.current
                : null;
            if (dispatch.kind === "edit") {
              const previewPalette = selectedColorsRef.current;
              const previewMultiColor = previewPalette.length > 1;
              void invoke("voxel_stroke_preview_at_screen", {
                args: {
                  nx: px,
                  ny: py,
                  tool: dispatch.tool,
                  color: activeColorRef.current,
                  ...(previewMultiColor
                    ? {
                        palette: previewPalette,
                        paintColorDistrib: paintColorDistribRef.current,
                        strokeSeed: currentStrokeSeedRef.current,
                      }
                    : {}),
                  material: activeMaterialRef.current,
                  brushRadius: brushRadiusRef.current,
                  brushShape: brushShapeRef.current,
                  sprayDensity: sprayDensityRef.current,
                  strokeMode: drawStrokeModeRef.current,
                  planeAxis: planeAxisRef.current,
                  strokeAux: mergedStrokeAux({}),
                  matchMaterial: matchMaterialSelectColorRef.current,
                  mirrorAxes:
                    (mirrorXRef.current ? 1 : 0) |
                    (mirrorYRef.current ? 2 : 0) |
                    (mirrorZRef.current ? 4 : 0),
                  ...(lineStart
                    ? {
                        strokeLineStartNx: lineStart.nx,
                        strokeLineStartNy: lineStart.ny,
                      }
                    : {}),
                  ...(!lineStart && brushPrev
                    ? {
                        strokeSegmentPrevNx: brushPrev.nx,
                        strokeSegmentPrevNy: brushPrev.ny,
                      }
                    : {}),
                },
              })
                .finally(() => {
                  if (strokeDrawStyleRef.current === "brush") {
                    lastStrokeNormRef.current = { nx: px, ny: py };
                  }
                })
                .catch(() => {});
              logPlaneStrokeDebug("move:preview", e, {
                nx: px,
                ny: py,
                tool: dispatch.tool,
                lineStart: lineStart ?? null,
                brushPrev: brushPrev ?? null,
              });
            } else {
              runStrokeAtScreen(px, py, {}, { lineStart, brushPrev });
              if (strokeDrawStyleRef.current === "brush") {
                lastStrokeNormRef.current = { nx: px, ny: py };
              }
            }
          }
        }
      }
      if (e.buttons && m === "sculpt" && !loading && !workBusy && !fillOperationPending) {
        const now = Date.now();
        if (now - lastStrokeEditMsRef.current >= 24) {
          lastStrokeEditMsRef.current = now;
          dragDidEditRef.current = true;
          if (
            sculptStrokeModeRef.current === "extrude" &&
            pointerStartRef.current &&
            (strokeViewportStartRef.current || extrudeRedragRef.current)
          ) {
            // Ray-based extrude: compute screen delta and send to Rust.
            // Use stored start position for re-drags during settings phase.
            const startNorm = extrudeStartNormRef.current ?? strokeViewportStartRef.current;
            if (startNorm) {
              // On first drag, persist the start position for later re-drags.
              if (!extrudeStartNormRef.current && strokeViewportStartRef.current) {
                extrudeStartNormRef.current = {
                  ...strokeViewportStartRef.current,
                };
              }
              const dpr = window.devicePixelRatio || 1;
              const screenDx = (e.clientX - pointerStartRef.current.x) * dpr;
              const screenDy = (pointerStartRef.current.y - e.clientY) * dpr; // screen up = +
              void invoke("extrude_ray_preview", {
                args: {
                  startNx: startNorm.nx,
                  startNy: startNorm.ny,
                  screenDx,
                  screenDy,
                  directionRef: extrudeDirectionRefRef.current,
                  color: activeColorRef.current,
                  material: activeMaterialRef.current,
                  brushRadius: sculptBrushRadiusRef.current,
                  brushShape: sculptBrushShapeToRust(sculptBrushShapeUiRef.current),
                  brushStrength: sculptBrushStrengthRef.current,
                  brushFalloff: sculptBrushFalloffRef.current,
                  strokeSeed: Math.floor(Math.random() * 0x1_0000_0000) >>> 0,
                  extrudeProfile: extrudeProfileRef.current,
                  extrudeEndCap: extrudeEndCapRef.current,
                  extrudeTaper: extrudeTaperRef.current,
                  extrudeTaperStart: extrudeTaperRef.current ? extrudeTaperStartRef.current : 0,
                  extrudeTaperEnd: extrudeTaperRef.current ? extrudeTaperEndRef.current : 0,
                },
              }).catch((err) => {
                console.error("[extrude_ray_preview]", err);
              });
            }
          } else {
            const sculptBrushPrev = lastStrokeNormRef.current;
            void invoke("voxel_sculpt_stroke_preview_at_screen", {
              args: buildSculptStrokeInvokeArgs(px, py, {
                strokeSegmentPrev: sculptBrushPrev,
              }),
            })
              .finally(() => {
                lastStrokeNormRef.current = { nx: px, ny: py };
              })
              .catch(() => {});
          }
        }
      }
      // selectExtrude drag: preview extruded selection along drag direction.
      if (e.buttons && m === "selectExtrude" && pointerStartRef.current && !loading && !workBusy) {
        const now = Date.now();
        if (now - lastStrokeEditMsRef.current >= 24) {
          lastStrokeEditMsRef.current = now;
          dragDidEditRef.current = true;
          const startNorm = extrudeStartNormRef.current ?? strokeViewportStartRef.current;
          if (startNorm) {
            if (!extrudeStartNormRef.current && strokeViewportStartRef.current) {
              extrudeStartNormRef.current = { ...strokeViewportStartRef.current };
            }
            const dpr = window.devicePixelRatio || 1;
            const screenDx = (e.clientX - pointerStartRef.current.x) * dpr;
            const screenDy = (pointerStartRef.current.y - e.clientY) * dpr;
            void invoke("selection_extrude_preview", {
              args: {
                screenDx,
                screenDy,
                directionRef: "camera",
                color: activeColorRef.current,
                material: activeMaterialRef.current,
              },
            }).catch((err) => {
              console.error("[selection_extrude_preview]", err);
            });
          }
        }
      }
      // Roof square/circle drag: compute pins during drag.
      if (
        e.buttons &&
        m === "generator" &&
        generatorKindRef.current === "roof" &&
        roofFirstClickRef.current &&
        !loading &&
        !workBusy
      ) {
        const shape = roofAreaShapeRef.current;
        if (shape === "square" || shape === "circle") {
          const now = Date.now();
          if (now - lastStrokeEditMsRef.current >= 40) {
            lastStrokeEditMsRef.current = now;
            dragDidEditRef.current = true;
            void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
              args: {
                nx: px,
                ny: py,
                tool: "add",
                strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
              },
            })
              .then((c) => {
                if (!c || !roofFirstClickRef.current) return;
                const first = roofFirstClickRef.current;
                let pins: [number, number, number][];
                if (shape === "square") {
                  const [x1, y1, z1] = first;
                  const [x2, , z2] = c;
                  pins = [
                    [x1, y1, z1],
                    [x2, y1, z1],
                    [x2, y1, z2],
                    [x1, y1, z2],
                  ];
                } else {
                  const [cx, cy, cz] = first;
                  const [ex, , ez] = c;
                  const dx = ex - cx;
                  const dz = ez - cz;
                  const r = Math.sqrt(dx * dx + dz * dz);
                  const N = 16;
                  pins = [];
                  for (let i = 0; i < N; i++) {
                    const angle = (2 * Math.PI * i) / N;
                    pins.push([
                      Math.round(cx + r * Math.cos(angle)),
                      cy,
                      Math.round(cz + r * Math.sin(angle)),
                    ]);
                  }
                }
                // Ensure pins wind so the roof grows upward (+Y).
                // In the XZ plane, positive signed area means CW from
                // above, which gives a downward normal — reverse to fix.
                let areaXZ = 0;
                for (let i = 0; i < pins.length; i++) {
                  const p = pins[i];
                  const q = pins[(i + 1) % pins.length];
                  areaXZ += p[0] * q[2] - q[0] * p[2];
                }
                if (areaXZ >= 0) {
                  pins.reverse();
                }
                setRoofPins(pins);
                roofPinsRef.current = pins;
                syncPreviewInput(px, py, previewModeForSync(interactionModeRef.current));
              })
              .catch(() => {});
          }
        }
      }
      return;
    }
    if (pointerStartRef.current) {
      const dx = e.clientX - pointerStartRef.current.x;
      const dy = e.clientY - pointerStartRef.current.y;
      maxPointerMoveRef.current = Math.max(maxPointerMoveRef.current, Math.hypot(dx, dy));
    }
    const dpr = window.devicePixelRatio || 1;
    const dx = (e.clientX - lastRef.current.x) * dpr;
    const dy = (e.clientY - lastRef.current.y) * dpr;
    lastRef.current = { x: e.clientX, y: e.clientY };
    if (e.buttons === 0) {
      logPlaneStrokeDebug("move:buttons-zero", e);
      if (startScreenLogoLoadedRef.current && interactionModeRef.current !== "fly") {
        const { nx, ny } = clientToViewportNormalized(e);
        void invoke("viewport_pointer", {
          ev: {
            kind: "move",
            nx,
            ny,
            dx: 0,
            dy: 0,
            button: e.button,
            buttons: 0,
            shiftKey: e.shiftKey,
          },
        }).catch(() => {});
      }
      return;
    }
    const { nx, ny } = clientToViewportNormalized(e);
    // Don't pan the camera during right-button drag in stamp/punch mode —
    // right-click in those modes rotates the stamp, not the camera.
    const stampPunchRightDrag =
      (interactionModeRef.current === "stamp" || interactionModeRef.current === "punch") &&
      (e.buttons & 2) !== 0 &&
      gestureRef.current?.mode !== "camera";
    if (interactionModeRef.current !== "fly" && !stampPunchRightDrag) {
      void invoke("viewport_pointer", {
        ev: {
          kind: "move",
          nx,
          ny,
          dx,
          dy,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey && gestureRef.current?.mode !== "voxel",
        },
      });
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("up:received", e);
    // Extrude re-drag: stay in settings phase after repositioning endpoint.
    if (extrudeRedragRef.current) {
      extrudeRedragRef.current = false;
      pointerStartRef.current = null;
      return;
    }
    if (probingRef.current && activePointerIdRef.current === e.pointerId) {
      // Probe is still in-flight — defer this up event so onPointerDown can
      // replay it once the gesture is established (see pendingPointerUpRef).
      pendingPointerUpRef.current = e;
      return;
    }

    if (dispatchPipelineAPointerUpEarly(e)) return;

    const g = gestureRef.current;
    const start = pointerStartRef.current;
    const moved = maxPointerMoveRef.current;
    const isThisPointer = g?.pointerId === e.pointerId;
    let hasPointerCaptureForUp = capturedPointerIdRef.current === e.pointerId;
    if (!hasPointerCaptureForUp && viewportRef.current) {
      try {
        hasPointerCaptureForUp = viewportRef.current.hasPointerCapture(e.pointerId);
      } catch {
        hasPointerCaptureForUp = false;
      }
    }

    if (
      isThisPointer &&
      g?.mode === "voxel" &&
      !loading &&
      !workBusy &&
      !fillOperationPending &&
      start &&
      e.button === 0 &&
      hasPointerCaptureForUp
    ) {
      const { nx, ny } = clientToViewportNormalized(e);
      const m = interactionModeRef.current;
      if (moved < 5) {
        if (m === "stamp") {
          void invoke("clipboard_stamp_at_screen", {
            args: {
              nx,
              ny,
              rotX: stampRotXRef.current,
              rotY: stampRotYRef.current,
              rotZ: stampRotZRef.current,
              originX: stampOriginXRef.current,
              originZ: stampOriginZRef.current,
            },
          }).catch(() => {});
        } else if (m === "punch") {
          void invoke("clipboard_punch_at_screen", {
            args: {
              nx,
              ny,
              rotX: stampRotXRef.current,
              rotY: stampRotYRef.current,
              rotZ: stampRotZRef.current,
              originX: stampOriginXRef.current,
              originZ: stampOriginZRef.current,
            },
          }).catch(() => {});
        } else if (m === "generator") {
          const gk = generatorKindRef.current;
          if (gk === "rocks") {
            if (!rocksPhase.active) {
              if (rocksAutoCommitOnMouseUpRef.current) {
                placeRocksAtScreen(nx, ny, rockPreviewSeedRef.current);
              } else {
                void invoke("lock_generator_preview_camera").catch(() => {});
                rocksPhase.enter("settings", { nx, ny, seed: rockPreviewSeedRef.current });
              }
            }
          } else if (gk === "grass") {
            if (!grassPhase.active) {
              if (grassAutoCommitOnMouseUpRef.current) {
                placeGrassAtScreen(nx, ny, grassPreviewSeedRef.current);
              } else {
                void invoke("lock_generator_preview_camera").catch(() => {});
                grassPhase.enter("settings", { nx, ny, seed: grassPreviewSeedRef.current });
              }
            }
          } else if (gk === "cloth") {
            if (!clothPhase.active) {
              void handleClothPinClick(nx, ny);
            }
          } else if (gk === "rope") {
            if (ropePhase.active) {
              // Already in settings phase — ignore clicks
            } else if (!ropeFirstScreen) {
              setRopeFirstScreen({ nx, ny });
              void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
                args: { nx, ny, tool: "add", strokeSnapToSurface: false },
              })
                .then((v) => {
                  if (v) setRopeFirstVoxel(v);
                })
                .catch(() => {});
            } else {
              // Enter settings phase instead of immediately generating
              void invoke("lock_generator_preview_camera").catch(() => {});
              ropePhase.enter("settings", {
                nx1: ropeFirstScreen.nx,
                ny1: ropeFirstScreen.ny,
                nx2: nx,
                ny2: ny,
              });
            }
          } else if (gk === "ashlar") {
            if (!ashlarPhase.active) {
              if (ashlarAutoCommitOnMouseUpRef.current) {
                placeAshlarAtScreen(nx, ny, ashlarPreviewSeedRef.current);
              } else {
                void invoke("lock_generator_preview_camera").catch(() => {});
                ashlarPhase.enter("settings", { nx, ny, seed: ashlarPreviewSeedRef.current });
              }
            }
          } else if (gk === "flora") {
            if (!floraPhase.active) {
              if (floraAutoCommitOnMouseUpRef.current) {
                placeFloraAtScreen(nx, ny, floraPreviewSeedRef.current);
              } else {
                void invoke("lock_generator_preview_camera").catch(() => {});
                floraPhase.enter("settings", { nx, ny, seed: floraPreviewSeedRef.current });
              }
            }
          } else if (gk === "shape") {
            if (!shapePhase.active) {
              void invoke("lock_generator_preview_camera").catch(() => {});
              // Resolve click to world coord for gizmo placement.
              void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
                args: { nx, ny, tool: "add" },
              })
                .then((c) => {
                  if (c) {
                    shapeGizmoPosRef.current = c;
                    void invoke("set_generator_gizmo_center", { center: [c[0], c[1], c[2]] }).catch(
                      () => {},
                    );
                  }
                })
                .catch(() => {});
              shapePhase.enter("settings", { nx, ny });
            }
          } else if (gk === "roof") {
            // Square/circle use drag-to-define (handled in pointer move/up).
            // Polygon still uses click-to-add-pin.
            if (roofAreaShapeRef.current === "polygon") {
              void invoke<[number, number, number] | null>("voxel_stroke_anchor_coord_at_screen", {
                args: {
                  nx,
                  ny,
                  tool: "add",
                  strokeSnapToSurface: selectionStrokeSnapToSurfaceRef.current,
                },
              })
                .then((c) => {
                  if (!c) return;
                  setRoofPins((v: any) => {
                    // Click on existing pin → remove it.
                    const idx = v.findIndex(
                      (p: any) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2],
                    );
                    const next = idx >= 0 ? v.filter((_: any, i: number) => i !== idx) : [...v, c];
                    roofPinsRef.current = next;
                    return next;
                  });
                })
                .catch(() => {});
            }
          }
        } else if (m === "squishy") {
          const mode = squishyModeRef.current;
          // Add is handled on pointer-down (hit voxel → place+drag-resize; miss → orbit).
          if (mode !== "add") {
            if (!squishyPhase.active) squishyPhase.enter("settings", {});
            void invoke("squishy_session_set_mode", { args: { mode } })
              .then(() => invoke<number | null>("squishy_pick_at_screen", { args: { nx, ny } }))
              .then((id) => {
                if (id == null) return;
                if (mode === "delete") {
                  return invoke("squishy_metaball_remove", { args: { id } });
                }
                return invoke("squishy_metaball_select", { args: { id } });
              })
              .then(() => invoke<{ balls: { id: number }[] }>("squishy_session_get"))
              .then((s) => setSquishyBallCount(s.balls?.length ?? 0))
              .catch(() => {});
          }
        }
      }
      if (m === "eyedropper") {
        if (moved < 5) {
          void invoke<{
            color: number;
            material: string;
          } | null>("voxel_pick_color_at_screen", {
            args: {
              nx,
              ny,
              tool: "add",
              color: activeColorRef.current,
              material: activeMaterialRef.current,
              brushRadius: 0,
              brushShape: brushShapeRef.current,
            },
          })
            .then((r) => {
              if (r) {
                setActiveColor(r.color);
                setActiveMaterial(r.material);
                const back = eyedropperReturnModeRef.current;
                if (back != null && back !== "eyedropper") {
                  setInteractionMode(back);
                }
              }
            })
            .catch(() => {});
        }
      } else if (getStrokeDispatch(m)) {
        const dispatch = getStrokeDispatch(m)!;
        const sm = drawStrokeModeRef.current;
        // Depth phase activation after drag (cuboid/cylinder)
        if (
          (sm === "cuboid" || sm === "cylinder") &&
          (dragDidEditRef.current || moved >= 5) &&
          strokeViewportStartRef.current
        ) {
          const phase = sm === "cuboid" ? cuboidPhase : cylinderPhase;
          const depthRef = sm === "cuboid" ? cuboidDepthRef : cylinderDepthRef;
          const setDepthUi = sm === "cuboid" ? setCuboidDepthUi : setCylinderDepthUi;
          depthRef.current = 1;
          setDepthUi(1);
          const lineStart = { ...strokeViewportStartRef.current };
          phase.enter("depth", { lineStart, endNorm: { nx, ny }, frozenGeo: null });
          {
            const im = interactionModeRef.current;
            const tool = im === "remove" ? "remove" : im === "paint" ? "paint" : "add";
            void invoke<CuboidPlaneGeo | null>("query_cuboid_plane_geometry", {
              args: {
                nx,
                ny,
                tool,
                color: activeColorRef.current,
                material: activeMaterialRef.current,
                brushRadius: brushRadiusRef.current,
                brushShape: brushShapeRef.current,
                sprayDensity: 0,
                strokeMode: sm,
                planeAxis: planeAxisRef.current,
                strokeAux: mergedStrokeAux({}),
                matchMaterial: false,
                strokeLineStartNx: lineStart.nx,
                strokeLineStartNy: lineStart.ny,
              },
            })
              .then((geo) => {
                phase.update({ frozenGeo: geo ?? null });
              })
              .catch(() => {});
          }
        }
        // Single-click handling (no drag)
        // For selection strokes, we must wait for the stroke invoke to
        // complete before calling selection_stroke_end, otherwise the
        // end command can race ahead and clear the accumulator.
        let clickStrokePromise: Promise<void> | null = null;
        if (!dragDidEditRef.current && moved < 5) {
          if ((sm === "cuboid" || sm === "cylinder") && strokeViewportStartRef.current) {
            // Single-click cuboid/cylinder: enter depth phase
            const phase = sm === "cuboid" ? cuboidPhase : cylinderPhase;
            const depthRef = sm === "cuboid" ? cuboidDepthRef : cylinderDepthRef;
            const setDepthUi = sm === "cuboid" ? setCuboidDepthUi : setCylinderDepthUi;
            depthRef.current = 1;
            setDepthUi(1);
            const lineStart = { ...strokeViewportStartRef.current };
            phase.enter("depth", { lineStart, endNorm: { nx, ny }, frozenGeo: null });
            {
              const im = interactionModeRef.current;
              const tool = im === "remove" ? "remove" : im === "paint" ? "paint" : "add";
              void invoke<CuboidPlaneGeo | null>("query_cuboid_plane_geometry", {
                args: {
                  nx,
                  ny,
                  tool,
                  color: activeColorRef.current,
                  material: activeMaterialRef.current,
                  brushRadius: brushRadiusRef.current,
                  brushShape: brushShapeRef.current,
                  sprayDensity: 0,
                  strokeMode: sm,
                  planeAxis: planeAxisRef.current,
                  strokeAux: mergedStrokeAux({}),
                  matchMaterial: false,
                  strokeLineStartNx: lineStart.nx,
                  strokeLineStartNy: lineStart.ny,
                },
              })
                .then((geo) => {
                  phase.update({ frozenGeo: geo ?? null });
                })
                .catch(() => {});
            }
          } else if (strokeModeSkipsDrag(sm)) {
            void handleStrokeAnchorClick(nx, ny);
          } else if (
            dispatch.kind === "selection" &&
            dispatch.interaction !== "select" &&
            sm !== "fill"
          ) {
            // Specialized selection click commands (selectByColor, etc.)
            void invokeSelectionSpecialClick(dispatch.interaction, nx, ny);
          } else {
            const lineStart = strokeViewportLineStartNorm();
            clickStrokePromise = runStrokeAtScreen(nx, ny, {}, { lineStart });
          }
        }
        logPlaneStrokeDebug("up:stroke-end", e, {
          moved,
          dragDidEdit: dragDidEditRef.current,
          strokeMode: sm,
          interactionMode: m,
        });
        if (dispatch.kind === "edit") {
          void invoke("voxel_stroke_end").catch(() => {});
        } else {
          const endSelection = () => {
            void invoke("selection_stroke_end").catch(() => {});
            selectionStrokeBegunRef.current = false;
          };
          if (clickStrokePromise) {
            void clickStrokePromise.then(endSelection);
          } else {
            endSelection();
          }
        }
        lastStrokeNormRef.current = null;
      } else if (m === "sculpt") {
        const sm = sculptStrokeModeRef.current;
        if (sm === "extrude" && (dragDidEditRef.current || moved >= 5)) {
          if (sculptExtrudeAutoCommitOnMouseUpRef.current) {
            extrudeStartNormRef.current = null;
            void invoke("voxel_stroke_end").catch(() => {});
          } else {
            // Extrude phased tool: enter settings phase instead of committing.
            // The preview union is already accumulated from the drag. Keep it visible.
            extrudePhase.enter("settings", {} as Record<string, never>);
          }
          lastStrokeNormRef.current = null;
        } else {
          if (!dragDidEditRef.current && moved < 5) {
            const wa = wallAreaShapeRef.current;
            if (sm === "wall" && wa === "polygon") {
              void handleWallSculptPolygonClick(nx, ny);
            }
            // For non-polygon sculpt clicks: the pointer-down preview already pushed to
            // sculpt_stroke_replay and stroke_preview_union. voxel_stroke_end commits
            // from that data. Calling voxel_sculpt_stroke_at_screen here pre-applies
            // voxels, leaving empty deltas at voxel_stroke_end and breaking undo.
          }
          if (sm === "extrude") {
            extrudeStartNormRef.current = null;
          }
          void invoke("voxel_stroke_end").catch(() => {});
          lastStrokeNormRef.current = null;
        }
      } else if (m === "selectExtrude") {
        if (dragDidEditRef.current || moved >= 5) {
          // Enter settings phase — preview union stays visible.
          extrudePhase.enter("settings", {} as Record<string, never>);
          lastStrokeNormRef.current = null;
          // Do NOT call voxel_stroke_end — preview must stay.
        } else {
          void invoke("voxel_stroke_preview_reset").catch(() => {});
          lastStrokeNormRef.current = null;
        }
      } else if (m === "generator" && generatorKindRef.current === "roof") {
        // Roof square/circle drag complete: clear first-click anchor.
        // Pins are already set from the move handler during drag.
        const shape = roofAreaShapeRef.current;
        if (shape === "square" || shape === "circle") {
          roofFirstClickRef.current = null;
          setRoofFirstClick(null);
        }
      }
    } else if (isThisPointer && g?.mode === "voxel" && e.button === 0 && !hasPointerCaptureForUp) {
      logPlaneStrokeDebug("up:ignored-no-capture", e, {
        moved,
      });
    }

    // Bone tool: pick/select/delete on click (works in camera gesture mode so
    // dragging orbits and only short clicks trigger selection).
    if (isThisPointer && interactionModeRef.current === "bone" && moved < 20 && e.button === 0) {
      const { nx, ny } = clientToViewportNormalized(e);
      const phase = bonePhase.ref.current?.phase ?? "build";
      const bm = boneModeRef.current;
      let action: Promise<void> | null = null;
      if (bm === "delete") {
        action = invoke<any>("bone_pick_at_screen", { args: { nx, ny } }).then((sel) => {
          if (sel != null) {
            bonePendingJointRef.current = null;
            return invoke("bone_remove", { args: { selection: sel } }).then(() => {});
          }
        });
      } else if (phase === "pose" && bm === "edit") {
        action = invoke<any>("bone_pick_at_screen", { args: { nx, ny } }).then((sel) =>
          invoke("bone_select", { args: { selection: sel ?? null } }).then(() => {}),
        );
      }
      if (action) {
        void action
          .then(() => invoke<any>("bone_session_get"))
          .then((s: any) => {
            setBoneJointCount(s.joints?.length ?? 0);
            setBoneBoneCount(s.bones?.length ?? 0);
          })
          .catch(() => {});
      }
    }

    if (isThisPointer && g?.mode === "camera" && interactionModeRef.current !== "fly") {
      const { nx, ny } = clientToViewportNormalized(e);
      void invoke("viewport_pointer", {
        ev: {
          kind: "up",
          nx,
          ny,
          dx: 0,
          dy: 0,
          button: e.button,
          buttons: e.buttons,
          shiftKey: e.shiftKey,
        },
      });
    }

    if (isThisPointer) {
      resetPointerGesture("pointer-up-complete", e);
      if (capturedPointerIdRef.current === e.pointerId) {
        capturedPointerIdRef.current = null;
      }
      // Bone pose mode: restore the skeleton preview immediately after a camera
      // orbit ends rather than waiting for the next pointer-move event.
      if (
        g?.mode === "camera" &&
        interactionModeRef.current === "bone" &&
        bonePhase.ref.current?.phase === "pose"
      ) {
        restoreBonePosePreviewAtEvent(e);
      }
    }
  };
  onPointerUpRef.current = onPointerUp;

  const onPointerLeave = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("leave", e);
    // Keep the select preview visible when the pointer moves to the sidebar / tool panel.
    // Also keep phased-tool previews alive — their coordinates are locked to
    // stored data anyway, so clearing on leave just creates a jarring flicker
    // when the user mouses over the settings overlay.
    const im = interactionModeRef.current;
    const anyPhaseActive =
      cuboidPhase.ref.current !== null ||
      cylinderPhase.ref.current !== null ||
      extrudePhase.ref.current !== null ||
      ropePhase.ref.current !== null ||
      clothPhase.ref.current !== null ||
      generatorPreviewPhaseActive();
    if (!anyPhaseActive && !getViewportSceneBehavior(im).preservePreviewOnPointerLeave) {
      clearPreview();
    }
    if (viewportCursorDebugEnabled) {
      setViewportCursorDebugJs(null);
      setViewportCursorDebugRust(null);
      viewportCursorDebugScreenRef.current = null;
      setViewportCursorDebugScreen(null);
    }
    if (startScreenLogoLoadedRef.current && interactionModeRef.current !== "fly") {
      void invoke("viewport_pointer", {
        ev: {
          kind: "leave",
          nx: 0.5,
          ny: 0.5,
          dx: 0,
          dy: 0,
          button: 0,
          buttons: 0,
          shiftKey: false,
        },
      }).catch(() => {});
    }
    // Never synthesize pointer-up from leave. Commit only on real pointer-up/cancel;
    // pointer capture keeps those events routed to the viewport during drags.
  };

  const onGotPointerCapture = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("capture:got", e);
  };

  const onLostPointerCapture = (e: React.PointerEvent) => {
    logPlaneStrokeDebug("capture:lost", e);
    restoreBonePosePreviewAtEvent(e);
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (interactionModeRef.current === "fly" || interactionModeRef.current === "walk") return;
    // Shape generator wheel shortcuts
    if (
      interactionModeRef.current === "generator" &&
      generatorKindRef.current === "shape" &&
      (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey)
    ) {
      const delta = e.deltaY > 0 ? -1 : 1;
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
        // Ctrl+Wheel → size ±1
        setShapeSize((v: number) => Math.max(1, Math.min(256, v + delta)));
      } else if (e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey) {
        // Shift+Wheel → rotX ±15°
        setShapeRotX((v: number) => (((v + delta * 15) % 360) + 360) % 360);
      } else if (e.altKey && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        // Alt+Wheel → rotY ±15°
        setShapeRotY((v: number) => (((v + delta * 15) % 360) + 360) % 360);
      } else if (e.shiftKey && e.altKey && !e.ctrlKey && !e.metaKey) {
        // Shift+Alt+Wheel → rotZ ±15°
        setShapeRotZ((v: number) => (((v + delta * 15) % 360) + 360) % 360);
      }
      return;
    }
    void invoke("viewport_wheel", {
      ev: { delta_x: e.deltaX, delta_y: e.deltaY },
    });
  };

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerLeave,
    onGotPointerCapture,
    onLostPointerCapture,
    onWheel,
    commitWallSculptPolygonStroke,
  };
}
