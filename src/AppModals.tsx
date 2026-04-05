import { invoke } from "@tauri-apps/api/core";
import { CollabJoinProgressModal } from "./CollabJoinProgressModal";
import { JoinSessionModal } from "./JoinSessionModal";
import { PreferencesModal } from "./PreferencesModal";
import { StampBookModal } from "./StampBookModal";
import type { StampBookEntryTuple } from "./stampBookStorage";
import type { StartShape } from "./preferences";
import { MAX_GRID_SIZE } from "./constants";
import type { InteractionMode } from "./types";

interface Props {
  // Leave-confirm dialog
  leaveConfirmOpen: boolean;
  setLeaveConfirmOpen: (open: boolean) => void;
  hostWsUrl: string | null;
  leaveSession: () => void;

  // Join session modal
  joinModalOpen: boolean;
  setJoinModalOpen: (open: boolean) => void;
  joinUrl: string;
  setJoinUrl: (url: string) => void;
  joinSession: (urlOverride?: string) => void;
  collabActive: boolean;
  collabJoinPending: boolean;

  // Collab join progress modal
  loading: boolean;
  loadProgress: number;
  loadPhase: string;
  pathLabel: string;
  cancelJoin: () => void;

  // Stamp book modal
  stampBookOpen: boolean;
  setStampBookOpen: (open: boolean) => void;
  selectionCount: number;
  setStampBookPatternActive: (active: boolean) => void;
  setInteractionMode: (mode: InteractionMode) => void;

  // Pending fill confirm dialog
  pendingFillConfirm: { resolve: (confirmed: boolean) => void } | null;

  // Preferences modal
  preferencesOpen: boolean;
  setPreferencesOpen: (open: boolean) => void;
  setShowFpsCounter: (v: boolean) => void;
  setShowPingLatency: (v: boolean) => void;
  setPrefsEnableUpnp: (v: boolean) => void;
  setDisplayName: (v: string) => void;
  setAccentColor: (v: string) => void;
  setHostPort: (v: number) => void;

  // Rotate dialog
  rotateDialogOpen: boolean;
  setRotateDialogOpen: (open: boolean) => void;
  rotateDialogAxis: 0 | 1 | 2;
  setRotateDialogAxis: (axis: 0 | 1 | 2) => void;
  rotateDialogDegrees: number;
  setRotateDialogDegrees: (degrees: number) => void;

  // Scale dialog
  scaleDialogOpen: boolean;
  setScaleDialogOpen: (open: boolean) => void;
  scaleDialogFactor: number;
  setScaleDialogFactor: (factor: number) => void;

  // New project dialog
  newProjectOpen: boolean;
  setNewProjectOpen: (open: boolean) => void;
  newGridSize: number;
  setNewGridSize: (size: number) => void;
  newGridShape: StartShape;
  setNewGridShape: (shape: StartShape) => void;
  createNewProject: () => void;
}

export function AppModals({
  leaveConfirmOpen,
  setLeaveConfirmOpen,
  hostWsUrl,
  leaveSession,
  joinModalOpen,
  setJoinModalOpen,
  joinUrl,
  setJoinUrl,
  joinSession,
  collabActive,
  collabJoinPending,
  loading,
  loadProgress,
  loadPhase,
  pathLabel,
  cancelJoin,
  stampBookOpen,
  setStampBookOpen,
  selectionCount,
  setStampBookPatternActive,
  setInteractionMode,
  pendingFillConfirm,
  preferencesOpen,
  setPreferencesOpen,
  setShowFpsCounter,
  setShowPingLatency,
  setPrefsEnableUpnp,
  setDisplayName,
  setAccentColor,
  setHostPort,
  rotateDialogOpen,
  setRotateDialogOpen,
  rotateDialogAxis,
  setRotateDialogAxis,
  rotateDialogDegrees,
  setRotateDialogDegrees,
  scaleDialogOpen,
  setScaleDialogOpen,
  scaleDialogFactor,
  setScaleDialogFactor,
  newProjectOpen,
  setNewProjectOpen,
  newGridSize,
  setNewGridSize,
  newGridShape,
  setNewGridShape,
  createNewProject,
}: Props) {
  return (
    <>
      {leaveConfirmOpen && (
        <div className="modal-overlay" onClick={() => setLeaveConfirmOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{hostWsUrl ? "End session?" : "Leave session?"}</h3>
            <p style={{ margin: "0 0 0.75rem", fontSize: "0.875rem" }}>
              {hostWsUrl
                ? "This will end the session for everyone."
                : "You will leave the current session."}
            </p>
            <div className="modal-buttons">
              <button type="button" onClick={() => setLeaveConfirmOpen(false)}>
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setLeaveConfirmOpen(false);
                  leaveSession();
                }}
              >
                {hostWsUrl ? "End session" : "Leave"}
              </button>
            </div>
          </div>
        </div>
      )}
      <JoinSessionModal
        open={joinModalOpen}
        onClose={() => setJoinModalOpen(false)}
        joinUrl={joinUrl}
        onJoinUrlChange={setJoinUrl}
        onJoin={joinSession}
        collabActive={collabActive}
        connecting={collabJoinPending}
      />
      <CollabJoinProgressModal
        open={collabJoinPending}
        loading={loading}
        loadProgress={loadProgress}
        loadPhase={loadPhase}
        pathLabel={pathLabel}
        onCancel={cancelJoin}
      />
      <StampBookModal
        open={stampBookOpen}
        onClose={() => setStampBookOpen(false)}
        selectionCount={selectionCount}
        onUseStamp={(entries: StampBookEntryTuple[]) => {
          void invoke("stamp_book_load_entries", {
            entries: entries.map(([dx, dy, dz, color, mat]) => ({
              dx,
              dy,
              dz,
              color,
              material: mat ?? "plastic",
            })),
          })
            .then(() => {
              void invoke("selection_clear").catch(() => {});
              setStampBookPatternActive(true);
              setInteractionMode("stamp");
            })
            .catch(() => {});
        }}
      />
      {pendingFillConfirm && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onKeyDown={(e) => {
            if (e.key === "Escape") pendingFillConfirm.resolve(false);
          }}
        >
          <div className="modal">
            <h3>Large fill</h3>
            <p style={{ fontSize: "0.85rem", margin: "0 0 0.75rem" }}>
              This fill covers a large area and may take a while. Continue?
            </p>
            <div className="modal-buttons">
              <button onClick={() => pendingFillConfirm.resolve(true)} autoFocus>
                Fill
              </button>
              <button onClick={() => pendingFillConfirm.resolve(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
      <PreferencesModal
        open={preferencesOpen}
        onClose={() => setPreferencesOpen(false)}
        onFpsCounterChange={setShowFpsCounter}
        onPingLatencyChange={setShowPingLatency}
        onEnableUpnpChange={setPrefsEnableUpnp}
        onCollabDisplayNameChange={setDisplayName}
        onCollabAccentColorChange={setAccentColor}
        onCollabHostPortChange={setHostPort}
        collabHosting={hostWsUrl != null}
      />
      {rotateDialogOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onClick={(e) => e.target === e.currentTarget && setRotateDialogOpen(false)}
          onKeyDown={(e) => e.key === "Escape" && setRotateDialogOpen(false)}
        >
          <div className="modal">
            <h3>Rotate selection</h3>
            <label className="modal-field">
              Axis
              <select
                value={rotateDialogAxis}
                onChange={(e) => setRotateDialogAxis(Number(e.target.value) as 0 | 1 | 2)}
              >
                <option value={0}>X</option>
                <option value={1}>Y</option>
                <option value={2}>Z</option>
              </select>
            </label>
            <label className="modal-field">
              Degrees
              <select
                value={rotateDialogDegrees}
                onChange={(e) => setRotateDialogDegrees(Number(e.target.value))}
              >
                <option value={90}>90°</option>
                <option value={180}>180°</option>
                <option value={270}>270°</option>
              </select>
            </label>
            <div className="modal-buttons">
              <button
                type="button"
                onClick={() => {
                  const quarters = rotateDialogDegrees / 90;
                  void invoke("selection_rotate", {
                    axis: rotateDialogAxis,
                    quarters,
                  }).catch(() => {});
                  setRotateDialogOpen(false);
                }}
              >
                Rotate
              </button>
              <button type="button" onClick={() => setRotateDialogOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {scaleDialogOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onClick={(e) => e.target === e.currentTarget && setScaleDialogOpen(false)}
          onKeyDown={(e) => e.key === "Escape" && setScaleDialogOpen(false)}
        >
          <div className="modal">
            <h3>Scale selection</h3>
            <label className="modal-field">
              Factor
              <input
                type="number"
                min={0.1}
                max={8}
                step={0.25}
                value={scaleDialogFactor}
                onChange={(e) =>
                  setScaleDialogFactor(Math.max(0.1, Math.min(8, Number(e.target.value))))
                }
              />
            </label>
            <div className="modal-buttons">
              <button
                type="button"
                onClick={() => {
                  void invoke("selection_scale", {
                    factor: scaleDialogFactor,
                  }).catch(() => {});
                  setScaleDialogOpen(false);
                }}
              >
                Scale
              </button>
              <button type="button" onClick={() => setScaleDialogOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {newProjectOpen ? (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
          onClick={(e) => e.target === e.currentTarget && setNewProjectOpen(false)}
          onKeyDown={(e) => e.key === "Escape" && setNewProjectOpen(false)}
        >
          <div className="modal">
            <h3>New project</h3>
            <label className="modal-field">
              Grid size (1–{MAX_GRID_SIZE.toLocaleString()})
              <input
                type="number"
                min={1}
                max={MAX_GRID_SIZE}
                step={1}
                value={newGridSize}
                onChange={(e) => setNewGridSize(Number(e.target.value))}
              />
            </label>
            <label className="modal-field">
              Starting shape
              <select
                value={newGridShape}
                onChange={(e) => setNewGridShape(e.target.value as StartShape)}
              >
                <option value="cube">Cube</option>
                <option value="orb">Orb</option>
                <option value="cylinder">Cylinder</option>
                <option value="hollowCube">Hollow cube</option>
                <option value="plane">Plane</option>
                <option value="circle">Circle</option>
                <option value="empty">Empty</option>
              </select>
            </label>
            <div className="modal-buttons">
              <button type="button" onClick={createNewProject}>
                Create
              </button>
              <button type="button" onClick={() => setNewProjectOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
