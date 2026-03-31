type Props = {
  open: boolean;
  loading: boolean;
  loadProgress: number;
  loadPhase: string;
  pathLabel: string;
  onCancel: () => void;
};

/** Overlay while joining a collab session (connect + optional host snapshot load). */
export function CollabJoinProgressModal({
  open,
  loading,
  loadProgress,
  loadPhase,
  pathLabel,
  onCancel,
}: Props) {
  if (!open) return null;

  let detail = "Connecting to host…";
  if (loading) {
    detail = loadPhase.trim().length > 0 ? loadPhase : "Loading host project…";
  } else if (pathLabel === "collab snapshot") {
    detail = "Finishing…";
  }

  const pct = Math.round(Math.min(1, Math.max(0, loadProgress)) * 100);

  return (
    <div
      className="modal-overlay collab-join-progress-overlay"
      role="alertdialog"
      aria-modal="true"
      aria-busy="true"
      aria-labelledby="collab-join-progress-title"
      aria-describedby="collab-join-progress-detail"
      tabIndex={-1}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onCancel();
        }
      }}
    >
      <div className="modal collab-join-progress-modal">
        <div className="collab-join-progress-spinner" aria-hidden />
        <h3
          id="collab-join-progress-title"
          className="collab-join-progress-title"
        >
          Joining session
        </h3>
        <p
          id="collab-join-progress-detail"
          className="collab-join-progress-detail"
        >
          {detail}
        </p>
        {loading ? (
          <div
            className="collab-join-progress-bar-wrap"
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className="collab-join-progress-bar"
              style={{ width: `${pct}%` }}
            />
          </div>
        ) : null}
        <button
          className="btn btn-secondary collab-join-progress-cancel"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
