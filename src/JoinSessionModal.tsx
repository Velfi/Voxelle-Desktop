import { loadRecentJoinUrls } from "./joinRecent";

type Props = {
  open: boolean;
  onClose: () => void;
  joinUrl: string;
  onJoinUrlChange: (url: string) => void;
  onJoin: (urlOverride?: string) => void;
  collabActive: boolean;
};

export function JoinSessionModal({
  open,
  onClose,
  joinUrl,
  onJoinUrlChange,
  onJoin,
  collabActive,
}: Props) {
  if (!open) return null;

  const recent = loadRecentJoinUrls();

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="join-session-title"
      tabIndex={-1}
      onClick={(e) => e.target === e.currentTarget && onClose()}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <div className="modal modal--join-session">
        <h3 id="join-session-title" className="modal--join-session-title">
          Join session
        </h3>
        {collabActive ? (
          <p className="join-session-blocked" role="status">
            Leave your current session first to join another host.
          </p>
        ) : null}
        {recent.length > 0 ? (
          <div className="join-session-recent">
            <div className="join-session-recent-label">Recent servers</div>
            <ul className="join-session-recent-list" aria-label="Recent join URLs">
              {recent.map((url) => (
                <li key={url}>
                  <button
                    type="button"
                    className="join-session-recent-link"
                    disabled={collabActive}
                    title={url}
                    onClick={() => {
                      onJoinUrlChange(url);
                      onJoin(url);
                    }}
                  >
                    {url}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        <label className="modal-field join-session-url-field">
          <span className="join-session-url-label">Server URL</span>
          <input
            type="text"
            value={joinUrl}
            onChange={(e) => onJoinUrlChange(e.target.value)}
            onKeyDown={(e) =>
              e.key === "Enter" && !collabActive && onJoin(undefined)
            }
            placeholder="ws://host:port"
            disabled={collabActive}
            spellCheck={false}
            autoComplete="off"
          />
        </label>
        <div className="modal-buttons join-session-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="join-session-submit"
            disabled={collabActive}
            onClick={() => onJoin(undefined)}
          >
            Join
          </button>
        </div>
      </div>
    </div>
  );
}
