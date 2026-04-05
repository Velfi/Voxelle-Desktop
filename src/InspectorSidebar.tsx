import { invoke } from "@tauri-apps/api/core";
import type { RosterEntry, SceneObjectRow } from "./types";

interface Props {
  rightSidebarExpanded: boolean;
  setRightSidebarExpanded: (updater: (v: boolean) => boolean) => void;

  // Scene objects
  sceneObjects: SceneObjectRow[];
  sceneObjectsErr: string | null;
  activeObjectId: number;
  setActiveObjectId: (id: number) => void;
  refreshSceneObjects: () => void;

  // Collab session
  collabActive: boolean;
  hostWsUrl: string | null;
  hostWanUrl: string | null;
  hostingCopied: boolean;
  copyHostingJoinAddress: () => void;
  prefsEnableUpnp: boolean;
  natPending: boolean;
  natError: string | null;
  hostPort: number;
  roster: RosterEntry[];
  localPeerId: number;
  amLeader: boolean;
  onRosterSnapCamera: (peerId: number) => void;
  setCanEdit: (peerId: number, canEdit: boolean) => void;
}

export function InspectorSidebar({
  rightSidebarExpanded,
  setRightSidebarExpanded,
  sceneObjects,
  sceneObjectsErr,
  activeObjectId,
  setActiveObjectId,
  refreshSceneObjects,
  collabActive,
  hostWsUrl,
  hostWanUrl,
  hostingCopied,
  copyHostingJoinAddress,
  prefsEnableUpnp,
  natPending,
  natError,
  hostPort,
  roster,
  localPeerId: _localPeerId,
  amLeader,
  onRosterSnapCamera,
  setCanEdit,
}: Props) {
  return (
    <aside
      className={
        rightSidebarExpanded
          ? "app-sidebar app-sidebar-right is-expanded"
          : "app-sidebar app-sidebar-right is-collapsed"
      }
      aria-label="Inspector"
    >
      <div className="sidebar-header sidebar-header-right">
        <button
          type="button"
          className="sidebar-expand-toggle sidebar-expand-toggle-right"
          onClick={() => setRightSidebarExpanded((v) => !v)}
          aria-expanded={rightSidebarExpanded}
          title={rightSidebarExpanded ? "Collapse inspector" : "Expand inspector"}
        >
          {rightSidebarExpanded ? (
            <>
              <span className="sidebar-expand-toggle-label">Inspector</span>
              <span className="sidebar-expand-toggle-icon" aria-hidden>
                »
              </span>
            </>
          ) : (
            <span className="sidebar-expand-toggle-icon" aria-hidden>
              «
            </span>
          )}
        </button>
      </div>
      {rightSidebarExpanded ? (
        <div className="sidebar-scroll">
          <div
            className="sidebar-expanded-slot sidebar-expanded-slot-right"
            aria-label="Inspector content"
          >
            <div className="inspector-objects">
              <h4 className="inspector-heading">Objects</h4>
              {sceneObjectsErr ? <p className="inspector-hint">{sceneObjectsErr}</p> : null}
              <ul className="inspector-object-list">
                {sceneObjects
                  .slice()
                  .sort((a, b) => a.sortOrder - b.sortOrder || a.id - b.id)
                  .map((o) => (
                    <li key={o.id} className="inspector-object-row">
                      <label className="inspector-active">
                        <input
                          type="radio"
                          name="activeObject"
                          checked={activeObjectId === o.id}
                          onChange={() => {
                            void invoke("set_active_object", {
                              id: o.id,
                            }).then(() => {
                              setActiveObjectId(o.id);
                              refreshSceneObjects();
                            });
                          }}
                        />
                        <span className="inspector-object-name">{o.name}</span>
                      </label>
                      <label className="inspector-visible">
                        <input
                          type="checkbox"
                          checked={o.visible}
                          onChange={(e) => {
                            void invoke("set_object_visible", {
                              id: o.id,
                              visible: e.target.checked,
                            }).then(() => refreshSceneObjects());
                          }}
                        />
                        Visible
                      </label>
                    </li>
                  ))}
              </ul>
              <button
                type="button"
                className="inspector-new-object"
                onClick={() => {
                  void invoke<number>("create_scene_object", {
                    name: "",
                  }).then(() => refreshSceneObjects());
                }}
              >
                New object
              </button>
            </div>
            {collabActive ? (
              <div className="inspector-collaboration">
                <h4 className="inspector-heading">Session</h4>
                {hostWsUrl ? (
                  <>
                    <button
                      type="button"
                      className="inspector-copy-invite-btn"
                      onClick={copyHostingJoinAddress}
                      title={hostingCopied ? "Copied" : "Copy invite link"}
                    >
                      <span className="inspector-copy-invite-label">
                        {hostingCopied ? "Copied!" : "Copy invite link"}
                      </span>
                      <code className="inspector-copy-invite-url">{hostWanUrl ?? hostWsUrl}</code>
                    </button>
                    {hostWanUrl ? (
                      <p className="collab-hint inspector-collab-hint">
                        Nearby: <code>{hostWsUrl}</code>
                      </p>
                    ) : null}
                    {prefsEnableUpnp && natPending ? (
                      <p
                        className="collab-hint collab-hint-muted inspector-collab-hint"
                        role="status"
                      >
                        Checking your router…
                      </p>
                    ) : null}
                    {natError ? (
                      <p
                        className="collab-hint collab-hint-warn inspector-collab-hint"
                        role="alert"
                      >
                        {natError} You can forward port {hostPort} in your router settings. Some
                        networks won&apos;t allow guests over the internet.
                      </p>
                    ) : null}
                  </>
                ) : null}
                <h4 className="inspector-heading inspector-roster-heading">Roster</h4>
                <ul className="collab-roster inspector-collab-roster">
                  {roster.map((r) => (
                    <li key={r.peerId}>
                      <button
                        type="button"
                        className="collab-roster-name"
                        onClick={() => onRosterSnapCamera(r.peerId)}
                        title="Jump to their view"
                      >
                        <span
                          className="collab-swatch"
                          style={{
                            background: `#${(r.colorRgb & 0xffffff).toString(16).padStart(6, "0")}`,
                          }}
                        />
                        {r.displayName}
                        {r.isLeader ? " (leader)" : ""}
                      </button>
                      {!r.isLeader && amLeader ? (
                        <>
                          <label className="collab-can-edit">
                            <input
                              type="checkbox"
                              checked={r.canEdit}
                              onChange={(e) => setCanEdit(r.peerId, e.target.checked)}
                            />
                            Edit
                          </label>
                          <button
                            type="button"
                            className="collab-kick"
                            title="Remove guest"
                            onClick={() =>
                              void invoke("collab_kick_peer", {
                                targetPeer: r.peerId,
                              })
                            }
                          >
                            Kick
                          </button>
                        </>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </aside>
  );
}
