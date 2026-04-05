// ── Status bar (footer) ──────────────────────────────────────────────
// Extracted from App.tsx.

import { memo, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { RosterEntry } from "./types";

export interface StatusBarProps {
  showStartScreen: boolean;
  statusBarMessage: string;
  pathLabel: string;
  collabActive: boolean;
  hostWsUrl: string | null;
  hostingCopied: boolean;
  copyHostingJoinAddress: () => void;
  roster: RosterEntry[];
  setLeaveConfirmOpen: (v: boolean) => void;
  startHost: () => void;
  showFpsCounter: boolean;
  showEditorChrome: boolean;
  fpsDisplayed: number;
  showPingLatency: boolean;
  pingMs: number | null;
}

export const StatusBar = memo(function StatusBar(props: StatusBarProps) {
  const {
    showStartScreen,
    statusBarMessage,
    pathLabel,
    collabActive,
    hostWsUrl,
    hostingCopied,
    copyHostingJoinAddress,
    roster,
    setLeaveConfirmOpen,
    startHost,
    showFpsCounter,
    showEditorChrome,
    fpsDisplayed,
    showPingLatency,
    pingMs,
  } = props;

  const [rtActive, setRtActive] = useState(false);

  useEffect(() => {
    void invoke<boolean>("get_raytrace_mode")
      .then(setRtActive)
      .catch(() => {});
    const unlisten = listen<boolean>("voxelle-raytrace-changed", (e) => {
      setRtActive(e.payload);
    });
    return () => {
      void unlisten.then((f) => f());
    };
  }, []);

  const toggleRt = () => {
    const next = !rtActive;
    void invoke("set_raytrace_mode", { enabled: next }).catch(() => {});
  };

  return (
    <footer
      className={`app-status-bar${showStartScreen ? " is-start-screen" : ""}`}
      role="contentinfo"
    >
      <div className="status-bar-main">
        <div
          className="status-bar-message"
          role="status"
          aria-live="polite"
          title={pathLabel || statusBarMessage}
        >
          {statusBarMessage}
        </div>
        {collabActive ? (
          <>
            {hostWsUrl ? (
              <button
                type="button"
                className="status-bar-hosting-btn"
                onClick={copyHostingJoinAddress}
                title={hostingCopied ? "Copied" : "Copy invite link"}
              >
                {hostingCopied
                  ? "Copied invite link"
                  : `Hosting \u00b7 ${roster.length} ${roster.length === 1 ? "person" : "people"}`}
              </button>
            ) : (
              <span className="status-bar-hosting-btn is-guest">
                {`In session \u00b7 ${roster.length} ${roster.length === 1 ? "person" : "people"}`}
              </span>
            )}
            <button
              type="button"
              className="status-bar-hosting-btn is-leave"
              onClick={() => setLeaveConfirmOpen(true)}
              title={hostWsUrl ? "End session" : "Leave session"}
            >
              {hostWsUrl ? "End" : "Leave"}
            </button>
          </>
        ) : !showStartScreen ? (
          <button
            type="button"
            className="status-bar-hosting-btn"
            onClick={startHost}
            title="Start a new session"
          >
            Start Session
          </button>
        ) : null}
      </div>
      {showEditorChrome ? (
        <button
          type="button"
          className={`status-bar-rt-btn${rtActive ? " is-active" : ""}`}
          onClick={toggleRt}
          title={rtActive ? "Disable ray tracing" : "Enable ray tracing"}
        >
          RT
        </button>
      ) : null}
      {showFpsCounter && showEditorChrome ? (
        <div className="fps-counter" role="status" aria-live="polite">
          {fpsDisplayed} FPS
        </div>
      ) : null}
      {showPingLatency && collabActive && pingMs !== null && showEditorChrome ? (
        <div className="fps-counter" role="status" aria-live="polite">
          {pingMs} ms
        </div>
      ) : null}
    </footer>
  );
});
