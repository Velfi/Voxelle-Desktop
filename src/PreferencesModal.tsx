import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  autosaveSettingsInvokeArgs,
  loadPreferences,
  normalizeCollabHostPort,
  preferencesWithCollabIdentity,
  savePreferences,
  TONE_MAPPING_OPTIONS,
  toneMappingToGpuMode,
  type ToneMappingPreference,
  type VoxelleDesktopPreferences,
} from "./preferences";

type Props = {
  open: boolean;
  onClose: () => void;
  onFpsCounterChange?: (show: boolean) => void;
  onEnableUpnpChange?: (enabled: boolean) => void;
  onCollabDisplayNameChange?: (name: string) => void;
  onCollabAccentColorChange?: (color: string) => void;
  onCollabHostPortChange?: (port: number) => void;
  /** When true, hosting port and UPnP cannot be edited (active host session). */
  collabHosting?: boolean;
};

export function PreferencesModal({
  open,
  onClose,
  onFpsCounterChange,
  onEnableUpnpChange,
  onCollabDisplayNameChange,
  onCollabAccentColorChange,
  onCollabHostPortChange,
  collabHosting = false,
}: Props) {
  const [prefs, setPrefs] = useState<VoxelleDesktopPreferences>(loadPreferences);

  useEffect(() => {
    if (open) setPrefs(loadPreferences());
  }, [open]);

  const applyToneToGpu = (t: ToneMappingPreference) => {
    void invoke("set_tone_mapping", { mode: toneMappingToGpuMode(t) }).catch(
      () => {},
    );
  };

  const onMovementDelta = (checked: boolean) => {
    const next = { ...prefs, showMovementDeltaHint: checked };
    setPrefs(next);
    savePreferences(next);
  };

  const onDragDelta = (checked: boolean) => {
    const next = { ...prefs, showDragDeltaHint: checked };
    setPrefs(next);
    savePreferences(next);
  };

  const onFps = (checked: boolean) => {
    const next = { ...prefs, showFpsCounter: checked };
    setPrefs(next);
    savePreferences(next);
    onFpsCounterChange?.(checked);
  };

  const onTone = (value: ToneMappingPreference) => {
    const next = { ...prefs, toneMapping: value };
    setPrefs(next);
    savePreferences(next);
    applyToneToGpu(value);
  };

  const onEnableUpnp = (checked: boolean) => {
    const next = { ...prefs, enableUpnp: checked };
    setPrefs(next);
    savePreferences(next);
    onEnableUpnpChange?.(checked);
  };

  const onCollabName = (raw: string) => {
    const clipped = raw.slice(0, 32);
    const toSave = preferencesWithCollabIdentity(prefs, clipped, prefs.collabAccentColor);
    setPrefs({
      ...prefs,
      collabDisplayName: toSave.collabDisplayName,
      collabAccentColor: toSave.collabAccentColor,
    });
    savePreferences(toSave);
    onCollabDisplayNameChange?.(toSave.collabDisplayName);
  };

  const onCollabColor = (raw: string) => {
    const toSave = preferencesWithCollabIdentity(prefs, prefs.collabDisplayName, raw);
    setPrefs({
      ...prefs,
      collabDisplayName: toSave.collabDisplayName,
      collabAccentColor: toSave.collabAccentColor,
    });
    savePreferences(toSave);
    onCollabAccentColorChange?.(toSave.collabAccentColor);
  };

  const onCollabHostPort = (raw: number) => {
    const n = normalizeCollabHostPort(raw);
    const next = { ...prefs, collabHostPort: n };
    setPrefs(next);
    savePreferences(next);
    onCollabHostPortChange?.(n);
  };

  const pushAutosaveToRust = (next: VoxelleDesktopPreferences) => {
    void invoke("set_autosave_settings", autosaveSettingsInvokeArgs(next)).catch(
      () => {},
    );
  };

  const onAutosaveEnabled = (checked: boolean) => {
    const next = { ...prefs, autosaveEnabled: checked };
    setPrefs(next);
    savePreferences(next);
    pushAutosaveToRust(next);
  };

  const onAutosaveInterval = (raw: number) => {
    const v = Math.max(0, Math.min(86400, Math.floor(raw)));
    const next = { ...prefs, autosaveIntervalSecs: Number.isFinite(v) ? v : 0 };
    setPrefs(next);
    savePreferences(next);
    pushAutosaveToRust(next);
  };

  const onAutosaveKeep = (raw: number) => {
    const v = Math.max(1, Math.min(64, Math.floor(raw)));
    const next = {
      ...prefs,
      autosaveKeepCount: Number.isFinite(v) ? v : 1,
    };
    setPrefs(next);
    savePreferences(next);
    pushAutosaveToRust(next);
  };

  if (!open) return null;

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="preferences-title"
      tabIndex={-1}
      onClick={(e) => e.target === e.currentTarget && onClose()}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <div className="modal modal--preferences">
        <h3 id="preferences-title" className="modal--preferences-title">
          Preferences
        </h3>
        <div className="modal--preferences-scroll">
          <label className="prefs-checkbox-label">
            <input
              type="checkbox"
              checked={prefs.showMovementDeltaHint}
              onChange={(e) => onMovementDelta(e.target.checked)}
            />
            Show movement delta hint (Δx, Δy, Δz near cursor during strokes)
          </label>
          <p className="prefs-field-hint prefs-desktop-note">
            Sculpt overlays match Voxelle web; not shown in this desktop build yet.
          </p>
          <label className="prefs-checkbox-label">
            <input
              type="checkbox"
              checked={prefs.showDragDeltaHint}
              onChange={(e) => onDragDelta(e.target.checked)}
            />
            Show selection move drag hint (line and delta at original position)
          </label>
          <label className="prefs-checkbox-label">
            <input
              type="checkbox"
              checked={prefs.showFpsCounter}
              onChange={(e) => onFps(e.target.checked)}
            />
            Show FPS counter (viewport overlay)
          </label>
          <h4 className="prefs-section-title">Collaboration</h4>
          <p className="prefs-field-hint prefs-section-hint">
            Name and color are shown to others when you host or join a session.
          </p>
          <label className="prefs-select-label">
            <span className="prefs-select-label-text">Display name</span>
            <input
              type="text"
              className="prefs-text-input"
              value={prefs.collabDisplayName}
              maxLength={32}
              onChange={(e) => onCollabName(e.target.value)}
            />
          </label>
          <label className="prefs-select-label">
            <span className="prefs-select-label-text">Accent color</span>
            <input
              type="color"
              className="prefs-color-input"
              value={prefs.collabAccentColor}
              onChange={(e) => onCollabColor(e.target.value)}
            />
          </label>
          {collabHosting ? (
            <p className="prefs-field-hint prefs-section-hint">
              Listen port and UPnP are locked while you are hosting. Stop the
              session to change them.
            </p>
          ) : null}
          <label
            className={`prefs-select-label${collabHosting ? " is-disabled" : ""}`}
          >
            <span className="prefs-select-label-text">Hosting port (TCP)</span>
            <input
              type="number"
              className="prefs-number-input"
              min={1}
              max={65535}
              disabled={collabHosting}
              value={prefs.collabHostPort}
              onChange={(e) => onCollabHostPort(Number(e.target.value))}
            />
            <span className="prefs-field-hint">
              WebSocket listen port when you start a session (default 27300).
            </span>
          </label>
          <p
            className={`prefs-field-hint prefs-section-hint${collabHosting ? " is-disabled" : ""}`}
          >
            When you host a session, this asks your router to open a port so guests
            outside your LAN can connect (UPnP). Off by default; use only if you
            need internet guests.
          </p>
          <label
            className={`prefs-checkbox-label${collabHosting ? " is-disabled" : ""}`}
          >
            <input
              type="checkbox"
              checked={prefs.enableUpnp}
              disabled={collabHosting}
              onChange={(e) => onEnableUpnp(e.target.checked)}
            />
            Enable UPnP when hosting
          </label>
          <h4 className="prefs-section-title">Autosave</h4>
          <p className="prefs-field-hint prefs-section-hint">
            Backups are stored in app data only; your project file on disk is not
            overwritten by autosave.
          </p>
          <label className="prefs-checkbox-label">
            <input
              type="checkbox"
              checked={prefs.autosaveEnabled}
              onChange={(e) => onAutosaveEnabled(e.target.checked)}
            />
            Enable timed autosave
          </label>
          <label
            className={`prefs-select-label${prefs.autosaveEnabled ? "" : " is-disabled"}`}
          >
            <span className="prefs-select-label-text">Interval (seconds)</span>
            <input
              type="number"
              className="prefs-number-input"
              min={0}
              max={86400}
              disabled={!prefs.autosaveEnabled}
              value={prefs.autosaveIntervalSecs}
              onChange={(e) => onAutosaveInterval(Number(e.target.value))}
            />
            <span className="prefs-field-hint">0 = never (same as disabling).</span>
          </label>
          <label
            className={`prefs-select-label${prefs.autosaveEnabled ? "" : " is-disabled"}`}
          >
            <span className="prefs-select-label-text">Backups to keep (per project)</span>
            <input
              type="number"
              className="prefs-number-input"
              min={1}
              max={64}
              disabled={!prefs.autosaveEnabled}
              value={prefs.autosaveKeepCount}
              onChange={(e) => onAutosaveKeep(Number(e.target.value))}
            />
            <span className="prefs-field-hint">
              Rotating files in app data; reopen uses the newest backup.
            </span>
          </label>
          <label className="prefs-select-label">
            <span className="prefs-select-label-text">Viewport tone mapping</span>
            <select
              className="prefs-tone-select"
              value={prefs.toneMapping}
              onChange={(e) =>
                onTone(e.target.value as ToneMappingPreference)
              }
            >
              {TONE_MAPPING_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="modal-buttons modal--preferences-footer">
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
