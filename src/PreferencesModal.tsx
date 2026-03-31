import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  APPEARANCE_THEME_OPTIONS,
  autosaveSettingsInvokeArgs,
  loadPreferences,
  normalizeCollabHostPort,
  preferencesWithCollabIdentity,
  savePreferences,
  TONE_MAPPING_OPTIONS,
  toneMappingToGpuMode,
  type AppearanceTheme,
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
  const [prefs, setPrefs] =
    useState<VoxelleDesktopPreferences>(loadPreferences);

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

  const onAppearanceTheme = (value: AppearanceTheme) => {
    const next = { ...prefs, appearanceTheme: value };
    setPrefs(next);
    savePreferences(next);
  };

  const onEnableUpnp = (checked: boolean) => {
    const next = { ...prefs, enableUpnp: checked };
    setPrefs(next);
    savePreferences(next);
    onEnableUpnpChange?.(checked);
  };

  const onCollabName = (raw: string) => {
    const clipped = raw.slice(0, 32);
    const toSave = preferencesWithCollabIdentity(
      prefs,
      clipped,
      prefs.collabAccentColor,
    );
    setPrefs({
      ...prefs,
      collabDisplayName: toSave.collabDisplayName,
      collabAccentColor: toSave.collabAccentColor,
    });
    savePreferences(toSave);
    onCollabDisplayNameChange?.(toSave.collabDisplayName);
  };

  const onCollabColor = (raw: string) => {
    const toSave = preferencesWithCollabIdentity(
      prefs,
      prefs.collabDisplayName,
      raw,
    );
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
    void invoke(
      "set_autosave_settings",
      autosaveSettingsInvokeArgs(next),
    ).catch(() => {});
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
            Movement hints while drawing
          </label>
          <label className="prefs-checkbox-label">
            <input
              type="checkbox"
              checked={prefs.showDragDeltaHint}
              onChange={(e) => onDragDelta(e.target.checked)}
            />
            Selection move hints
          </label>
          <label className="prefs-checkbox-label">
            <input
              type="checkbox"
              checked={prefs.showFpsCounter}
              onChange={(e) => onFps(e.target.checked)}
            />
            FPS overlay
          </label>
          <label className="prefs-select-label">
            <span className="prefs-select-label-text">Appearance</span>
            <select
              className="prefs-tone-select"
              value={prefs.appearanceTheme}
              onChange={(e) =>
                onAppearanceTheme(e.target.value as AppearanceTheme)
              }
            >
              {APPEARANCE_THEME_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <span className="prefs-field-hint">
              Light uses an unbleached paper tone. Auto follows this device.
            </span>
          </label>
          <h4 className="prefs-section-title">Collaboration</h4>
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
              Port and internet sharing are locked while you host. End the
              session to change them.
            </p>
          ) : null}
          <label
            className={`prefs-select-label${collabHosting ? " is-disabled" : ""}`}
          >
            <span className="prefs-select-label-text">Host port</span>
            <input
              type="number"
              className="prefs-number-input"
              min={1}
              max={65535}
              disabled={collabHosting}
              value={prefs.collabHostPort}
              onChange={(e) => onCollabHostPort(Number(e.target.value))}
            />
            <span className="prefs-field-hint">Default 27300.</span>
          </label>
          <p
            className={`prefs-field-hint prefs-section-hint${collabHosting ? " is-disabled" : ""}`}
          >
            Lets friends online join without manual router setup. Leave off
            unless you need it.
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
            Internet guests (UPnP)
          </label>
          <h4 className="prefs-section-title">Autosave</h4>
          <p className="prefs-field-hint prefs-section-hint">
            Backups live in the app — your file on disk updates when you save.
          </p>
          <label className="prefs-checkbox-label">
            <input
              type="checkbox"
              checked={prefs.autosaveEnabled}
              onChange={(e) => onAutosaveEnabled(e.target.checked)}
            />
            Timed backups
          </label>
          <label
            className={`prefs-select-label${prefs.autosaveEnabled ? "" : " is-disabled"}`}
          >
            <span className="prefs-select-label-text">Every (seconds)</span>
            <input
              type="number"
              className="prefs-number-input"
              min={0}
              max={86400}
              disabled={!prefs.autosaveEnabled}
              value={prefs.autosaveIntervalSecs}
              onChange={(e) => onAutosaveInterval(Number(e.target.value))}
            />
            <span className="prefs-field-hint">0 turns off the timer.</span>
          </label>
          <label
            className={`prefs-select-label${prefs.autosaveEnabled ? "" : " is-disabled"}`}
          >
            <span className="prefs-select-label-text">Keep backups</span>
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
              Per project; oldest drops first.
            </span>
          </label>
          <label className="prefs-select-label">
            <span className="prefs-select-label-text">Display look</span>
            <select
              className="prefs-tone-select"
              value={prefs.toneMapping}
              onChange={(e) => onTone(e.target.value as ToneMappingPreference)}
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
