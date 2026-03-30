import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  loadPreferences,
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
};

export function PreferencesModal({ open, onClose, onFpsCounterChange }: Props) {
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
