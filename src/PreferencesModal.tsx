import { useEffect, useRef, useState } from "react";
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
  type StartShape,
  type ToneMappingPreference,
  type VoxelleDesktopPreferences,
} from "./preferences";

const SECTIONS = [
  { id: "prefs-general", label: "General" },
  { id: "prefs-collab", label: "Collaboration" },
  { id: "prefs-autosave", label: "Autosave" },
  { id: "prefs-graphics", label: "Graphics" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

type Props = {
  open: boolean;
  onClose: () => void;
  onFpsCounterChange?: (show: boolean) => void;
  onPingLatencyChange?: (show: boolean) => void;
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
  onPingLatencyChange,
  onEnableUpnpChange,
  onCollabDisplayNameChange,
  onCollabAccentColorChange,
  onCollabHostPortChange,
  collabHosting = false,
}: Props) {
  const [prefs, setPrefs] = useState<VoxelleDesktopPreferences>(loadPreferences);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [activeSection, setActiveSection] = useState<SectionId>("prefs-general");
  const [avatarNames, setAvatarNames] = useState<string[]>([]);
  const [userAvatarNames, setUserAvatarNames] = useState<string[]>([]);

  const scrollToSection = (id: SectionId) => {
    const container = scrollRef.current;
    if (!container) return;
    const el = container.querySelector<HTMLElement>(`#${id}`);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const onPrefsScroll = () => {
    const container = scrollRef.current;
    if (!container) return;
    const containerTop = container.getBoundingClientRect().top;
    let active: SectionId = "prefs-general";
    for (const section of SECTIONS) {
      const el = container.querySelector<HTMLElement>(`#${section.id}`);
      if (el && el.getBoundingClientRect().top - containerTop <= 20) {
        active = section.id;
      }
    }
    setActiveSection(active);
  };

  useEffect(() => {
    if (open) setPrefs(loadPreferences());
  }, [open]);

  useEffect(() => {
    void (invoke("avatar_list_embedded") as Promise<string[]>)
      .then(setAvatarNames)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!open) return;
    void (invoke("avatar_list_user") as Promise<string[]>)
      .then(setUserAvatarNames)
      .catch(() => {});
  }, [open]);

  const applyToneToGpu = (t: ToneMappingPreference) => {
    void invoke("set_tone_mapping", { mode: toneMappingToGpuMode(t) }).catch(() => {});
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

  const onGizmoOnTop = (checked: boolean) => {
    const next = { ...prefs, gizmoOnTop: checked };
    setPrefs(next);
    savePreferences(next);
    void invoke("set_gizmo_on_top", { enabled: checked }).catch(() => {});
  };

  const onFps = (checked: boolean) => {
    const next = { ...prefs, showFpsCounter: checked };
    setPrefs(next);
    savePreferences(next);
    onFpsCounterChange?.(checked);
  };

  const onPingLatency = (checked: boolean) => {
    const next = { ...prefs, showPingLatency: checked };
    setPrefs(next);
    savePreferences(next);
    onPingLatencyChange?.(checked);
  };

  const onReopenLastProject = (checked: boolean) => {
    const next = { ...prefs, reopenLastProject: checked };
    setPrefs(next);
    savePreferences(next);
  };

  const onAutoCheckUpdates = (checked: boolean) => {
    const next = { ...prefs, autoCheckUpdates: checked };
    setPrefs(next);
    savePreferences(next);
  };

  const onNewProjectDefaultSize = (raw: number) => {
    const v = Math.max(1, Math.min(256, Math.floor(raw)));
    const next = { ...prefs, newProjectDefaultSize: Number.isFinite(v) ? v : 32 };
    setPrefs(next);
    savePreferences(next);
  };

  const onNewProjectDefaultShape = (value: StartShape) => {
    const next = { ...prefs, newProjectDefaultShape: value };
    setPrefs(next);
    savePreferences(next);
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
    // Store the raw value so the user can freely clear the field.
    // Normalize only for persistence and collab broadcast.
    const toSave = preferencesWithCollabIdentity(prefs, clipped, prefs.collabAccentColor);
    setPrefs({
      ...prefs,
      collabDisplayName: clipped,
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

  const onCollabAvatar = (name: string) => {
    const next = { ...prefs, collabAvatarName: name };
    setPrefs(next);
    savePreferences(next);
    void invoke("set_local_avatar", { avatarName: name }).catch(() => {});
  };

  const onCollabHostPort = (raw: number) => {
    const n = normalizeCollabHostPort(raw);
    const next = { ...prefs, collabHostPort: n };
    setPrefs(next);
    savePreferences(next);
    onCollabHostPortChange?.(n);
  };

  const onEmissionLighting = (checked: boolean) => {
    const next = { ...prefs, enableEmissionLighting: checked };
    setPrefs(next);
    savePreferences(next);
    void invoke("set_emission_lighting", { enabled: checked }).catch(() => {});
  };

  const onSoftShadows = (checked: boolean) => {
    const next = { ...prefs, softShadows: checked };
    setPrefs(next);
    savePreferences(next);
    void invoke("set_soft_shadows", { enabled: checked }).catch(() => {});
  };

  const onSoftSunshafts = (checked: boolean) => {
    const next = { ...prefs, softSunshafts: checked };
    setPrefs(next);
    savePreferences(next);
    void invoke("set_soft_sunshafts", { enabled: checked }).catch(() => {});
  };

  const onHdr = (checked: boolean) => {
    const next = { ...prefs, hdr: checked };
    setPrefs(next);
    savePreferences(next);
    void invoke("set_hdr_output", { enabled: checked }).catch(() => {});
    // Switch tone mapper: HDR mode (6) when on, restore user pref when off.
    void invoke("set_tone_mapping", {
      mode: checked ? 6 : toneMappingToGpuMode(next.toneMapping),
    }).catch(() => {});
  };

  const pushAutosaveToRust = (next: VoxelleDesktopPreferences) => {
    void invoke("set_autosave_settings", autosaveSettingsInvokeArgs(next)).catch(() => {});
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
        <div className="modal--preferences-body">
          <nav className="prefs-toc" aria-label="Preferences sections">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                type="button"
                className={`prefs-toc-item${activeSection === s.id ? " is-active" : ""}`}
                onClick={() => scrollToSection(s.id)}
              >
                {s.label}
              </button>
            ))}
          </nav>
          <div className="modal--preferences-scroll" ref={scrollRef} onScroll={onPrefsScroll}>
            <div id="prefs-general" className="prefs-section-anchor" />
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
                checked={prefs.gizmoOnTop}
                onChange={(e) => onGizmoOnTop(e.target.checked)}
              />
              Always render gizmo on top
            </label>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.showFpsCounter}
                onChange={(e) => onFps(e.target.checked)}
              />
              FPS overlay
            </label>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.reopenLastProject}
                onChange={(e) => onReopenLastProject(e.target.checked)}
              />
              Automatically reopen last project on startup
            </label>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.autoCheckUpdates}
                onChange={(e) => onAutoCheckUpdates(e.target.checked)}
              />
              Check for updates on startup
            </label>
            <label className="prefs-select-label">
              <span className="prefs-select-label-text">Default new project size</span>
              <input
                type="number"
                className="prefs-number-input"
                min={1}
                max={256}
                step={1}
                value={prefs.newProjectDefaultSize}
                onChange={(e) => onNewProjectDefaultSize(Number(e.target.value))}
              />
              <span className="prefs-field-hint">Grid size pre-filled when creating a new project (1–256).</span>
            </label>
            <label className="prefs-select-label">
              <span className="prefs-select-label-text">Default new project shape</span>
              <select
                className="prefs-tone-select"
                value={prefs.newProjectDefaultShape}
                onChange={(e) => onNewProjectDefaultShape(e.target.value as StartShape)}
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
            <label className="prefs-select-label">
              <span className="prefs-select-label-text">Appearance</span>
              <select
                className="prefs-tone-select"
                value={prefs.appearanceTheme}
                onChange={(e) => onAppearanceTheme(e.target.value as AppearanceTheme)}
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
            <p className="prefs-field-hint prefs-section-hint">
              Coordinates for the real-time sun position button in the lighting panel.
            </p>
            <label className="prefs-select-label">
              <span className="prefs-select-label-text">Latitude</span>
              <input
                type="number"
                className="prefs-number-input"
                min={-90}
                max={90}
                step={0.1}
                value={prefs.sunLocationLat}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (!Number.isFinite(v)) return;
                  const next = { ...prefs, sunLocationLat: Math.max(-90, Math.min(90, v)) };
                  setPrefs(next);
                  savePreferences(next);
                }}
              />
            </label>
            <label className="prefs-select-label">
              <span className="prefs-select-label-text">Longitude</span>
              <input
                type="number"
                className="prefs-number-input"
                min={-180}
                max={180}
                step={0.1}
                value={prefs.sunLocationLon}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (!Number.isFinite(v)) return;
                  const next = { ...prefs, sunLocationLon: Math.max(-180, Math.min(180, v)) };
                  setPrefs(next);
                  savePreferences(next);
                }}
              />
            </label>
            <hr className="prefs-section-divider" />
            <h4 id="prefs-collab" className="prefs-section-title">
              Collaboration
            </h4>
            <label className="prefs-select-label">
              <span className="prefs-select-label-text">Display name</span>
              <input
                type="text"
                className="prefs-text-input"
                value={prefs.collabDisplayName}
                placeholder="Artist"
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
            <label className="prefs-select-label">
              <span className="prefs-select-label-text">Avatar</span>
              <select
                className="prefs-tone-select"
                value={prefs.collabAvatarName}
                onChange={(e) => onCollabAvatar(e.target.value)}
              >
                <option value="">Default (glowing dot)</option>
                <optgroup label="Built-in">
                  {avatarNames.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </optgroup>
                {userAvatarNames.length > 0 && (
                  <optgroup label="My Avatars">
                    {userAvatarNames.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </label>
            <div className="prefs-select-label">
              <span className="prefs-select-label-text" />
              <button
                className="prefs-action-button"
                onClick={() => void invoke("avatar_open_user_folder").catch(() => {})}
              >
                Open avatars folder
              </button>
            </div>
            <p className="prefs-field-hint">
              To create an avatar: build your character in Voxelle, then use <strong>File › Save As</strong> to save it as a <code>.voxelle</code> file. Drop that file into the avatars folder and reopen Preferences to see it here. Files must be under 64 KB.
            </p>
            {collabHosting ? (
              <p className="prefs-field-hint prefs-section-hint">
                Port and internet sharing are locked while you host. End the session to change them.
              </p>
            ) : null}
            <label className={`prefs-select-label${collabHosting ? " is-disabled" : ""}`}>
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
              Lets friends online join without manual router setup. Leave off unless you need it.
            </p>
            <label className={`prefs-checkbox-label${collabHosting ? " is-disabled" : ""}`}>
              <input
                type="checkbox"
                checked={prefs.enableUpnp}
                disabled={collabHosting}
                onChange={(e) => onEnableUpnp(e.target.checked)}
              />
              Internet guests (UPnP)
            </label>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.showPingLatency}
                onChange={(e) => onPingLatency(e.target.checked)}
              />
              Ping latency overlay
            </label>
            <hr className="prefs-section-divider" />
            <h4 id="prefs-autosave" className="prefs-section-title">
              Autosave
            </h4>
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
            <label className={`prefs-select-label${prefs.autosaveEnabled ? "" : " is-disabled"}`}>
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
            <label className={`prefs-select-label${prefs.autosaveEnabled ? "" : " is-disabled"}`}>
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
              <span className="prefs-field-hint">Per project; oldest drops first.</span>
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
            <hr className="prefs-section-divider" />
            <h4 id="prefs-graphics" className="prefs-section-title">
              Graphics
            </h4>
            <p className="prefs-field-hint prefs-section-hint">
              Changing these settings rebuilds the scene mesh.
            </p>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.enableEmissionLighting}
                onChange={(e) => onEmissionLighting(e.target.checked)}
              />
              Emission lighting
            </label>
            <p className="prefs-field-hint prefs-section-hint">
              Glow voxels cast colored light onto nearby surfaces.
            </p>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.softShadows}
                onChange={(e) => onSoftShadows(e.target.checked)}
              />
              Soft shadows
            </label>
            <p className="prefs-field-hint prefs-section-hint">
              Smooths shadow edges using multi-tap filtering.
            </p>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.softSunshafts}
                onChange={(e) => onSoftSunshafts(e.target.checked)}
              />
              Soft sunshafts
            </label>
            <p className="prefs-field-hint prefs-section-hint">
              Smooths sun shaft banding with per-pixel jitter.
            </p>
            <label className="prefs-checkbox-label">
              <input
                type="checkbox"
                checked={prefs.hdr}
                onChange={(e) => onHdr(e.target.checked)}
              />
              HDR output
            </label>
            <p className="prefs-field-hint prefs-section-hint">
              Extended dynamic range — highlights can exceed SDR white. Requires an HDR-capable
              display.
            </p>
          </div>
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
