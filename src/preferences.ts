/** Overlap with Voxelle web `voxelle-preferences` for shared keys. */

import { invoke } from "@tauri-apps/api/core";

export const VOXELLE_PREFERENCES_KEY = "voxelle-preferences";

export const TONE_MAPPING_OPTIONS = [
  { value: "neutral" as const, label: "Neutral" },
  { value: "aces" as const, label: "Filmic" },
  { value: "linear" as const, label: "Linear" },
  { value: "none" as const, label: "None" },
  { value: "agx" as const, label: "AgX" },
  { value: "reinhard" as const, label: "Reinhard" },
] as const;

export type ToneMappingPreference = (typeof TONE_MAPPING_OPTIONS)[number]["value"];

export type AppearanceTheme = "auto" | "light" | "dark";

export const APPEARANCE_THEME_OPTIONS = [
  { value: "auto" as const, label: "Auto (system)" },
  { value: "light" as const, label: "Light" },
  { value: "dark" as const, label: "Dark" },
] as const;

export function isAppearanceTheme(v: unknown): v is AppearanceTheme {
  return v === "auto" || v === "light" || v === "dark";
}

const TONE_ORDER: ToneMappingPreference[] = [
  "neutral",
  "aces",
  "linear",
  "none",
  "agx",
  "reinhard",
];

export function toneMappingToGpuMode(t: ToneMappingPreference): number {
  return Math.max(0, TONE_ORDER.indexOf(t));
}

export function isToneMappingPreference(v: unknown): v is ToneMappingPreference {
  return typeof v === "string" && (TONE_ORDER as readonly string[]).includes(v);
}

export type StartShape = "cube" | "orb" | "cylinder" | "hollowCube" | "plane" | "circle" | "empty";

const START_SHAPES: readonly StartShape[] = [
  "cube",
  "orb",
  "cylinder",
  "hollowCube",
  "plane",
  "circle",
  "empty",
];

export function isStartShape(v: unknown): v is StartShape {
  return typeof v === "string" && (START_SHAPES as readonly string[]).includes(v);
}

export type VoxelleDesktopPreferences = {
  showMovementDeltaHint: boolean;
  showDragDeltaHint: boolean;
  showFpsCounter: boolean;
  /** Show round-trip ping latency in the footer while in a session. */
  showPingLatency: boolean;
  /** Shown to others when you host or join a session. */
  collabDisplayName: string;
  /** Hex color (#rgb or #rrggbb) for your collaboration accent. */
  collabAccentColor: string;
  /** TCP port for the collaboration WebSocket when you host (1–65535). */
  collabHostPort: number;
  /** UPnP port mapping when hosting (off by default). */
  enableUpnp: boolean;
  toneMapping: ToneMappingPreference;
  /** When false, no timed autosave runs. */
  autosaveEnabled: boolean;
  /** Seconds between autosaves; 0 means never (same as disabled for the timer). */
  autosaveIntervalSecs: number;
  /** Rotating backup slots per project in app data (1–64). */
  autosaveKeepCount: number;
  /** UI chrome: follow OS, force light (paper), or force dark. */
  appearanceTheme: AppearanceTheme;
  /** Automatically reopen the last project when the app starts. */
  reopenLastProject: boolean;
  /** Bake irradiance from glow voxels onto nearby surfaces (re-runs on mesh rebuild). */
  enableEmissionLighting: boolean;
  /** Replace the rasterized renderer with a progressive GPU ray tracer. */
  raytraceEnabled: boolean;
  /** Use PCF soft shadows instead of hard single-tap shadows. */
  softShadows: boolean;
  /** Jitter sun-shaft ray march to smooth banding artefacts. */
  softSunshafts: boolean;
  /** Output HDR (Rgba16Float) to the display when supported. */
  hdr: boolean;
  /** Latitude for real-time sun position (-90 to 90). */
  sunLocationLat: number;
  /** Longitude for real-time sun position (-180 to 180). */
  sunLocationLon: number;
  /** Always draw the selection gizmo on top of scene geometry. */
  gizmoOnTop: boolean;
  /** Default grid size pre-filled in the New Project dialog (1–256). */
  newProjectDefaultSize: number;
  /** Default starting shape pre-filled in the New Project dialog. */
  newProjectDefaultShape: StartShape;
  /** Avatar name shown to other collab peers. Empty string = default glowing dot. */
  collabAvatarName: string;
};

const DEFAULTS: VoxelleDesktopPreferences = {
  showMovementDeltaHint: false,
  showDragDeltaHint: true,
  showFpsCounter: false,
  showPingLatency: false,
  collabDisplayName: "Artist",
  collabAccentColor: "#6699cc",
  collabHostPort: 27300,
  enableUpnp: false,
  toneMapping: "neutral",
  autosaveEnabled: true,
  autosaveIntervalSecs: 120,
  autosaveKeepCount: 5,
  appearanceTheme: "auto",
  reopenLastProject: false,
  enableEmissionLighting: true,
  raytraceEnabled: false,
  softShadows: true,
  softSunshafts: true,
  hdr: false,
  sunLocationLat: 41.9,
  sunLocationLon: -87.6,
  gizmoOnTop: true,
  newProjectDefaultSize: 32,
  newProjectDefaultShape: "circle",
  collabAvatarName: "",
};

const LEGACY_AUTOSAVE_INTERVAL_KEY = "voxelleAutosaveSecs";
const LEGACY_COLLAB_NAME_KEY = "voxelleCollabDisplayName";
const LEGACY_COLLAB_COLOR_KEY = "voxelleCollabColor";

const HEX_COLOR = /^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/;

export function normalizeCollabDisplayName(raw: string): string {
  const t = raw.trim().slice(0, 32);
  return t.length > 0 ? t : DEFAULTS.collabDisplayName;
}

/** Whether `theme` resolves to the light (paper) UI, including OS preference when `auto`. */
export function appearanceThemeResolvesToLight(theme: AppearanceTheme): boolean {
  if (theme === "light") return true;
  if (theme === "dark") return false;
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-color-scheme: light)").matches;
}

/** Sets `data-appearance` on `<html>` and syncs cold-start GPU gradient (Tauri). */
export function applyAppearanceToDocument(theme: AppearanceTheme): void {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("data-appearance", theme);
  const light = appearanceThemeResolvesToLight(theme);
  void invoke("set_start_screen_light", { light }).catch(() => {});
}

export function normalizeCollabAccentColor(raw: string): string {
  const t = raw.trim();
  if (!HEX_COLOR.test(t)) return DEFAULTS.collabAccentColor;
  if (t.length === 4) {
    const a = t[1]!;
    const b = t[2]!;
    const c = t[3]!;
    return `#${a}${a}${b}${b}${c}${c}`.toLowerCase();
  }
  return t.toLowerCase();
}

/** Merge current prefs with collaboration identity (normalized). */
export function preferencesWithCollabIdentity(
  base: VoxelleDesktopPreferences,
  displayName: string,
  accentColor: string,
): VoxelleDesktopPreferences {
  return {
    ...base,
    collabDisplayName: normalizeCollabDisplayName(displayName),
    collabAccentColor: normalizeCollabAccentColor(accentColor),
  };
}

function clampInt(n: number, min: number, max: number): number {
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, Math.floor(n)));
}

/** Collaboration host listen port (TCP / WebSocket). */
export function normalizeCollabHostPort(raw: unknown): number {
  if (typeof raw === "number" && Number.isFinite(raw)) return clampInt(raw, 1, 65535);
  return DEFAULTS.collabHostPort;
}

export function loadPreferences(): VoxelleDesktopPreferences {
  if (typeof localStorage === "undefined") return { ...DEFAULTS };
  try {
    const raw = localStorage.getItem(VOXELLE_PREFERENCES_KEY);
    let legacyInterval: number | undefined;
    try {
      const ls = localStorage.getItem(LEGACY_AUTOSAVE_INTERVAL_KEY);
      if (ls != null) {
        const v = Number(ls);
        if (Number.isFinite(v) && v >= 0) legacyInterval = v;
      }
    } catch {
      /* ignore */
    }
    if (!raw) {
      const base = { ...DEFAULTS };
      if (legacyInterval !== undefined) base.autosaveIntervalSecs = legacyInterval;
      try {
        const ln = localStorage.getItem(LEGACY_COLLAB_NAME_KEY);
        const lc = localStorage.getItem(LEGACY_COLLAB_COLOR_KEY);
        if (ln != null && ln.trim() !== "") base.collabDisplayName = normalizeCollabDisplayName(ln);
        if (lc != null && lc.trim() !== "") base.collabAccentColor = normalizeCollabAccentColor(lc);
      } catch {
        /* ignore */
      }
      return base;
    }
    const data = JSON.parse(raw) as unknown;
    if (!data || typeof data !== "object") return { ...DEFAULTS };
    const o = data as Record<string, unknown>;
    const intervalRaw = o.autosaveIntervalSecs;
    const interval =
      typeof intervalRaw === "number" && Number.isFinite(intervalRaw)
        ? clampInt(intervalRaw, 0, 86400)
        : legacyInterval !== undefined
          ? legacyInterval
          : DEFAULTS.autosaveIntervalSecs;
    return {
      showMovementDeltaHint:
        typeof o.showMovementDeltaHint === "boolean"
          ? o.showMovementDeltaHint
          : DEFAULTS.showMovementDeltaHint,
      showDragDeltaHint:
        typeof o.showDragDeltaHint === "boolean" ? o.showDragDeltaHint : DEFAULTS.showDragDeltaHint,
      showFpsCounter:
        typeof o.showFpsCounter === "boolean" ? o.showFpsCounter : DEFAULTS.showFpsCounter,
      showPingLatency:
        typeof o.showPingLatency === "boolean" ? o.showPingLatency : DEFAULTS.showPingLatency,
      collabDisplayName: (() => {
        if (typeof o.collabDisplayName === "string")
          return normalizeCollabDisplayName(o.collabDisplayName);
        try {
          const leg =
            typeof localStorage !== "undefined"
              ? localStorage.getItem(LEGACY_COLLAB_NAME_KEY)
              : null;
          if (leg != null && leg.trim() !== "") return normalizeCollabDisplayName(leg);
        } catch {
          /* ignore */
        }
        return DEFAULTS.collabDisplayName;
      })(),
      collabAccentColor: (() => {
        if (typeof o.collabAccentColor === "string")
          return normalizeCollabAccentColor(o.collabAccentColor);
        try {
          const leg =
            typeof localStorage !== "undefined"
              ? localStorage.getItem(LEGACY_COLLAB_COLOR_KEY)
              : null;
          if (leg != null && leg.trim() !== "") return normalizeCollabAccentColor(leg);
        } catch {
          /* ignore */
        }
        return DEFAULTS.collabAccentColor;
      })(),
      collabHostPort:
        typeof o.collabHostPort === "number" && Number.isFinite(o.collabHostPort)
          ? normalizeCollabHostPort(o.collabHostPort)
          : DEFAULTS.collabHostPort,
      enableUpnp: typeof o.enableUpnp === "boolean" ? o.enableUpnp : DEFAULTS.enableUpnp,
      toneMapping: isToneMappingPreference(o.toneMapping) ? o.toneMapping : DEFAULTS.toneMapping,
      autosaveEnabled:
        typeof o.autosaveEnabled === "boolean" ? o.autosaveEnabled : DEFAULTS.autosaveEnabled,
      autosaveIntervalSecs: interval,
      autosaveKeepCount:
        typeof o.autosaveKeepCount === "number" && Number.isFinite(o.autosaveKeepCount)
          ? clampInt(o.autosaveKeepCount, 1, 64)
          : DEFAULTS.autosaveKeepCount,
      appearanceTheme: isAppearanceTheme(o.appearanceTheme)
        ? o.appearanceTheme
        : DEFAULTS.appearanceTheme,
      reopenLastProject:
        typeof o.reopenLastProject === "boolean" ? o.reopenLastProject : DEFAULTS.reopenLastProject,
      enableEmissionLighting:
        typeof o.enableEmissionLighting === "boolean"
          ? o.enableEmissionLighting
          : DEFAULTS.enableEmissionLighting,
      raytraceEnabled:
        typeof o.raytraceEnabled === "boolean" ? o.raytraceEnabled : DEFAULTS.raytraceEnabled,
      softShadows: typeof o.softShadows === "boolean" ? o.softShadows : DEFAULTS.softShadows,
      softSunshafts:
        typeof o.softSunshafts === "boolean" ? o.softSunshafts : DEFAULTS.softSunshafts,
      hdr: typeof o.hdr === "boolean" ? o.hdr : DEFAULTS.hdr,
      sunLocationLat:
        typeof o.sunLocationLat === "number" && Number.isFinite(o.sunLocationLat)
          ? Math.max(-90, Math.min(90, o.sunLocationLat as number))
          : DEFAULTS.sunLocationLat,
      sunLocationLon:
        typeof o.sunLocationLon === "number" && Number.isFinite(o.sunLocationLon)
          ? Math.max(-180, Math.min(180, o.sunLocationLon as number))
          : DEFAULTS.sunLocationLon,
      gizmoOnTop: typeof o.gizmoOnTop === "boolean" ? o.gizmoOnTop : DEFAULTS.gizmoOnTop,
      newProjectDefaultSize:
        typeof o.newProjectDefaultSize === "number" && Number.isFinite(o.newProjectDefaultSize)
          ? clampInt(o.newProjectDefaultSize, 1, 256)
          : DEFAULTS.newProjectDefaultSize,
      newProjectDefaultShape: isStartShape(o.newProjectDefaultShape)
        ? o.newProjectDefaultShape
        : DEFAULTS.newProjectDefaultShape,
      collabAvatarName:
        typeof o.collabAvatarName === "string"
          ? o.collabAvatarName.trim().slice(0, 64)
          : DEFAULTS.collabAvatarName,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function savePreferences(prefs: VoxelleDesktopPreferences): void {
  if (typeof localStorage === "undefined") return;
  try {
    const prevRaw = localStorage.getItem(VOXELLE_PREFERENCES_KEY);
    let merged: Record<string, unknown> = {};
    if (prevRaw) {
      try {
        const p = JSON.parse(prevRaw) as unknown;
        if (p && typeof p === "object") merged = { ...(p as object) } as Record<string, unknown>;
      } catch {
        /* ignore */
      }
    }
    merged.showMovementDeltaHint = prefs.showMovementDeltaHint;
    merged.showDragDeltaHint = prefs.showDragDeltaHint;
    merged.showFpsCounter = prefs.showFpsCounter;
    merged.showPingLatency = prefs.showPingLatency;
    merged.collabDisplayName = prefs.collabDisplayName;
    merged.collabAccentColor = prefs.collabAccentColor;
    merged.collabAvatarName = prefs.collabAvatarName;
    merged.collabHostPort = prefs.collabHostPort;
    merged.enableUpnp = prefs.enableUpnp;
    merged.toneMapping = prefs.toneMapping;
    merged.autosaveEnabled = prefs.autosaveEnabled;
    merged.autosaveIntervalSecs = prefs.autosaveIntervalSecs;
    merged.autosaveKeepCount = prefs.autosaveKeepCount;
    merged.appearanceTheme = prefs.appearanceTheme;
    merged.reopenLastProject = prefs.reopenLastProject;
    merged.enableEmissionLighting = prefs.enableEmissionLighting;
    merged.raytraceEnabled = prefs.raytraceEnabled;
    merged.softShadows = prefs.softShadows;
    merged.softSunshafts = prefs.softSunshafts;
    merged.hdr = prefs.hdr;
    merged.sunLocationLat = prefs.sunLocationLat;
    merged.sunLocationLon = prefs.sunLocationLon;
    merged.gizmoOnTop = prefs.gizmoOnTop;
    merged.newProjectDefaultSize = prefs.newProjectDefaultSize;
    merged.newProjectDefaultShape = prefs.newProjectDefaultShape;
    localStorage.setItem(VOXELLE_PREFERENCES_KEY, JSON.stringify(merged));
    applyAppearanceToDocument(prefs.appearanceTheme);
  } catch {
    /* ignore */
  }
}

/** Payload for `invoke("set_autosave_settings", …)` (Tauri `args` wrapper). */
export function autosaveSettingsInvokeArgs(p: VoxelleDesktopPreferences) {
  return {
    args: {
      enabled: p.autosaveEnabled,
      intervalSecs: clampInt(p.autosaveIntervalSecs, 0, 86400),
      keepCount: clampInt(p.autosaveKeepCount, 1, 64),
    },
  };
}
