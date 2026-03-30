/** Overlap with Voxelle web `voxelle-preferences` for shared keys. */

export const VOXELLE_PREFERENCES_KEY = "voxelle-preferences";

export const TONE_MAPPING_OPTIONS = [
  { value: "neutral" as const, label: "Neutral (balanced)" },
  { value: "aces" as const, label: "ACES Filmic" },
  { value: "linear" as const, label: "Linear" },
  { value: "none" as const, label: "None" },
  { value: "agx" as const, label: "AgX" },
  { value: "reinhard" as const, label: "Reinhard" },
] as const;

export type ToneMappingPreference = (typeof TONE_MAPPING_OPTIONS)[number]["value"];

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

export type VoxelleDesktopPreferences = {
  showMovementDeltaHint: boolean;
  showDragDeltaHint: boolean;
  showFpsCounter: boolean;
  toneMapping: ToneMappingPreference;
};

const DEFAULTS: VoxelleDesktopPreferences = {
  showMovementDeltaHint: false,
  showDragDeltaHint: true,
  showFpsCounter: false,
  toneMapping: "neutral",
};

export function loadPreferences(): VoxelleDesktopPreferences {
  if (typeof localStorage === "undefined") return { ...DEFAULTS };
  try {
    const raw = localStorage.getItem(VOXELLE_PREFERENCES_KEY);
    if (!raw) return { ...DEFAULTS };
    const data = JSON.parse(raw) as unknown;
    if (!data || typeof data !== "object") return { ...DEFAULTS };
    const o = data as Record<string, unknown>;
    return {
      showMovementDeltaHint:
        typeof o.showMovementDeltaHint === "boolean"
          ? o.showMovementDeltaHint
          : DEFAULTS.showMovementDeltaHint,
      showDragDeltaHint:
        typeof o.showDragDeltaHint === "boolean"
          ? o.showDragDeltaHint
          : DEFAULTS.showDragDeltaHint,
      showFpsCounter:
        typeof o.showFpsCounter === "boolean"
          ? o.showFpsCounter
          : DEFAULTS.showFpsCounter,
      toneMapping: isToneMappingPreference(o.toneMapping)
        ? o.toneMapping
        : DEFAULTS.toneMapping,
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
        if (p && typeof p === "object") merged = { ...(p as object) } as Record<
          string,
          unknown
        >;
      } catch {
        /* ignore */
      }
    }
    merged.showMovementDeltaHint = prefs.showMovementDeltaHint;
    merged.showDragDeltaHint = prefs.showDragDeltaHint;
    merged.showFpsCounter = prefs.showFpsCounter;
    merged.toneMapping = prefs.toneMapping;
    localStorage.setItem(VOXELLE_PREFERENCES_KEY, JSON.stringify(merged));
  } catch {
    /* ignore */
  }
}
