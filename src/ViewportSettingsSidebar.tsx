import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getSunPosition } from "./solarPosition";
import { loadPreferences } from "./preferences";

export type SceneLightingPayload = {
  ambientIntensity: number;
  sunlightIntensity: number;
  lightColor: string;
  lightAngle: number;
  lightElevation: number;
  enableShadows: boolean;
  enableSky: boolean;
  backgroundColor: string;
  exposureEv: number;
  autoExposure: boolean;
};

const TONE_EXPOSURE_MIN = -5;
const TONE_EXPOSURE_MAX = 5;

function defaultLighting(): SceneLightingPayload {
  return {
    ambientIntensity: 1,
    sunlightIntensity: 1,
    lightColor: "#ffffff",
    lightAngle: 45,
    lightElevation: 45,
    enableShadows: true,
    enableSky: true,
    backgroundColor: "#0a0b0e",
    exposureEv: 0,
    autoExposure: false,
  };
}

type LightPresetId =
  | "balanced"
  | "sunny"
  | "cloudy"
  | "incandescent"
  | "fluorescent"
  | "moonlight"
  | "dark";

const LIGHT_PRESETS: readonly {
  id: LightPresetId;
  title: string;
  ambientIntensity: number;
  sunlightIntensity: number;
  lightColor: string;
  lightAngle: number;
  lightElevation: number;
  enableShadows: boolean;
}[] = [
  {
    id: "balanced",
    title: "Balanced",
    ambientIntensity: 1,
    sunlightIntensity: 1,
    lightColor: "#ffffff",
    lightAngle: 45,
    lightElevation: 45,
    enableShadows: true,
  },
  {
    id: "sunny",
    title: "Sunny day",
    ambientIntensity: 0.45,
    sunlightIntensity: 2.3,
    lightColor: "#fff8e8",
    lightAngle: 50,
    lightElevation: 55,
    enableShadows: true,
  },
  {
    id: "cloudy",
    title: "Cloudy day",
    ambientIntensity: 1.1,
    sunlightIntensity: 0.65,
    lightColor: "#d4dce2",
    lightAngle: 55,
    lightElevation: 45,
    enableShadows: true,
  },
  {
    id: "incandescent",
    title: "Incandescent",
    ambientIntensity: 0.35,
    sunlightIntensity: 1.2,
    lightColor: "#ffc080",
    lightAngle: 35,
    lightElevation: 55,
    enableShadows: true,
  },
  {
    id: "fluorescent",
    title: "Fluorescent",
    ambientIntensity: 0.88,
    sunlightIntensity: 0.9,
    lightColor: "#e0f0ea",
    lightAngle: 35,
    lightElevation: 55,
    enableShadows: true,
  },
  {
    id: "moonlight",
    title: "Moonlight",
    ambientIntensity: 0.08,
    sunlightIntensity: 0.18,
    lightColor: "#8aa8d8",
    lightAngle: 110,
    lightElevation: 26,
    enableShadows: true,
  },
  {
    id: "dark",
    title: "Total darkness",
    ambientIntensity: 0,
    sunlightIntensity: 0,
    lightColor: "#ffffff",
    lightAngle: 45,
    lightElevation: 40,
    enableShadows: false,
  },
];

function safeHex(v: string, fallback: string): string {
  const t = v.trim();
  return /^#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$/.test(t) ? t : fallback;
}

function matchesPreset(l: SceneLightingPayload, p: (typeof LIGHT_PRESETS)[number]): boolean {
  return (
    l.ambientIntensity === p.ambientIntensity &&
    l.sunlightIntensity === p.sunlightIntensity &&
    l.lightColor.toLowerCase() === p.lightColor.toLowerCase() &&
    l.lightAngle === p.lightAngle &&
    l.lightElevation === p.lightElevation &&
    l.enableShadows === p.enableShadows
  );
}

type Props = {
  loading: boolean;
  workBusy: boolean;
};

export function ViewportSettingsSidebar({ loading, workBusy }: Props) {
  const disabled = loading || workBusy;
  const [orthographic, setOrthographic] = useState(false);
  const [focalMm, setFocalMm] = useState(29);
  const [lighting, setLighting] = useState<SceneLightingPayload>(defaultLighting);

  const refreshFromNative = useCallback(() => {
    void invoke<boolean>("get_orthographic")
      .then(setOrthographic)
      .catch(() => {});
    void invoke<number>("get_focal_length_mm")
      .then(setFocalMm)
      .catch(() => {});
    void invoke<SceneLightingPayload>("get_scene_lighting")
      .then(setLighting)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshFromNative();
  }, [refreshFromNative]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void listen("voxelle-loaded", (ev) => {
      const p = ev.payload as { lighting?: SceneLightingPayload | null };
      if (p.lighting) {
        setLighting(p.lighting);
      } else {
        setLighting(defaultLighting());
      }
      void invoke<number>("get_focal_length_mm")
        .then(setFocalMm)
        .catch(() => {});
      void invoke<boolean>("get_orthographic")
        .then(setOrthographic)
        .catch(() => {});
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, []);

  const pushLighting = useCallback(async (next: SceneLightingPayload) => {
    setLighting(next);
    await invoke("set_scene_lighting", { args: next });
  }, []);

  const onFocalChange = useCallback(async (mm: number) => {
    setFocalMm(mm);
    await invoke("set_focal_length_mm", { mm });
  }, []);

  return (
    <div className="sidebar-viewport-settings" aria-label="Viewport">
      <div className="sidebar-section-label">Camera</div>
      {!orthographic ? (
        <div className="tool-options-range-label tool-options-range-with-value">
          <span>Zoom</span>
          <input
            type="range"
            min={15}
            max={200}
            step={1}
            list="focal-notches"
            value={Math.round(focalMm)}
            disabled={disabled}
            onChange={(e) => {
              void onFocalChange(Number(e.target.value));
            }}
          />
          <datalist id="focal-notches">
            {[24, 35, 50, 85, 105, 135, 200].map((mm) => (
              <option key={mm} value={mm} label={`${mm}mm`} />
            ))}
          </datalist>
          <span className="tool-options-range-value">{Math.round(focalMm)}mm</span>
        </div>
      ) : (
        <p className="sidebar-pane-hint sidebar-toolpanel-hint">
          Zoom applies in perspective mode only.
        </p>
      )}

      <div className="sidebar-section-label">Scene</div>
      <label className="tool-options-checkbox-row">
        <input
          type="checkbox"
          checked={lighting.enableSky}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({ ...lighting, enableSky: e.target.checked });
          }}
        />
        <span>Sky &amp; horizon</span>
      </label>
      <label className="tool-options-range-label">
        <span>Background</span>
        <input
          type="color"
          value={safeHex(lighting.backgroundColor, "#0a0b0e")}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({ ...lighting, backgroundColor: e.target.value });
          }}
        />
      </label>

      <div className="sidebar-viewport-preset-row">
        <span className="sidebar-section-label sidebar-section-label-inline">Light</span>
        <div className="sidebar-preset-toolbar" role="toolbar" aria-label="Lighting presets">
          {LIGHT_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={
                matchesPreset(lighting, p) ? "sidebar-preset-btn is-active" : "sidebar-preset-btn"
              }
              disabled={disabled}
              title={p.title}
              aria-label={p.title}
              aria-pressed={matchesPreset(lighting, p)}
              onClick={() => {
                void pushLighting({
                  ...lighting,
                  ambientIntensity: p.ambientIntensity,
                  sunlightIntensity: p.sunlightIntensity,
                  lightColor: p.lightColor,
                  lightAngle: p.lightAngle,
                  lightElevation: p.lightElevation,
                  enableShadows: p.enableShadows,
                });
              }}
            >
              <span className="sidebar-preset-emoji" aria-hidden>
                {p.id === "balanced"
                  ? "📐"
                  : p.id === "sunny"
                    ? "☀️"
                    : p.id === "cloudy"
                      ? "☁️"
                      : p.id === "incandescent"
                        ? "💡"
                        : p.id === "fluorescent"
                          ? "🔦"
                          : p.id === "moonlight"
                            ? "🌙"
                            : "🌑"}
              </span>
            </button>
          ))}
        </div>
      </div>

      <label className="tool-options-range-label tool-options-range-with-value">
        <span>Exposure</span>
        <input
          type="range"
          min={TONE_EXPOSURE_MIN}
          max={TONE_EXPOSURE_MAX}
          step={0.01}
          value={lighting.exposureEv}
          disabled={disabled}
          title={
            lighting.autoExposure
              ? "EV bias added on top of auto exposure (±5 EV)"
              : "Exposure in EV stops (neutral 0)"
          }
          onChange={(e) => {
            void pushLighting({
              ...lighting,
              exposureEv: Number(e.target.value),
            });
          }}
        />
        <span className="tool-options-range-value">
          {lighting.autoExposure ? "Bias " : ""}
          {lighting.exposureEv >= 0 ? "+" : ""}
          {lighting.exposureEv.toFixed(2)} EV
        </span>
      </label>
      <label className="tool-options-checkbox-row">
        <input
          type="checkbox"
          checked={lighting.autoExposure}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({ ...lighting, autoExposure: e.target.checked });
          }}
        />
        <span>Auto exposure</span>
      </label>

      <label className="tool-options-range-label tool-options-range-with-value">
        <span>Ambient</span>
        <input
          type="range"
          min={0}
          max={1.5}
          step={0.05}
          value={lighting.ambientIntensity}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({
              ...lighting,
              ambientIntensity: Number(e.target.value),
            });
          }}
        />
        <span className="tool-options-range-value">{lighting.ambientIntensity.toFixed(2)}</span>
      </label>
      <label className="tool-options-range-label tool-options-range-with-value">
        <span>Sunlight</span>
        <input
          type="range"
          min={0}
          max={20}
          step={0.05}
          value={lighting.sunlightIntensity}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({
              ...lighting,
              sunlightIntensity: Number(e.target.value),
            });
          }}
        />
        <span className="tool-options-range-value">{lighting.sunlightIntensity.toFixed(2)}</span>
      </label>
      <label className="tool-options-range-label">
        <span>Sun color</span>
        <input
          type="color"
          value={safeHex(lighting.lightColor, "#ffffff")}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({ ...lighting, lightColor: e.target.value });
          }}
        />
      </label>
      <label className="tool-options-range-label tool-options-range-with-value">
        <span>Angle</span>
        <input
          type="range"
          min={0}
          max={360}
          step={1}
          value={lighting.lightAngle}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({
              ...lighting,
              lightAngle: Number(e.target.value),
            });
          }}
        />
        <span className="tool-options-range-value">{Math.round(lighting.lightAngle)}°</span>
      </label>
      <div className="sidebar-sun-position-row">
        <button
          type="button"
          className="sidebar-sun-position-btn"
          disabled={disabled}
          title="Set angle and elevation to the current real sun position"
          aria-label="Apply real-time sun position"
          onClick={() => {
            const prefs = loadPreferences();
            const { azimuthDeg, altitudeDeg } = getSunPosition(
              new Date(),
              prefs.sunLocationLat,
              prefs.sunLocationLon,
            );
            void pushLighting({
              ...lighting,
              lightAngle: ((Math.round(azimuthDeg) % 360) + 360) % 360,
              lightElevation: Math.max(5, Math.min(90, Math.round(altitudeDeg))),
            });
          }}
        >
          <span aria-hidden>🕐</span>
          <span>Real sun</span>
        </button>
      </div>
      <label className="tool-options-range-label tool-options-range-with-value">
        <span>Elevation</span>
        <input
          type="range"
          min={5}
          max={90}
          step={1}
          value={lighting.lightElevation}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({
              ...lighting,
              lightElevation: Number(e.target.value),
            });
          }}
        />
        <span className="tool-options-range-value">{Math.round(lighting.lightElevation)}°</span>
      </label>
      <label className="tool-options-checkbox-row">
        <input
          type="checkbox"
          checked={lighting.enableShadows}
          disabled={disabled}
          onChange={(e) => {
            void pushLighting({ ...lighting, enableShadows: e.target.checked });
          }}
        />
        <span>Shadows</span>
      </label>
    </div>
  );
}
