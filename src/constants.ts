// ── Constants and small utilities extracted from App.tsx ──────────────

import type { LastSessionInfo, PaintColorDistrib, BrushShape, SculptBrushShapeUi } from "./types";

/** Desktop viewer: cap new-project grid edge length (web allows larger). */
export const MAX_GRID_SIZE = 256;

// ── localStorage key constants ─────────────────────────────────────
export const LS_RENDERING_MODE = "voxelleDesktopRenderingMode";
export const LS_SIDEBAR_EXPANDED = "voxelleSidebarExpanded";
export const LS_RIGHT_SIDEBAR_EXPANDED = "voxelleRightSidebarExpanded";
export const LS_TOOLS_FLOATING = "voxelleToolsFloating";
export const LS_TOOLS_FLOAT_POS = "voxelleToolsFloatPos";
export const LS_PALETTE_FLOATING = "voxellePaletteFloating";
export const LS_PALETTE_FLOAT_POS = "voxellePaletteFloatPos";
export const LS_PALETTE_FLOAT_SIZE = "voxellePaletteFloatSize";
/** `localStorage` = `"1"`: show JS vs Rust viewport cursor overlay (see `get_viewport_cursor_debug`). */
export const LS_VIEWPORT_CURSOR_DEBUG = "voxelleDebugViewportCursor";
export const LS_PAINT_COLOR_DISTRIB = "voxellePaintColorDistrib";

// ── Chat / ping constants ──────────────────────────────────────────
export const CHAT_TOAST_CAP = 5;
export const PING_HUD_MS = 7000;
export const PING_MP3_URL = `${import.meta.env.BASE_URL}ping.mp3`;

// ── Material options ───────────────────────────────────────────────
export const MATERIAL_OPTIONS: { id: string; label: string }[] = [
  { id: "plastic", label: "Plastic" },
  { id: "metal", label: "Metal" },
  { id: "rubber", label: "Rubber" },
  { id: "glass", label: "Glass" },
  { id: "water", label: "Water" },
  { id: "glow", label: "Glow" },
  { id: "velvet", label: "Velvet" },
  { id: "wax", label: "Wax" },
  { id: "holographic", label: "Holographic" },
];

/** Web `MAX_BRUSH_SIZE - 1` (slider index 0..63 -> display 1..64). */
export const SCULPT_BRUSH_MAX_INDEX = 63;

/** Must match Rust `ONGOING_UNSAVED_PROJECT_LABEL` (`get_last_session_info`). */
export const ONGOING_UNSAVED_PROJECT_LABEL = "An unsaved project";

// ── Paint color distribution defaults / persistence ────────────────
export const DEFAULT_PAINT_COLOR_DISTRIB: PaintColorDistrib = {
  mode: "whiteNoise",
  fbm: {
    octaves: 4,
    lacunarity: 2,
    persistence: 0.5,
    frequency: 0.15,
    noiseSeed: 0x12345678,
    quantized: false,
  },
  gradient: {
    kind: "linear",
    linearAxis: 1,
    scale: 0.08,
    phase: 0,
    radialCenter: [0, 0, 0],
    quantized: false,
  },
  dither: {
    orderedSize: 4,
    orderedStrength: 0.35,
    errorDiffusion: "none",
  },
};

export function loadPaintColorDistrib(): PaintColorDistrib {
  try {
    const s = localStorage.getItem(LS_PAINT_COLOR_DISTRIB);
    if (s) return { ...DEFAULT_PAINT_COLOR_DISTRIB, ...JSON.parse(s) };
  } catch {}
  return DEFAULT_PAINT_COLOR_DISTRIB;
}

// ── Utility functions ──────────────────────────────────────────────

/**
 * CSS layout viewport size for mapping `clientX`/`clientY` and layout fractions to the native surface.
 * Prefer `document.documentElement.clientWidth/Height` over `window.inner*` so the denominator matches
 * the pointer coordinate span (inner includes scrollbar gutter; client does not).
 */
export function layoutViewportCssSize(): { w: number; h: number } {
  const de = document.documentElement;
  const w = de.clientWidth || window.innerWidth;
  const h = de.clientHeight || window.innerHeight;
  return { w: Math.max(1, w), h: Math.max(1, h) };
}

/** Map texture-normalized nx, ny to position inside `.viewport` for debug overlay markers. */
export function viewportCursorOverlayPercent(
  nx: number,
  ny: number,
): { leftPct: number; topPct: number } {
  return { leftPct: nx * 100, topPct: ny * 100 };
}

export function playPingSound() {
  try {
    const a = new Audio(PING_MP3_URL);
    a.volume = 0.85;
    void a.play().catch(() => {});
  } catch {
    /* ignore */
  }
}

// ── Seagull "wah" speech sound ──────────────────────────────────────────────
const WAH_MP3_URL = `${import.meta.env.BASE_URL}wah.mp3`;
let wahAudioCtx: AudioContext | null = null;
let wahBuffer: AudioBuffer | null = null;
let wahBufferLoading = false;

async function ensureWahBuffer() {
  if (wahBuffer) return;
  if (wahBufferLoading) return;
  wahBufferLoading = true;
  try {
    if (!wahAudioCtx) wahAudioCtx = new AudioContext();
    const resp = await fetch(WAH_MP3_URL);
    const arrayBuf = await resp.arrayBuffer();
    wahBuffer = await wahAudioCtx.decodeAudioData(arrayBuf);
  } catch {
    wahBufferLoading = false;
  }
}

/** Play wah.mp3 several times at random pitches to simulate seagull speech. */
export function playSeagullSpeech() {
  void ensureWahBuffer().then(() => {
    if (!wahAudioCtx || !wahBuffer) return;
    const ctx = wahAudioCtx;
    if (ctx.state === "suspended") void ctx.resume();

    const count = 4 + Math.floor(Math.random() * 3); // 4-7 repetitions
    for (let i = 0; i < count; i++) {
      const src = ctx.createBufferSource();
      src.buffer = wahBuffer;
      // Random pitch between 1.0x and 1.6x
      src.playbackRate.value = 1.0 + Math.random() * 0.6;

      const gain = ctx.createGain();
      gain.gain.value = 0.55 + Math.random() * 0.25;
      src.connect(gain).connect(ctx.destination);

      // Stagger start times so they sound like syllables
      const delay = i * (0.08 + Math.random() * 0.1);
      src.start(ctx.currentTime + delay);
    }
  });
}

export function basename(path: string): string {
  const n = path.replace(/\\/g, "/");
  const i = n.lastIndexOf("/");
  return i >= 0 ? n.slice(i + 1) : n;
}

/** Maps low-level Tauri updater errors to text users can act on. */
export function userFacingUpdaterError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err ?? "unknown error");
  if (
    raw.includes("None of the fallback platforms") &&
    raw.includes("were found in the response")
  ) {
    let platform = "your computer";
    if (raw.includes("darwin-x86_64")) {
      platform = "Intel Macs";
    } else if (raw.includes("darwin-aarch64") || raw.includes("aarch64")) {
      platform = "Apple Silicon Macs";
    } else if (raw.includes("windows")) {
      platform = "Windows";
    } else if (raw.includes("linux")) {
      platform = "Linux";
    }
    return [
      `This release's update file doesn't include a build for ${platform}.`,
      "That often happens when a release only ships some platforms, or the update manifest (latest.json) wasn't merged correctly.",
      "",
      "What you can do: download the installer or archive that matches your system from the releases page and install it manually:",
      "https://github.com/Velfi/Voxelle-Desktop/releases",
    ].join("\n");
  }
  return raw.length > 0 ? raw : "Update failed (unknown error).";
}

/** Optional note when reopening (backup vs file). */
export function lastProjectReopenBlurb(info: LastSessionInfo): string | null {
  if (!info.lastDocumentPath) return null;
  if (info.lastDocumentPath === ONGOING_UNSAVED_PROJECT_LABEL && info.autosaveExists) {
    return "The project is an autosave and will be overwritten by your next autosave.";
  }
  if (!info.documentExists && info.autosaveExists) {
    return "Couldn't find the file — opened your backup instead.";
  }
  if (info.documentExists && info.autosaveExists && info.autosaveNewerThanDocument) {
    return "Backup is newer than the saved file.";
  }
  if (info.documentExists && info.autosaveExists && !info.autosaveNewerThanDocument) {
    return null;
  }
  return null;
}

export function sculptBrushShapeToRust(s: SculptBrushShapeUi): BrushShape {
  return s;
}
