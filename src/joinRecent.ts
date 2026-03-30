const KEY = "voxelleCollabRecentJoinUrls";
const MAX = 5;

export function loadRecentJoinUrls(): string[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const data = JSON.parse(raw) as unknown;
    if (!Array.isArray(data)) return [];
    return data
      .filter((u): u is string => typeof u === "string")
      .map((u) => u.trim())
      .filter((u) => u.length > 0)
      .slice(0, MAX);
  } catch {
    return [];
  }
}

/** Most recent first, deduped, cap at 5. */
export function rememberJoinedUrl(url: string): void {
  if (typeof localStorage === "undefined") return;
  const t = url.trim();
  if (!t) return;
  try {
    const prev = loadRecentJoinUrls();
    const next = [t, ...prev.filter((u) => u !== t)].slice(0, MAX);
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
}
