/** IndexedDB persistence for Voxelle stamp book (desktop). No thumbnail support. */

export type StampBookEntryTuple = [number, number, number, number, string?];

export type StampBookRecord = {
  id: string;
  name: string;
  order: number;
  entries: StampBookEntryTuple[];
  createdAt: number;
  tags?: string[];
};

const DB_NAME = "voxelle-stamp-book";
const STORE_NAME = "stamps";
const DB_VERSION = 1;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
  });
}

function closeOnComplete(
  db: IDBDatabase,
  tx: IDBTransaction,
  resolve: () => void,
  reject: (e: unknown) => void,
) {
  tx.oncomplete = () => {
    db.close();
    resolve();
  };
  tx.onerror = () => {
    db.close();
    reject(tx.error ?? new Error("IndexedDB transaction failed"));
  };
  tx.onabort = () => {
    db.close();
    reject(tx.error ?? new Error("IndexedDB transaction aborted"));
  };
}

export async function listStampsFromIndexedDb(): Promise<StampBookRecord[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).getAll();
    req.onsuccess = () => {
      db.close();
      const rows = (req.result ?? []) as StampBookRecord[];
      rows.sort((a, b) => a.order - b.order || a.createdAt - b.createdAt);
      resolve(rows);
    };
    req.onerror = () => {
      db.close();
      reject(req.error);
    };
  });
}

export async function putStamp(record: StampBookRecord): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(record);
    closeOnComplete(db, tx, resolve, reject);
  });
}

export async function deleteStamp(id: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(id);
    closeOnComplete(db, tx, resolve, reject);
  });
}

export async function getStampById(id: string): Promise<StampBookRecord | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(id);
    req.onsuccess = () => {
      db.close();
      resolve((req.result as StampBookRecord) ?? null);
    };
    req.onerror = () => {
      db.close();
      reject(req.error);
    };
  });
}

export async function replaceAllOrders(records: StampBookRecord[]): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    for (const r of records) store.put(r);
    closeOnComplete(db, tx, resolve, reject);
  });
}

const MAX_TAG_LEN = 64;

export function normalizeStampTags(raw: unknown): string[] {
  if (raw === undefined || raw === null) return [];
  if (Array.isArray(raw)) {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of raw) {
      const s = String(t).trim().toLowerCase().slice(0, MAX_TAG_LEN);
      if (s && !seen.has(s)) {
        seen.add(s);
        out.push(s);
      }
    }
    return out;
  }
  if (typeof raw === "string") {
    return normalizeStampTags(
      raw
        .split(/[,;]+/)
        .map((p) => p.trim())
        .filter(Boolean),
    );
  }
  return [];
}

export function stampMatchesSearch(record: StampBookRecord, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const name = record.name.toLowerCase();
  const tags = normalizeStampTags(record.tags);
  const words = q.split(/\s+/).filter(Boolean);
  return words.every((word) => name.includes(word) || tags.some((t) => t.includes(word)));
}

export type ParsedStampImport = {
  name: string;
  entries: StampBookEntryTuple[];
  tags: string[];
};

export function parseStampLibraryJson(
  text: string,
): { ok: true; stamps: ParsedStampImport[] } | { ok: false; error: string } {
  try {
    const data = JSON.parse(text) as {
      voxelleStampLibrary?: unknown;
      stamps?: unknown;
    };
    if (data?.voxelleStampLibrary !== 1 || !Array.isArray(data.stamps)) {
      return {
        ok: false,
        error: "Not a Voxelle stamp library (expected voxelleStampLibrary: 1).",
      };
    }
    const out: ParsedStampImport[] = [];
    for (const s of data.stamps as unknown[]) {
      if (!s || typeof s !== "object") continue;
      const obj = s as { name?: unknown; entries?: unknown; tags?: unknown };
      if (typeof obj.name !== "string" || !Array.isArray(obj.entries)) continue;
      const entries: StampBookEntryTuple[] = [];
      for (const row of obj.entries) {
        if (!Array.isArray(row) || row.length < 4) continue;
        const [dx, dy, dz, c] = row as number[];
        if (
          typeof dx !== "number" ||
          typeof dy !== "number" ||
          typeof dz !== "number" ||
          typeof c !== "number" ||
          !Number.isFinite(dx + dy + dz + c)
        )
          continue;
        const mat = row.length >= 5 && typeof row[4] === "string" ? row[4] : undefined;
        entries.push(
          mat !== undefined ? [dx, dy, dz, c & 0xffffff, mat] : [dx, dy, dz, c & 0xffffff],
        );
      }
      if (entries.length === 0) continue;
      out.push({
        name: (obj.name as string).trim() || "Stamp",
        entries,
        tags: normalizeStampTags(obj.tags),
      });
    }
    if (out.length === 0) return { ok: false, error: "No valid stamps found in file." };
    return { ok: true, stamps: out };
  } catch {
    return { ok: false, error: "Invalid JSON." };
  }
}

export function stampRecordsToLibraryJson(records: StampBookRecord[]): string {
  return JSON.stringify(
    {
      voxelleStampLibrary: 1 as const,
      stamps: records.map((r) => ({
        id: r.id,
        name: r.name,
        entries: r.entries,
        ...(normalizeStampTags(r.tags).length > 0 ? { tags: normalizeStampTags(r.tags) } : {}),
      })),
    },
    null,
    2,
  );
}

export async function saveNewStamp(
  name: string,
  entries: StampBookEntryTuple[],
  tagInput: string,
): Promise<StampBookRecord> {
  const list = await listStampsFromIndexedDb();
  const maxOrder = list.reduce((m, r) => Math.max(m, r.order), -1);
  const id =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `stamp-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
  const record: StampBookRecord = {
    id,
    name: name.trim(),
    order: maxOrder + 1,
    entries,
    createdAt: Date.now(),
    tags: normalizeStampTags(tagInput),
  };
  await putStamp(record);
  return record;
}

export async function importStampsFromParsed(
  stamps: ParsedStampImport[],
): Promise<StampBookRecord[]> {
  const list = await listStampsFromIndexedDb();
  let maxOrder = list.reduce((m, r) => Math.max(m, r.order), -1);
  const created: StampBookRecord[] = [];
  for (const p of stamps) {
    const id =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `stamp-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const record: StampBookRecord = {
      id,
      name: p.name,
      order: ++maxOrder,
      entries: p.entries,
      createdAt: Date.now(),
      tags: normalizeStampTags(p.tags),
    };
    await putStamp(record);
    created.push(record);
  }
  return created;
}
