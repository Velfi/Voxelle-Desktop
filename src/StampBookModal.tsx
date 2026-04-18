import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  deleteStamp,
  importStampsFromParsed,
  listStampsFromIndexedDb,
  normalizeStampTags,
  parseStampLibraryJson,
  putStamp,
  saveNewStamp,
  stampMatchesSearch,
  stampRecordsToLibraryJson,
  type StampBookEntryTuple,
  type StampBookRecord,
} from "./stampBookStorage";
import { useOverlayDismiss } from "./hooks/useOverlayDismiss";

type Tab = "manage" | "save" | "share";

type Props = {
  open: boolean;
  onClose: () => void;
  selectionCount: number;
  onUseStamp: (entries: StampBookEntryTuple[]) => void;
};

export function StampBookModal({ open, onClose, selectionCount, onUseStamp }: Props) {
  const [stamps, setStamps] = useState<StampBookRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<Tab>("manage");

  // Manage tab edit state
  const [editName, setEditName] = useState("");
  const [editTags, setEditTags] = useState("");

  // Save tab state
  const [saveName, setSaveName] = useState("");
  const [saveTags, setSaveTags] = useState("");
  const [saving, setSaving] = useState(false);

  // Share tab state
  const [importError, setImportError] = useState<string | null>(null);
  const [importSuccess, setImportSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadStamps = useCallback(async () => {
    const list = await listStampsFromIndexedDb();
    setStamps(list);
  }, []);

  useEffect(() => {
    if (open) {
      void loadStamps();
      setImportError(null);
      setImportSuccess(null);
    }
  }, [open, loadStamps]);

  const selected = stamps.find((s) => s.id === selectedId) ?? null;

  // Sync edit fields when selection changes
  useEffect(() => {
    if (selected) {
      setEditName(selected.name);
      setEditTags(normalizeStampTags(selected.tags).join(", "));
    }
  }, [selected?.id]);

  const filtered = stamps.filter((s) => stampMatchesSearch(s, search));

  function handleSelect(id: string) {
    setSelectedId(id);
    setTab("manage");
  }

  async function handleSaveName() {
    if (!selected || !editName.trim()) return;
    const updated = { ...selected, name: editName.trim() };
    await putStamp(updated);
    await loadStamps();
  }

  async function handleSaveTags() {
    if (!selected) return;
    const updated = {
      ...selected,
      tags: normalizeStampTags(editTags),
    };
    await putStamp(updated);
    await loadStamps();
  }

  async function handleDelete() {
    if (!selected) return;
    await deleteStamp(selected.id);
    setSelectedId(null);
    await loadStamps();
  }

  function handleUse() {
    if (!selected) return;
    onUseStamp(selected.entries);
    onClose();
  }

  async function handleSaveStamp() {
    const name = saveName.trim();
    if (!name || selectionCount === 0) return;
    setSaving(true);
    try {
      const entries = await invoke<[number, number, number, number, string][]>(
        "get_selection_as_stamp_entries",
      );
      if (!entries || entries.length === 0) return;
      const tupleEntries: StampBookEntryTuple[] = entries.map(([dx, dy, dz, color, mat]) => [
        dx,
        dy,
        dz,
        color,
        mat,
      ]);
      await saveNewStamp(name, tupleEntries, saveTags);
      setSaveName("");
      setSaveTags("");
      await loadStamps();
      setTab("manage");
    } finally {
      setSaving(false);
    }
  }

  function handleExport() {
    const toExport = search ? filtered : stamps;
    if (toExport.length === 0) return;
    const json = stampRecordsToLibraryJson(toExport);
    const blob = new Blob([json], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "stamps.voxelle-stamps.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImportError(null);
    setImportSuccess(null);
    try {
      const text = await file.text();
      const result = parseStampLibraryJson(text);
      if (!result.ok) {
        setImportError(result.error);
        return;
      }
      const created = await importStampsFromParsed(result.stamps);
      await loadStamps();
      setImportSuccess(`Imported ${created.length} stamp${created.length !== 1 ? "s" : ""}.`);
    } catch {
      setImportError("Failed to read file.");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  const dismiss = useOverlayDismiss(onClose);
  if (!open) return null;

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Stamp book"
      tabIndex={-1}
      {...dismiss}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <div className="modal modal--stamp-book">
        <div className="stamp-book-layout">
          {/* Left: stamp list */}
          <div className="stamp-book-list-col">
            <div className="stamp-book-list-header">
              <span className="stamp-book-title">Stamp Book</span>
              <input
                type="text"
                className="stamp-book-search"
                placeholder="Search…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Search stamps"
              />
            </div>
            <div className="stamp-book-list" role="listbox" aria-label="Stamps">
              {filtered.length === 0 ? (
                <div className="stamp-book-empty">
                  {stamps.length === 0
                    ? "No stamps yet. Save a selection to get started."
                    : "No stamps match your search."}
                </div>
              ) : (
                filtered.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    role="option"
                    aria-selected={s.id === selectedId}
                    className={
                      s.id === selectedId ? "stamp-book-item is-selected" : "stamp-book-item"
                    }
                    onClick={() => handleSelect(s.id)}
                  >
                    <div className="stamp-book-item-swatch">
                      {s.entries.length > 0 ? (
                        <svg
                          viewBox="0 0 10 10"
                          xmlns="http://www.w3.org/2000/svg"
                          aria-hidden="true"
                        >
                          {s.entries.slice(0, 64).map((e, i) => (
                            <rect
                              key={i}
                              x={(e[0] % 10) * 1}
                              y={(e[2] % 10) * 1}
                              width="1"
                              height="1"
                              fill={`#${(e[3] & 0xffffff).toString(16).padStart(6, "0")}`}
                            />
                          ))}
                        </svg>
                      ) : null}
                    </div>
                    <div className="stamp-book-item-info">
                      <span className="stamp-book-item-name">{s.name}</span>
                      {s.tags && s.tags.length > 0 ? (
                        <span className="stamp-book-item-tags">
                          {s.tags.slice(0, 3).join(", ")}
                        </span>
                      ) : null}
                      <span className="stamp-book-item-count">
                        {s.entries.length} voxel
                        {s.entries.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Right: tabbed panel */}
          <div className="stamp-book-panel-col">
            <div className="stamp-book-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={tab === "manage"}
                className={tab === "manage" ? "stamp-book-tab is-active" : "stamp-book-tab"}
                onClick={() => setTab("manage")}
              >
                Manage
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "save"}
                className={tab === "save" ? "stamp-book-tab is-active" : "stamp-book-tab"}
                onClick={() => setTab("save")}
              >
                Save
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "share"}
                className={tab === "share" ? "stamp-book-tab is-active" : "stamp-book-tab"}
                onClick={() => setTab("share")}
              >
                Share
              </button>
            </div>

            <div className="stamp-book-panel-body">
              {tab === "manage" && (
                <div className="stamp-book-manage">
                  {selected ? (
                    <>
                      <label className="modal-field">
                        Name
                        <div className="stamp-book-inline-edit">
                          <input
                            type="text"
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            onBlur={() => void handleSaveName()}
                            onKeyDown={(e) => e.key === "Enter" && void handleSaveName()}
                          />
                        </div>
                      </label>
                      <label className="modal-field">
                        Tags (comma-separated)
                        <div className="stamp-book-inline-edit">
                          <input
                            type="text"
                            value={editTags}
                            onChange={(e) => setEditTags(e.target.value)}
                            onBlur={() => void handleSaveTags()}
                            onKeyDown={(e) => e.key === "Enter" && void handleSaveTags()}
                            placeholder="e.g. arch, stone"
                          />
                        </div>
                      </label>
                      <div className="stamp-book-voxel-count">
                        {selected.entries.length} voxel
                        {selected.entries.length !== 1 ? "s" : ""}
                      </div>
                      <div className="stamp-book-manage-actions">
                        <button
                          type="button"
                          className="stamp-book-btn stamp-book-btn--primary"
                          onClick={handleUse}
                        >
                          Use stamp
                        </button>
                        <button
                          type="button"
                          className="stamp-book-btn stamp-book-btn--danger"
                          onClick={() => void handleDelete()}
                        >
                          Delete
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="stamp-book-empty">
                      Select a stamp from the list to manage it.
                    </div>
                  )}
                </div>
              )}

              {tab === "save" && (
                <div className="stamp-book-save">
                  <label className="modal-field">
                    Name
                    <input
                      type="text"
                      value={saveName}
                      onChange={(e) => setSaveName(e.target.value)}
                      placeholder="My stamp"
                      onKeyDown={(e) => e.key === "Enter" && void handleSaveStamp()}
                    />
                  </label>
                  <label className="modal-field">
                    Tags (comma-separated)
                    <input
                      type="text"
                      value={saveTags}
                      onChange={(e) => setSaveTags(e.target.value)}
                      placeholder="e.g. arch, stone"
                    />
                  </label>
                  {selectionCount === 0 && (
                    <div className="stamp-book-hint">
                      Make a selection in the viewport to save it as a stamp.
                    </div>
                  )}
                  <div className="stamp-book-save-actions">
                    <button
                      type="button"
                      className="stamp-book-btn stamp-book-btn--primary"
                      disabled={!saveName.trim() || selectionCount === 0 || saving}
                      onClick={() => void handleSaveStamp()}
                    >
                      {saving ? "Saving…" : "Save selection"}
                    </button>
                  </div>
                </div>
              )}

              {tab === "share" && (
                <div className="stamp-book-share">
                  <div className="stamp-book-share-section">
                    <div className="stamp-book-share-label">Export</div>
                    <p className="stamp-book-share-desc">
                      {search
                        ? `Export ${filtered.length} matching stamp${filtered.length !== 1 ? "s" : ""} as JSON.`
                        : `Export all ${stamps.length} stamp${stamps.length !== 1 ? "s" : ""} as JSON.`}
                    </p>
                    <button
                      type="button"
                      className="stamp-book-btn stamp-book-btn--secondary"
                      disabled={(search ? filtered : stamps).length === 0}
                      onClick={handleExport}
                    >
                      Export to file…
                    </button>
                  </div>
                  <div className="stamp-book-share-section">
                    <div className="stamp-book-share-label">Import</div>
                    <p className="stamp-book-share-desc">
                      Import stamps from a .voxelle-stamps.json file. Existing stamps are kept.
                    </p>
                    <button
                      type="button"
                      className="stamp-book-btn stamp-book-btn--secondary"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Import from file…
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".json,.voxelle-stamps.json"
                      style={{ display: "none" }}
                      onChange={(e) => void handleImportFile(e)}
                    />
                    {importError && <div className="stamp-book-error">{importError}</div>}
                    {importSuccess && <div className="stamp-book-success">{importSuccess}</div>}
                  </div>
                </div>
              )}
            </div>

            <div className="stamp-book-footer">
              <button
                type="button"
                className="stamp-book-btn stamp-book-btn--secondary"
                onClick={onClose}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
