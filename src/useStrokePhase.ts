/**
 * Reusable stroke-phase state machine.
 *
 * Many voxel tools follow the same lifecycle:
 *   drag → enter phased UI → adjust settings → commit (or cancel).
 *
 * `useStrokePhase` encapsulates the state + ref pair, phase list with
 * forward/back navigation, and keyboard (Escape / Enter) handling so that
 * each tool only declares its phase names and data shape.
 *
 * @example
 * ```ts
 * const cuboid = useStrokePhase<CuboidPhaseData>({
 *   phases: ["depth"],
 *   onCancel: () => { invoke("voxel_stroke_preview_reset"); },
 *   onCommit: (data) => { commitCuboidSolidAtScreen(data); },
 * });
 *
 * // pointer-up:
 * cuboid.enter("depth", { lineStart, endNorm });
 *
 * // later:
 * cuboid.cancel();   // or cuboid.commit();
 * ```
 */

import { useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StrokePhaseSnapshot<TData> {
  /** Current phase id (one of the ids passed to `phases`). */
  phase: string;
  /** Index into the `phases` array. */
  phaseIndex: number;
  /** Arbitrary per-tool payload carried through the phase lifecycle. */
  data: TData;
}

export interface UseStrokePhaseOpts<TData> {
  /**
   * Ordered list of phase ids.  `advance()` moves forward, `retreat()` moves
   * back.  Must contain at least one entry.
   */
  phases: string[];
  /** Called when the phase is cancelled (Escape, or explicit `cancel()`). */
  onCancel?: (snapshot: StrokePhaseSnapshot<TData> | null) => void;
  /** Called when the phase is committed (Enter, or explicit `commit()`). */
  onCommit?: (snapshot: StrokePhaseSnapshot<TData>) => void;
  /**
   * When true the hook registers a global keydown listener that maps
   * Escape → cancel and Enter → commit.  Defaults to **true**.
   */
  keyboard?: boolean;
}

export interface StrokePhaseHandle<TData> {
  // -- state -----------------------------------------------------------------

  /** Reactive state — triggers re-renders.  `null` when inactive. */
  readonly snapshot: StrokePhaseSnapshot<TData> | null;
  /** Ref mirror of `snapshot` — safe for use inside closures / callbacks. */
  readonly ref: React.RefObject<StrokePhaseSnapshot<TData> | null>;
  /** `true` when a phase is active (`snapshot !== null`). */
  readonly active: boolean;

  // -- transitions -----------------------------------------------------------

  /** Enter the machine at a given phase with initial data. */
  enter(phase: string, data: TData): void;
  /** Move to the next phase, optionally merging new data. */
  advance(patch?: Partial<TData>): void;
  /** Move to the previous phase, optionally merging new data. */
  retreat(patch?: Partial<TData>): void;
  /** Jump to an arbitrary phase by id, optionally merging data. */
  goTo(phase: string, patch?: Partial<TData>): void;
  /** Update data without changing the current phase. */
  update(patch: Partial<TData>): void;

  // -- terminal --------------------------------------------------------------

  /** Cancel the phase (calls `onCancel`, resets to null). */
  cancel(): void;
  /** Commit the phase (calls `onCommit`, resets to null). */
  commit(): void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useStrokePhase<TData>(opts: UseStrokePhaseOpts<TData>): StrokePhaseHandle<TData> {
  const { phases, onCancel, onCommit, keyboard = true } = opts;

  const [snapshot, setSnapshot] = useState<StrokePhaseSnapshot<TData> | null>(null);
  const ref = useRef<StrokePhaseSnapshot<TData> | null>(null);

  // Keep refs for callbacks so the keydown listener doesn't go stale.
  const onCancelRef = useRef(onCancel);
  onCancelRef.current = onCancel;
  const onCommitRef = useRef(onCommit);
  onCommitRef.current = onCommit;

  // -- helpers ---------------------------------------------------------------

  function set(next: StrokePhaseSnapshot<TData> | null) {
    ref.current = next;
    setSnapshot(next);
  }

  function phaseIndex(id: string): number {
    const idx = phases.indexOf(id);
    if (idx === -1) {
      console.warn(`[useStrokePhase] unknown phase "${id}"`);
      return 0;
    }
    return idx;
  }

  // -- transitions -----------------------------------------------------------

  function enter(phase: string, data: TData) {
    const idx = phaseIndex(phase);
    set({ phase, phaseIndex: idx, data });
  }

  function advance(patch?: Partial<TData>) {
    const cur = ref.current;
    if (!cur) return;
    const nextIdx = Math.min(cur.phaseIndex + 1, phases.length - 1);
    const data = patch ? { ...cur.data, ...patch } : cur.data;
    set({ phase: phases[nextIdx], phaseIndex: nextIdx, data });
  }

  function retreat(patch?: Partial<TData>) {
    const cur = ref.current;
    if (!cur) return;
    const prevIdx = Math.max(cur.phaseIndex - 1, 0);
    const data = patch ? { ...cur.data, ...patch } : cur.data;
    set({ phase: phases[prevIdx], phaseIndex: prevIdx, data });
  }

  function goTo(phase: string, patch?: Partial<TData>) {
    const cur = ref.current;
    if (!cur) return;
    const idx = phaseIndex(phase);
    const data = patch ? { ...cur.data, ...patch } : cur.data;
    set({ phase: phases[idx], phaseIndex: idx, data });
  }

  function update(patch: Partial<TData>) {
    const cur = ref.current;
    if (!cur) return;
    set({ ...cur, data: { ...cur.data, ...patch } });
  }

  // -- terminal --------------------------------------------------------------

  function cancel() {
    const cur = ref.current;
    set(null);
    onCancelRef.current?.(cur);
  }

  function commit() {
    const cur = ref.current;
    if (!cur) return;
    set(null);
    onCommitRef.current?.(cur);
  }

  // -- keyboard --------------------------------------------------------------

  useEffect(() => {
    if (!keyboard || !snapshot) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        cancel();
      } else if (e.key === "Enter" && !e.repeat) {
        e.preventDefault();
        commit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // Re-bind when snapshot presence changes (active ↔ inactive).
  }, [keyboard, snapshot != null]);

  // -- handle ----------------------------------------------------------------

  return {
    snapshot,
    ref,
    active: snapshot != null,
    enter,
    advance,
    retreat,
    goTo,
    update,
    cancel,
    commit,
  };
}
