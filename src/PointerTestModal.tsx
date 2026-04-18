import { useCallback, useEffect, useRef, useState } from "react";
import { getCurrentWindow, LogicalPosition } from "@tauri-apps/api/window";
import { useOverlayDismiss } from "./hooks/useOverlayDismiss";

interface Props {
  open: boolean;
  onClose: () => void;
}

type LockStatus = "none" | "pending" | "active" | "failed";
type CaptureStatus = "none" | "active";

export function PointerTestModal({ open, onClose }: Props) {
  const lockTargetRef = useRef<HTMLDivElement | null>(null);
  const captureTargetRef = useRef<HTMLDivElement | null>(null);

  const [lockStatus, setLockStatus] = useState<LockStatus>("none");
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus>("none");
  const [capturedPointerId, setCapturedPointerId] = useState<number | null>(null);

  const [lockDx, setLockDx] = useState(0);
  const [lockDy, setLockDy] = useState(0);
  const [captureDx, setCaptureDx] = useState(0);
  const [captureDy, setCaptureDy] = useState(0);
  const [lockMoveCount, setLockMoveCount] = useState(0);
  const [captureMoveCount, setCaptureMoveCount] = useState(0);

  const [tauriGrabStatus, setTauriGrabStatus] = useState<"idle" | "active" | "failed">("idle");
  const [log, setLog] = useState<string[]>([]);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 30));
  }, []);

  // ── Pointer Lock ──────────────────────────────────────────────────────────

  const requestLock = useCallback(async () => {
    const el = lockTargetRef.current;
    if (!el) return;
    setLockStatus("pending");
    addLog("requestPointerLock() called…");
    try {
      await el.requestPointerLock();
      const locked = document.pointerLockElement === el;
      setLockStatus(locked ? "active" : "failed");
      addLog(locked ? "✓ Pointer lock ACTIVE" : "✗ Pointer lock not granted (element mismatch)");
    } catch (err) {
      setLockStatus("failed");
      addLog(`✗ requestPointerLock threw: ${String(err)}`);
    }
  }, [addLog]);

  const releaseLock = useCallback(() => {
    if (document.pointerLockElement) {
      document.exitPointerLock();
      addLog("exitPointerLock() called");
    } else {
      addLog("exitPointerLock() — nothing locked");
    }
    setLockStatus("none");
    setLockDx(0);
    setLockDy(0);
    setLockMoveCount(0);
  }, [addLog]);

  // ── Pointer Capture ───────────────────────────────────────────────────────

  const onCapturePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const el = captureTargetRef.current;
      if (!el) return;
      try {
        el.setPointerCapture(e.pointerId);
        setCaptureStatus("active");
        setCapturedPointerId(e.pointerId);
        setCaptureDx(0);
        setCaptureDy(0);
        setCaptureMoveCount(0);
        addLog(
          `✓ setPointerCapture(${e.pointerId}) — hasPointerCapture=${el.hasPointerCapture(e.pointerId)}`,
        );
      } catch (err) {
        addLog(`✗ setPointerCapture threw: ${String(err)}`);
      }
    },
    [addLog],
  );

  const onCapturePointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.movementX === 0 && e.movementY === 0) return;
    setCaptureDx((x) => x + e.movementX);
    setCaptureDy((y) => y + e.movementY);
    setCaptureMoveCount((n) => n + 1);
  }, []);

  const onCapturePointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const el = captureTargetRef.current;
      if (!el) return;
      try {
        el.releasePointerCapture(e.pointerId);
        addLog(`releasePointerCapture(${e.pointerId})`);
      } catch {
        /* */
      }
      setCaptureStatus("none");
      setCapturedPointerId(null);
    },
    [addLog],
  );

  const onCaptureLostCapture = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      addLog(`lostpointercapture — pointerId=${e.pointerId}`);
      setCaptureStatus("none");
      setCapturedPointerId(null);
    },
    [addLog],
  );

  // ── Tauri native grab ─────────────────────────────────────────────────────

  const testTauriGrab = useCallback(async () => {
    addLog("Testing Tauri setCursorGrab(true) + setCursorVisible(false)…");
    const w = getCurrentWindow();
    try {
      await w.setCursorGrab(true);
      await w.setCursorVisible(false);
      setTauriGrabStatus("active");
      addLog("✓ Tauri cursor grab active — click 'Release' to undo");
    } catch (err) {
      setTauriGrabStatus("failed");
      addLog(`✗ Tauri grab failed: ${String(err)}`);
    }
  }, [addLog]);

  const releaseTauriGrab = useCallback(async () => {
    const w = getCurrentWindow();
    try {
      await w.setCursorGrab(false);
      await w.setCursorVisible(true);
    } catch {
      /* */
    }
    setTauriGrabStatus("idle");
    addLog("Tauri cursor grab released");
  }, [addLog]);

  const testTauriCenter = useCallback(async () => {
    const w = getCurrentWindow();
    const vp = lockTargetRef.current;
    if (!vp) return;
    const r = vp.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    try {
      await w.setCursorPosition(new LogicalPosition(cx, cy));
      addLog(`✓ setCursorPosition(${Math.round(cx)}, ${Math.round(cy)})`);
    } catch (err) {
      addLog(`✗ setCursorPosition failed: ${String(err)}`);
    }
  }, [addLog]);

  // ── Global pointermove when locked ────────────────────────────────────────

  useEffect(() => {
    if (!open) return;
    const onMove = (e: PointerEvent) => {
      if (document.pointerLockElement === lockTargetRef.current) {
        if (e.movementX === 0 && e.movementY === 0) return;
        setLockDx((x) => x + e.movementX);
        setLockDy((y) => y + e.movementY);
        setLockMoveCount((n) => n + 1);
      }
    };
    document.addEventListener("pointermove", onMove, true);
    return () => document.removeEventListener("pointermove", onMove, true);
  }, [open]);

  // ── Pointer lock change ───────────────────────────────────────────────────

  useEffect(() => {
    if (!open) return;
    const onChange = () => {
      const locked = document.pointerLockElement === lockTargetRef.current;
      setLockStatus(locked ? "active" : "none");
      if (!locked) {
        addLog("pointerlockchange — lock lost");
      }
    };
    document.addEventListener("pointerlockchange", onChange);
    document.addEventListener("pointerlockerror", () => {
      setLockStatus("failed");
      addLog("✗ pointerlockerror event fired");
    });
    return () => {
      document.removeEventListener("pointerlockchange", onChange);
    };
  }, [open, addLog]);

  // ── Cleanup on close ─────────────────────────────────────────────────────

  useEffect(() => {
    if (!open) {
      if (document.pointerLockElement) document.exitPointerLock();
      void releaseTauriGrab();
    }
  }, [open, releaseTauriGrab]);

  const dismiss = useOverlayDismiss(onClose);
  if (!open) return null;

  const lockColor =
    lockStatus === "active"
      ? "#4caf50"
      : lockStatus === "failed"
        ? "#f44336"
        : lockStatus === "pending"
          ? "#ff9800"
          : "#888";
  const captureColor = captureStatus === "active" ? "#4caf50" : "#888";
  const tauriColor =
    tauriGrabStatus === "active" ? "#4caf50" : tauriGrabStatus === "failed" ? "#f44336" : "#888";

  return (
    <div className="modal-overlay" {...dismiss}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Pointer test"
        tabIndex={-1}
        style={{ minWidth: 520, maxWidth: 620 }}
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose();
        }}
      >
        <h2 style={{ marginTop: 0 }}>Pointer Capture / Lock Test</h2>

        {/* ── Row 1: Pointer Lock ── */}
        <section style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <strong>Pointer Lock</strong>
            <span style={{ color: lockColor, fontWeight: "bold", fontSize: 12 }}>
              ● {lockStatus.toUpperCase()}
            </span>
          </div>
          <div
            ref={lockTargetRef}
            style={{
              background: lockStatus === "active" ? "#1a3a1a" : "#1e1e2e",
              border: `2px solid ${lockColor}`,
              borderRadius: 6,
              padding: "12px 16px",
              cursor: lockStatus === "active" ? "none" : "crosshair",
              userSelect: "none",
              minHeight: 60,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 13,
              color: "#aaa",
            }}
            onClick={lockStatus === "none" || lockStatus === "failed" ? requestLock : undefined}
          >
            {lockStatus === "active" ? (
              <span style={{ color: "#4caf50" }}>
                Lock active — move mouse | moves: {lockMoveCount} | Σdx: {lockDx} Σdy: {lockDy}
              </span>
            ) : lockStatus === "pending" ? (
              <span style={{ color: "#ff9800" }}>Requesting…</span>
            ) : (
              <span>Click here to request pointer lock</span>
            )}
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
            <button type="button" onClick={() => void requestLock()}>
              Request lock
            </button>
            <button type="button" onClick={releaseLock}>
              Release
            </button>
            <span style={{ marginLeft: "auto", fontSize: 11, color: "#666", alignSelf: "center" }}>
              document.pointerLockElement === target:{" "}
              <strong>{String(document.pointerLockElement === lockTargetRef.current)}</strong>
            </span>
          </div>
        </section>

        {/* ── Row 2: Pointer Capture ── */}
        <section style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <strong>Pointer Capture</strong>
            <span style={{ color: captureColor, fontWeight: "bold", fontSize: 12 }}>
              ● {captureStatus === "active" ? `ACTIVE (id ${capturedPointerId ?? "?"})` : "NONE"}
            </span>
          </div>
          <div
            ref={captureTargetRef}
            style={{
              background: captureStatus === "active" ? "#1a2a3a" : "#1e1e2e",
              border: `2px solid ${captureColor}`,
              borderRadius: 6,
              padding: "12px 16px",
              cursor: "crosshair",
              userSelect: "none",
              minHeight: 60,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 13,
              color: "#aaa",
            }}
            onPointerDown={onCapturePointerDown}
            onPointerMove={onCapturePointerMove}
            onPointerUp={onCapturePointerUp}
            onLostPointerCapture={onCaptureLostCapture}
          >
            {captureStatus === "active" ? (
              <span style={{ color: "#4caf50" }}>
                Captured — drag freely | moves: {captureMoveCount} | Σdx: {captureDx} Σdy:{" "}
                {captureDy}
              </span>
            ) : (
              <span>Pointerdown here to capture — then drag outside</span>
            )}
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: "#666" }}>
            hasPointerCapture:{" "}
            <strong>
              {capturedPointerId !== null && captureTargetRef.current
                ? String(captureTargetRef.current.hasPointerCapture(capturedPointerId))
                : "n/a"}
            </strong>
          </div>
        </section>

        {/* ── Row 3: Tauri native ── */}
        <section style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <strong>Tauri setCursorGrab / setCursorPosition</strong>
            <span style={{ color: tauriColor, fontWeight: "bold", fontSize: 12 }}>
              ● {tauriGrabStatus.toUpperCase()}
            </span>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button type="button" onClick={() => void testTauriGrab()}>
              Grab + hide cursor
            </button>
            <button type="button" onClick={() => void releaseTauriGrab()}>
              Release grab
            </button>
            <button type="button" onClick={() => void testTauriCenter()}>
              Center cursor in box above
            </button>
          </div>
        </section>

        {/* ── Log ── */}
        <section>
          <strong style={{ fontSize: 12 }}>Log</strong>
          <div
            style={{
              background: "#111",
              borderRadius: 4,
              padding: "6px 8px",
              marginTop: 4,
              height: 120,
              overflowY: "auto",
              fontFamily: "monospace",
              fontSize: 11,
              color: "#ccc",
            }}
          >
            {log.length === 0 ? (
              <span style={{ color: "#555" }}>No events yet</span>
            ) : (
              log.map((line, i) => <div key={i}>{line}</div>)
            )}
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
            <button type="button" onClick={() => setLog([])}>
              Clear log
            </button>
            <button
              type="button"
              onClick={() => void navigator.clipboard.writeText(log.slice().reverse().join("\n"))}
            >
              Copy log
            </button>
            <button type="button" onClick={onClose} style={{ marginLeft: "auto" }}>
              Close
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
