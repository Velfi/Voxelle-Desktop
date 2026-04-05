---
category: changed
---

Fly and walk camera navigation now bundles mouse-look input with movement in a single IPC call per frame instead of two, reducing input latency at high frame rates. On macOS, raw mouse deltas are now read directly via CGGetLastMouseDelta for smoother pointer-lock behavior.
