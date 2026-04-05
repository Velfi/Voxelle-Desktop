# Z-Index Reference

All z-index layers used in the UI, grouped into tiers.
When adding new layers, pick a value within the appropriate tier.

## Tiers

### Tier 4 — Top-level floating UI (200–1000)

| z-index | Selector                  | Description               |
| ------: | ------------------------- | ------------------------- |
|    1000 | `.floating-palette-panel` | Draggable color palette   |
|     200 | `.modal-overlay`          | Preferences / modal scrim |

### Tier 3 — Panels & overlays (40–100)

| z-index | Selector                                                | Description                         |
| ------: | ------------------------------------------------------- | ----------------------------------- |
|     100 | `.tool-options-panel`                                   | Tool options (bottom-left floating) |
|      50 | `.app-sidebar.is-floating`                              | Floating tools sidebar              |
|      42 | `.app-sidebar-right.is-collapsed .sidebar-header-right` | Right sidebar toggle (collapsed)    |
|      40 | top-center banner (line ~1639)                          | Top-center notification banner      |

### Tier 2 — Secondary chrome (5–20)

| z-index | Selector                        | Description                          |
| ------: | ------------------------------- | ------------------------------------ |
|      20 | `.collab-join-progress-overlay` | Collab join progress                 |
|      10 | modal header bar (line ~1275)   | Modal sticky header                  |
|       6 | chat send button area (~2193)   | Chat send button                     |
|       5 | viewport overlay buttons        | Top-right buttons, viewport overlays |

### Tier 1 — Base chrome (1–3)

| z-index | Selector                        | Description                    |
| ------: | ------------------------------- | ------------------------------ |
|       3 | `.app-sidebar` (docked)         | Left tools sidebar             |
|       3 | `.app-footer`                   | Status bar                     |
|       3 | viewport crosshair overlay      | Crosshair overlay              |
|       2 | various inline elements         | Selected swatches, toast items |
|       1 | `.sidebar-palette-swatch:hover` | Palette swatch hover ring      |
