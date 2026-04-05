## 0.2.7 — 2026-04-05

### Added
- Bone generator: place joints on the model surface and connect them into an armature, then commit to voxelize the skeleton into a creature or character shape. Supports add, edit, and delete modes with IK drag for posing.
- Generators now respect active symmetry axes. Any generator edit (rocks, grass, rope, cloth, shapes, bones, etc.) is automatically mirrored across the selected X/Y/Z axes, matching the behaviour of manual voxel painting.
- Shape generator: place geometric primitives (cube, orb, cylinder, hollow cube, plane, circle) onto the model surface with configurable size and X/Y/Z rotation. The placement gizmo lets you reposition and rotate the shape before committing.
- New .voxelle V5 file format uses zstd compression instead of gzip for faster save and load times. Large files are also parsed in parallel on multi-core systems. V3 and V4 files continue to open normally.

### Changed
- Collaborative editing now sends voxel deltas as binary (bincode) frames instead of JSON text, reducing message size and improving sync performance in large edit operations.

### Fixed
- Undo history is now capped at a maximum number of entries, preventing unbounded memory growth during long editing sessions.

# Changelog

## 0.2.6 — 2026-04-04

### Added
- "Check for updates on startup" preference — the app now silently checks for updates at launch and prompts to install. Can be disabled in Preferences.
- Fragment-based changelog system — drop `.md` files in `.changes/` and they are compiled into `CHANGELOG.md` at release time.
- Added `logo_set_light_intensity` command to control the start-screen logo sun/direct light intensity.
- Ray tracing toggle button in the status bar for quick access without navigating menus.
- Surface glow illumination in the ray tracer — nearby glow voxels now cast soft colored light onto surrounding surfaces.

### Changed
- Glow voxels are now twice as bright in the ray tracer (4x → 8x emission multiplier).
- Glow emission in the greedy mesher now uses a spatial hash grid, significantly improving mesh build times for scenes with many glow voxels.
- Tuned material lighting: raised ambient contribution for plastic, rubber, metal, holographic, and glow materials; increased default sunlight intensity to 2.0.
- Logo explode effect now displaces whole voxels as rigid cubes instead of scattering individual faces. Uses naive (1×1 quad) meshing with per-voxel center passed to the shader.
- GitHub releases and the auto-updater now pull release notes from the changelog instead of using auto-generated notes.
- Ray tracing is now an independent toggle instead of a mutually exclusive rendering mode, so it can be combined with any mesh renderer.

### Fixed
- Collab peer avatars now center correctly on their voxel geometry instead of being offset toward the grid origin.
- Collab peer avatars now face the correct direction and render with proper normals.
- FPS counter no longer jitters the status bar layout as the number changes.
- Fixed sRGB-to-linear color conversion to use the proper piecewise curve instead of a gamma 2.2 approximation, in both the greedy mesher and the ray tracer.

### Removed
- Removed Piscina, Insecta, and Fauna generators (hooks, presets, tool options, and viewport wiring). They will be replaced by a better tool.

