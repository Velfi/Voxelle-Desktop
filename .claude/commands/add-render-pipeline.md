Add a new wgpu render pipeline to Voxelle Desktop. Pipeline to add: $ARGUMENTS

A pipeline touches four places in `src-tauri/src/render/mod.rs`. Miss any one and the build will fail or the pipeline will be silently unused.

## Step 1 — Add fields to the `WgpuViewer` struct

Find the struct definition (search for `struct WgpuViewer`). Add one or more fields near related pipelines:

```rust
/// Brief description of what this pipeline renders.
pipeline_my_thing_front: wgpu::RenderPipeline,
// add _occluded partner if this needs a "behind geometry" ghost pass
pipeline_my_thing_occluded: wgpu::RenderPipeline,
```

If the pipeline has associated runtime state (e.g. a flag like `gizmo_on_top: bool`), add that here too.

## Step 2 — Create the pipeline descriptor

Find the constructor (`WgpuViewer::new` or `WgpuViewer::create`). Add the `device.create_render_pipeline(...)` call near the existing related pipelines. Start by copying the nearest similar pipeline and adjusting:

Key fields to customise:
- `label` — unique debug name (`Some("my_thing_front")`)
- `entry_point` — fragment shader entry point (e.g. `"fs_gizmo_front"`, `"fs_gizmo_occluded"`)
- `depth_compare` — `LessEqual` for front pass, `Greater` for occluded pass, `Always` to ignore depth
- `depth_write_enabled` — almost always `false` for overlay geometry
- `bias` — negative for front (pushes forward), positive for occluded (pulls back), `default()` for Always
- `topology` — `TriangleList` for solid geometry, `LineList` for wireframes

Common shader modules already in use: `shader_collab_lines` (coloured line/tri verts), `shader_preview` (preview cubes). Check which fragment entry points are available in the WGSL file before inventing new ones.

## Step 3 — Store in the struct init

In the same constructor, find the `Ok(Self { ... })` or `WgpuViewer { ... }` block. Add the new field(s):

```rust
pipeline_my_thing_front,
pipeline_my_thing_occluded,
// runtime state with non-default init:
my_flag: true,
```

Rust struct shorthand (`pipeline_my_thing_front,`) works when the local variable name matches the field name — use it.

## Step 4 — Call it in a draw method

Either add to an existing `draw_*` method or create a new one following the `draw_gizmo` pattern:

```rust
fn draw_my_thing(&self, pass: &mut wgpu::RenderPass<'_>) {
    pass.set_bind_group(0, &self.bind_scene_opaque, &[]);
    if let Some(ref vb) = self.my_vertex_buffer {
        if self.my_vertex_count >= 3 {
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_pipeline(&self.pipeline_my_thing_occluded);
            pass.draw(0..self.my_vertex_count, 0..1);
            pass.set_pipeline(&self.pipeline_my_thing_front);
            pass.draw(0..self.my_vertex_count, 0..1);
        }
    }
}
```

Then call the draw method from the main `render` function at the right point in the pass order (after opaque geometry, before or after transparency depending on blend mode).

## Checklist before finishing

- [ ] Struct field(s) added to `WgpuViewer`
- [ ] Pipeline descriptor(s) created in the constructor
- [ ] Field(s) stored in the struct init block
- [ ] Draw method calls the pipeline
- [ ] Draw method is called from the render loop at the correct pass order
- [ ] Build passes: `cargo build --manifest-path src-tauri/Cargo.toml`
