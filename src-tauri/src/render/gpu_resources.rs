/// Number of mip levels in the bloom downsample/upsample pyramid.
pub(crate) const BLOOM_LEVELS: usize = 5;

/// Opaque mesh vertex: `vec3 pos, vec3 n, vec3 color, mat_kind, ao, vec3 emission_tint` → 14×`f32`.
pub(crate) const OPAQUE_VERTEX_STRIDE: u64 = 56;

/// Mascot/logo vertex: same as opaque + `vec3 voxel_center` → 17×`f32`.
pub(crate) const MASCOT_VERTEX_STRIDE: u64 = 68;

/// [`Maintain::Wait`] can starve other threads while the GPU drains; use a short [`Maintain::Poll`]
/// loop with yields during heavy mesh rebuild so the app stays responsive.
#[inline]
pub(crate) fn poll_device_yielding_until_queue_empty(device: &wgpu::Device) {
    loop {
        if device.poll(wgpu::Maintain::Poll).is_queue_empty() {
            break;
        }
        std::thread::yield_now();
    }
}

pub(crate) fn hdr_format() -> wgpu::TextureFormat {
    wgpu::TextureFormat::Rgba16Float
}

pub(crate) fn create_rt_accum_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt_accum"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_format(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

pub(crate) fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: OPAQUE_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 36,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 40,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 44,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Mascot/logo vertex layout: same as [`vertex_layout`] + `vec3 voxel_center` at location 6.
pub(crate) fn mascot_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: MASCOT_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 36,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 40,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 44,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 56,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Prototype vertex for instanced preview: position (vec3) + normal (vec3) = 24 bytes.
pub(crate) fn preview_proto_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 24, // 2 × vec3<f32>
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Per-instance data for instanced preview: model matrix (4×vec4) + color (vec3) + mat_kind (f32) = 80 bytes.
pub(crate) fn preview_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 80, // 4×16 + 12 + 4
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            // model_c0
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
            // model_c1
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            // model_c2
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            },
            // model_c3
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x4,
            },
            // inst_color
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32x3,
            },
            // inst_mat_kind
            wgpu::VertexAttribute {
                offset: 76,
                shader_location: 7,
                format: wgpu::VertexFormat::Float32,
            },
        ],
    }
}

pub(crate) fn vertex_layout_collab_lines() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: (3 + 3) * 4,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fs_entry: &'static str,
    targets: &[Option<wgpu::ColorTargetState>],
    depth_stencil: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fullscreen"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Read-only depth snapshot for SSR / OIT: main depth is a depth attachment while those passes
/// sample depth; WebGPU forbids overlapping attachment + shader read on the same subresource.
/// We copy into this texture (`COPY_DST` | `TEXTURE_BINDING` only) instead of manual barriers.
pub(crate) fn create_depth_snapshot(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_snapshot"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// OIT accumulation (Rgba16Float) + revealage (R16Float) textures for WBOIT.
pub(crate) fn create_oit_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let w = width.max(1);
    let h = height.max(1);
    let extent = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

    let accum_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("oit_accum"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    });
    let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let reveal_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("oit_revealage"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage,
        view_formats: &[],
    });
    let reveal_view = reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());

    (accum_tex, accum_view, reveal_tex, reveal_view)
}

/// SSR fullscreen pass output (Rgba16Float): rgb = reflected colour, a = confidence.
pub(crate) fn create_ssr_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssr_output"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Tonemapped sRGB output at **viewport** resolution before [`copy_texture_to_texture`] into the swapchain.
pub(crate) fn create_present_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let w = width.max(1);
    let h = height.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("present"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

pub(crate) fn create_shadow_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

pub(crate) fn create_screen_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    hdr_fmt: wgpu::TextureFormat,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let w = width.max(1);
    let h = height.max(1);
    let extent = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let color_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let hdr_opaque_usage = color_usage | wgpu::TextureUsages::COPY_SRC;
    let hdr_final_usage = color_usage | wgpu::TextureUsages::COPY_DST;
    let depth_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;

    let hdr_opaque_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr_opaque"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: hdr_opaque_usage,
        view_formats: &[],
    });
    let hdr_opaque_view = hdr_opaque_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: hdr_final_usage,
        view_formats: &[],
    });
    let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let nrm_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("normal"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: color_usage,
        view_formats: &[],
    });
    let nrm_view = nrm_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("main_depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: depth_usage,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bloom_a_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom_a"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: color_usage,
        view_formats: &[],
    });
    let bloom_a_view = bloom_a_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bloom_b_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom_b"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: hdr_fmt,
        usage: color_usage,
        view_formats: &[],
    });
    let bloom_b_view = bloom_b_tex.create_view(&wgpu::TextureViewDescriptor::default());

    (
        hdr_opaque_tex,
        hdr_opaque_view,
        hdr_tex,
        hdr_view,
        nrm_tex,
        nrm_view,
        depth_tex,
        depth_view,
        bloom_a_tex,
        bloom_a_view,
        bloom_b_tex,
        bloom_b_view,
    )
}

pub(crate) fn create_bloom_pyramid(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    hdr_fmt: wgpu::TextureFormat,
) -> (
    Vec<wgpu::Texture>,
    Vec<wgpu::TextureView>,
    Vec<wgpu::Texture>,
    Vec<wgpu::TextureView>,
) {
    let color_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let mut a_textures = Vec::with_capacity(BLOOM_LEVELS);
    let mut a_views = Vec::with_capacity(BLOOM_LEVELS);
    let mut b_textures = Vec::with_capacity(BLOOM_LEVELS);
    let mut b_views = Vec::with_capacity(BLOOM_LEVELS);
    for i in 0..BLOOM_LEVELS {
        let w = (width >> (i + 1)).max(1);
        let h = (height >> (i + 1)).max(1);
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("bloom_pyr_a_{i}")),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: hdr_fmt,
            usage: color_usage,
            view_formats: &[],
        });
        let a_view = a.create_view(&wgpu::TextureViewDescriptor::default());
        let b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("bloom_pyr_b_{i}")),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: hdr_fmt,
            usage: color_usage,
            view_formats: &[],
        });
        let b_view = b.create_view(&wgpu::TextureViewDescriptor::default());
        a_textures.push(a);
        a_views.push(a_view);
        b_textures.push(b);
        b_views.push(b_view);
    }
    (a_textures, a_views, b_textures, b_views)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_bloom_pyramid_bind_groups(
    device: &wgpu::Device,
    post_bloom_layout: &wgpu::BindGroupLayout,
    post_blur_layout: &wgpu::BindGroupLayout,
    bloom_a_view: &wgpu::TextureView,
    pyr_a_views: &[wgpu::TextureView],
    pyr_b_views: &[wgpu::TextureView],
    sampler_linear: &wgpu::Sampler,
    post_blur_buf: &wgpu::Buffer,
    post_blit_weight_buf: &wgpu::Buffer,
) -> (
    Vec<wgpu::BindGroup>,
    Vec<wgpu::BindGroup>,
    Vec<wgpu::BindGroup>,
    wgpu::BindGroup,
    Vec<wgpu::BindGroup>,
    Vec<wgpu::BindGroup>,
) {
    let blit_bg = |label: &str, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: post_bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler_linear),
                },
            ],
        })
    };

    // bind_blit_down[0] reads bloom_a (full-res extract); [i] reads pyr_a[i-1].
    let bind_blit_down = (0..BLOOM_LEVELS)
        .map(|i| {
            let src = if i == 0 {
                bloom_a_view
            } else {
                &pyr_a_views[i - 1]
            };
            blit_bg(&format!("blit_down_{i}"), src)
        })
        .collect();

    // bind_blit_up[i] reads pyr_a[i+1] for additive upsample into pyr_a[i].
    let bind_blit_up = (0..BLOOM_LEVELS - 1)
        .map(|i| blit_bg(&format!("blit_up_{i}"), &pyr_a_views[i + 1]))
        .collect();

    // bind_blit_up_weighted[i]: same source textures as bind_blit_up but using post_blur_layout
    // so fs_blit_weighted can read the per-level weight uniform before the additive blend.
    let bind_blit_up_weighted: Vec<wgpu::BindGroup> = (0..BLOOM_LEVELS - 1)
        .map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("blit_up_w_{i}")),
                layout: post_blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&pyr_a_views[i + 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: post_blit_weight_buf.as_entire_binding(),
                    },
                ],
            })
        })
        .collect();

    // Final replace-blit: pyr_a[0] → bloom_a.
    let bind_blit_final = blit_bg("blit_final", &pyr_a_views[0]);

    let blur_bg = |label: &str, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: post_blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_blur_buf.as_entire_binding(),
                },
            ],
        })
    };

    let bind_blur_pyr_h = (0..BLOOM_LEVELS)
        .map(|i| blur_bg(&format!("blur_pyr_h_{i}"), &pyr_a_views[i]))
        .collect();
    let bind_blur_pyr_v = (0..BLOOM_LEVELS)
        .map(|i| blur_bg(&format!("blur_pyr_v_{i}"), &pyr_b_views[i]))
        .collect();

    (
        bind_blit_down,
        bind_blit_up,
        bind_blit_up_weighted,
        bind_blit_final,
        bind_blur_pyr_h,
        bind_blur_pyr_v,
    )
}
