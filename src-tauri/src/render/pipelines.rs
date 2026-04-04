//! Pipeline and bind-group-layout builder functions extracted from [`super::WgpuViewer::new`].
//!
//! Each function creates one logical group of GPU pipelines, keeping the constructor small.

use super::gpu;
use super::gpu_resources::*;
use super::preview_hdr_blend;

// ── Bind group layouts ──────────────────────────────────────────────────────

pub(crate) struct BindGroupLayouts {
    pub scene_layout0: wgpu::BindGroupLayout,
    pub scene_layout1: wgpu::BindGroupLayout,
    pub shadow_vs_layout: wgpu::BindGroupLayout,
    pub post_bloom_layout: wgpu::BindGroupLayout,
    pub post_blur_layout: wgpu::BindGroupLayout,
    pub post_composite_layout: wgpu::BindGroupLayout,
}

pub(crate) fn create_bind_group_layouts(device: &wgpu::Device) -> BindGroupLayouts {
    let scene_layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene0"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    });

    let scene_layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene1"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // binding 2: depth_for_ssr — opaque scene depth for SSR ray marching
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // binding 3: samp_depth — non-filtering sampler for depth
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            // binding 4: ssr_opts — SSR parameters uniform
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let shadow_vs_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_vs"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let post_bloom_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_bloom"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let post_blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_blur"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let post_composite_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_composite"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // depth texture for world-space mood effects
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // GlobalState (storage) for camera matrices
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    BindGroupLayouts {
        scene_layout0,
        scene_layout1,
        shadow_vs_layout,
        post_bloom_layout,
        post_blur_layout,
        post_composite_layout,
    }
}

// ── Scene pipelines (opaque + preview variants) ─────────────────────────────

pub(crate) struct ScenePipelines {
    pub pipeline_opaque: wgpu::RenderPipeline,
    pub pipeline_preview_occluded: wgpu::RenderPipeline,
    pub pipeline_preview_front: wgpu::RenderPipeline,
    pub pipeline_preview_front_wire: wgpu::RenderPipeline,
    pub pipeline_preview_inst_occluded: wgpu::RenderPipeline,
    pub pipeline_preview_inst_front: wgpu::RenderPipeline,
    pub pipeline_preview_inst_front_wire: wgpu::RenderPipeline,
    pub pipeline_gen_preview_inst_front: wgpu::RenderPipeline,
    pub pipeline_gen_preview_inst_occluded: wgpu::RenderPipeline,
    pub pipeline_gen_preview_inst_front_wire: wgpu::RenderPipeline,
}

pub(crate) fn create_scene_pipelines(
    device: &wgpu::Device,
    scene_layout0: &wgpu::BindGroupLayout,
) -> ScenePipelines {
    let vf = hdr_format();
    let shader_scene = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scene"),
        source: wgpu::ShaderSource::Wgsl(gpu::scene::WGSL.into()),
    });

    let pl_opaque = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_opaque"),
        bind_group_layouts: &[scene_layout0],
        push_constant_ranges: &[],
    });

    let pipeline_opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("opaque"),
        layout: Some(&pl_opaque),
        vertex: wgpu::VertexState {
            module: &shader_scene,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_scene,
            entry_point: Some("fs_opaque_mrt"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: vf,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: vf,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let preview_targets = &[
        Some(wgpu::ColorTargetState {
            format: vf,
            blend: Some(preview_hdr_blend()),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: vf,
            blend: None,
            write_mask: wgpu::ColorWrites::empty(),
        }),
    ];

    let pipeline_preview_occluded =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview_occluded"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_occluded_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_preview_front =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview_front"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_front_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -0.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_preview_front_wire =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview_front_wire"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_front_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    // GPU-instanced preview pipelines
    let inst_bufs = &[preview_proto_vertex_layout(), preview_instance_layout()];

    let pipeline_preview_inst_occluded =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview_inst_occluded"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_preview_instanced"),
                buffers: inst_bufs,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_occluded_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_preview_inst_front =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview_inst_front"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_preview_instanced"),
                buffers: inst_bufs,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_front_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -0.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_preview_inst_front_wire =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview_inst_front_wire"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_preview_instanced"),
                buffers: inst_bufs,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_front_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    // Lit generator preview pipelines (opaque, self-shadowing)
    let pipeline_gen_preview_inst_occluded =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gen_preview_inst_occluded"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_preview_instanced"),
                buffers: inst_bufs,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_lit_occluded_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gen_preview_inst_front =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gen_preview_inst_front"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_preview_instanced"),
                buffers: inst_bufs,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_lit_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -0.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gen_preview_inst_front_wire =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gen_preview_inst_front_wire"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_scene,
                entry_point: Some("vs_preview_instanced"),
                buffers: inst_bufs,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_scene,
                entry_point: Some("fs_preview_lit_mrt"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    ScenePipelines {
        pipeline_opaque,
        pipeline_preview_occluded,
        pipeline_preview_front,
        pipeline_preview_front_wire,
        pipeline_preview_inst_occluded,
        pipeline_preview_inst_front,
        pipeline_preview_inst_front_wire,
        pipeline_gen_preview_inst_front,
        pipeline_gen_preview_inst_occluded,
        pipeline_gen_preview_inst_front_wire,
    }
}

// ── Overlay pipelines (collab lines, grid borders, gizmos) ──────────────────

pub(crate) struct OverlayPipelines {
    pub pipeline_collab_lines_occluded: wgpu::RenderPipeline,
    pub pipeline_collab_lines_front: wgpu::RenderPipeline,
    pub pipeline_grid_border_lines: wgpu::RenderPipeline,
    pub pipeline_gizmo_lines_front: wgpu::RenderPipeline,
    pub pipeline_gizmo_lines_occluded: wgpu::RenderPipeline,
    pub pipeline_gizmo_tris_front: wgpu::RenderPipeline,
    pub pipeline_gizmo_tris_occluded: wgpu::RenderPipeline,
    pub pipeline_gizmo_lines_always: wgpu::RenderPipeline,
    pub pipeline_gizmo_tris_always: wgpu::RenderPipeline,
}

pub(crate) fn create_overlay_pipelines(
    device: &wgpu::Device,
    scene_layout0: &wgpu::BindGroupLayout,
) -> OverlayPipelines {
    let vf = hdr_format();
    let shader_collab_lines = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("collab_peer_lines"),
        source: wgpu::ShaderSource::Wgsl(gpu::collab_peer_lines::WGSL.into()),
    });

    let pl_opaque = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_opaque"),
        bind_group_layouts: &[scene_layout0],
        push_constant_ranges: &[],
    });

    let preview_targets = &[
        Some(wgpu::ColorTargetState {
            format: vf,
            blend: Some(preview_hdr_blend()),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: vf,
            blend: None,
            write_mask: wgpu::ColorWrites::empty(),
        }),
    ];

    let pipeline_collab_lines_occluded =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("collab_lines_occluded"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_collab_line_occluded"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_collab_lines_front =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("collab_lines_front"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_collab_line_front"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -0.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_grid_border_lines =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid_border_lines"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_grid_border_line"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -0.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gizmo_lines_front =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_lines_front"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_gizmo_front"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -0.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gizmo_lines_occluded =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_lines_occluded"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_gizmo_occluded"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gizmo_tris_front =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_tris_front"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_gizmo_front"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -4,
                    slope_scale: -1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gizmo_tris_occluded =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_tris_occluded"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_gizmo_occluded"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gizmo_lines_always =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_lines_always"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_gizmo_front"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let pipeline_gizmo_tris_always =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_tris_always"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_collab_lines,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout_collab_lines()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_collab_lines,
                entry_point: Some("fs_gizmo_front"),
                targets: preview_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    OverlayPipelines {
        pipeline_collab_lines_occluded,
        pipeline_collab_lines_front,
        pipeline_grid_border_lines,
        pipeline_gizmo_lines_front,
        pipeline_gizmo_lines_occluded,
        pipeline_gizmo_tris_front,
        pipeline_gizmo_tris_occluded,
        pipeline_gizmo_lines_always,
        pipeline_gizmo_tris_always,
    }
}

// ── Avatar pipeline ─────────────────────────────────────────────────────────

pub(crate) struct AvatarPipeline {
    pub pipeline_avatar: wgpu::RenderPipeline,
    pub avatar_bind_layout: wgpu::BindGroupLayout,
}

pub(crate) fn create_avatar_pipeline(device: &wgpu::Device) -> AvatarPipeline {
    let vf = hdr_format();
    let shader_avatar = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("avatar"),
        source: wgpu::ShaderSource::Wgsl(gpu::avatar::WGSL.into()),
    });

    let avatar_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("avatar_uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let avatar_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("avatar"),
        bind_group_layouts: &[&avatar_bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline_avatar = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("avatar"),
        layout: Some(&avatar_pl_layout),
        vertex: wgpu::VertexState {
            module: &shader_avatar,
            entry_point: Some("vs_avatar"),
            buffers: &[vertex_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_avatar,
            entry_point: Some("fs_avatar"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: vf,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: vf,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    AvatarPipeline {
        pipeline_avatar,
        avatar_bind_layout,
    }
}

// ── Sky pipelines ───────────────────────────────────────────────────────────

pub(crate) struct SkyPipelines {
    pub pipeline_sky: wgpu::RenderPipeline,
    pub pipeline_start_screen_bg: wgpu::RenderPipeline,
}

pub(crate) fn create_sky_pipelines(
    device: &wgpu::Device,
    scene_layout0: &wgpu::BindGroupLayout,
) -> SkyPipelines {
    let vf = hdr_format();

    let pl_opaque = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_opaque"),
        bind_group_layouts: &[scene_layout0],
        push_constant_ranges: &[],
    });

    let mrt_targets = &[
        Some(wgpu::ColorTargetState {
            format: vf,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: vf,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];
    let sky_depth = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    let shader_sky = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky"),
        source: wgpu::ShaderSource::Wgsl(gpu::sky::WGSL.into()),
    });
    let pipeline_sky = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky"),
        layout: Some(&pl_opaque),
        vertex: wgpu::VertexState {
            module: &shader_sky,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_sky,
            entry_point: Some("fs_sky_mrt"),
            targets: mrt_targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: sky_depth.clone(),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let shader_start_screen_bg = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("start_screen_bg"),
        source: wgpu::ShaderSource::Wgsl(gpu::start_screen_bg::WGSL.into()),
    });
    let pipeline_start_screen_bg =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("start_screen_bg"),
            layout: Some(&pl_opaque),
            vertex: wgpu::VertexState {
                module: &shader_start_screen_bg,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_start_screen_bg,
                entry_point: Some("fs_start_screen_mrt"),
                targets: mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: sky_depth,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    SkyPipelines {
        pipeline_sky,
        pipeline_start_screen_bg,
    }
}

// ── OIT pipelines ───────────────────────────────────────────────────────────

pub(crate) struct OitPipelines {
    pub pipeline_oit_accum: wgpu::RenderPipeline,
    pub pipeline_oit_composite: wgpu::RenderPipeline,
    pub oit_composite_layout: wgpu::BindGroupLayout,
}

pub(crate) fn create_oit_pipelines(
    device: &wgpu::Device,
    scene_layout0: &wgpu::BindGroupLayout,
    scene_layout1: &wgpu::BindGroupLayout,
) -> OitPipelines {
    let vf = hdr_format();

    let pl_trans = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_trans"),
        bind_group_layouts: &[scene_layout0, scene_layout1],
        push_constant_ranges: &[],
    });

    let shader_scene = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scene"),
        source: wgpu::ShaderSource::Wgsl(gpu::scene::WGSL.into()),
    });

    let pipeline_oit_accum = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("oit_accum"),
        layout: Some(&pl_trans),
        vertex: wgpu::VertexState {
            module: &shader_scene,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_scene,
            entry_point: Some("fs_oit_accum"),
            targets: &[
                // Target 0: accumulation (additive blend)
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // Target 1: revealage (multiplicative: dst * (1 - src))
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::OneMinusSrc,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::OneMinusSrc,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: None, // Glass visible from both sides
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -2,
                slope_scale: -1.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let shader_oit_composite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("oit_composite"),
        source: wgpu::ShaderSource::Wgsl(gpu::oit_composite::WGSL.into()),
    });
    let oit_composite_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oit_composite"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    let pl_oit_composite = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_oit_composite"),
        bind_group_layouts: &[&oit_composite_layout],
        push_constant_ranges: &[],
    });
    let pipeline_oit_composite =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_composite"),
            layout: Some(&pl_oit_composite),
            vertex: wgpu::VertexState {
                module: &shader_oit_composite,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_oit_composite,
                entry_point: Some("fs_oit_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: vf,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    OitPipelines {
        pipeline_oit_accum,
        pipeline_oit_composite,
        oit_composite_layout,
    }
}

// ── Shadow pipeline ─────────────────────────────────────────────────────────

pub(crate) fn create_shadow_pipeline(
    device: &wgpu::Device,
    shadow_vs_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader_shadow = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shadow"),
        source: wgpu::ShaderSource::Wgsl(gpu::shadow::WGSL.into()),
    });
    let pl_shadow = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_shadow"),
        bind_group_layouts: &[shadow_vs_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow"),
        layout: Some(&pl_shadow),
        vertex: wgpu::VertexState {
            module: &shader_shadow,
            entry_point: Some("vs_shadow"),
            buffers: &[vertex_layout()],
            compilation_options: Default::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

// ── Post-processing pipelines ───────────────────────────────────────────────

pub(crate) struct PostPipelines {
    pub pipeline_bloom_extract: wgpu::RenderPipeline,
    pub pipeline_blur: wgpu::RenderPipeline,
    pub pipeline_blit: wgpu::RenderPipeline,
    pub pipeline_blit_weighted_add: wgpu::RenderPipeline,
    pub pipeline_composite: wgpu::RenderPipeline,
    /// Stored for pipeline rebuild on HDR toggle.
    pub shader_composite: wgpu::ShaderModule,
    /// Stored for pipeline rebuild on HDR toggle.
    pub composite_pipeline_layout: wgpu::PipelineLayout,
    pub pipeline_meter: wgpu::RenderPipeline,
}

pub(crate) fn create_post_pipelines(
    device: &wgpu::Device,
    post_bloom_layout: &wgpu::BindGroupLayout,
    post_blur_layout: &wgpu::BindGroupLayout,
    post_composite_layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> PostPipelines {
    let vf = hdr_format();

    let pl_bloom = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_bloom"),
        bind_group_layouts: &[post_bloom_layout],
        push_constant_ranges: &[],
    });
    let pl_blur = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_blur"),
        bind_group_layouts: &[post_blur_layout],
        push_constant_ranges: &[],
    });
    let pl_comp = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_comp"),
        bind_group_layouts: &[post_composite_layout],
        push_constant_ranges: &[],
    });

    let shader_bloom_ex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bloom_ex"),
        source: wgpu::ShaderSource::Wgsl(gpu::post_bloom_extract::WGSL.into()),
    });
    let shader_blur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blur"),
        source: wgpu::ShaderSource::Wgsl(gpu::post_blur::WGSL.into()),
    });
    let shader_composite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite"),
        source: wgpu::ShaderSource::Wgsl(gpu::post_composite::WGSL.into()),
    });
    let shader_meter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("meter_lum"),
        source: wgpu::ShaderSource::Wgsl(gpu::meter_luminance::WGSL.into()),
    });
    let shader_blit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit"),
        source: wgpu::ShaderSource::Wgsl(gpu::post_blit::WGSL.into()),
    });

    let hdr_target = &[Some(wgpu::ColorTargetState {
        format: vf,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline_bloom_extract =
        fullscreen_pipeline(device, &pl_blur, &shader_bloom_ex, "fs_bloom_extract", hdr_target, None);
    let pipeline_blur =
        fullscreen_pipeline(device, &pl_blur, &shader_blur, "fs_blur", hdr_target, None);
    let pipeline_blit =
        fullscreen_pipeline(device, &pl_bloom, &shader_blit, "fs_blit", hdr_target, None);

    let additive = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let pipeline_blit_weighted_add = fullscreen_pipeline(
        device,
        &pl_blur,
        &shader_blit,
        "fs_blit_weighted",
        &[Some(wgpu::ColorTargetState {
            format: vf,
            blend: Some(additive),
            write_mask: wgpu::ColorWrites::ALL,
        })],
        None,
    );

    let pipeline_composite = fullscreen_pipeline(
        device,
        &pl_comp,
        &shader_composite,
        "fs_composite",
        &[Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        None,
    );

    let pipeline_meter = fullscreen_pipeline(
        device,
        &pl_bloom,
        &shader_meter,
        "fs_meter",
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R32Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        None,
    );

    PostPipelines {
        pipeline_bloom_extract,
        pipeline_blur,
        pipeline_blit,
        pipeline_blit_weighted_add,
        pipeline_composite,
        shader_composite,
        composite_pipeline_layout: pl_comp,
        pipeline_meter,
    }
}

// ── SSR pipeline ────────────────────────────────────────────────────────────

pub(crate) struct SsrPipeline {
    pub pipeline_ssr_fullscreen: wgpu::RenderPipeline,
    pub ssr_layout: wgpu::BindGroupLayout,
}

pub(crate) fn create_ssr_pipeline(device: &wgpu::Device) -> SsrPipeline {
    let ssr_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ssr_fullscreen"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let pl_ssr = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl_ssr_fullscreen"),
        bind_group_layouts: &[&ssr_layout],
        push_constant_ranges: &[],
    });
    let shader_ssr = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("post_ssr"),
        source: wgpu::ShaderSource::Wgsl(gpu::post_ssr::WGSL.into()),
    });
    let pipeline_ssr_fullscreen = fullscreen_pipeline(
        device,
        &pl_ssr,
        &shader_ssr,
        "fs_ssr",
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        None,
    );

    SsrPipeline {
        pipeline_ssr_fullscreen,
        ssr_layout,
    }
}

// ── Raytrace pipeline ───────────────────────────────────────────────────────

pub(crate) struct RaytracePipeline {
    pub pipeline_raytrace: wgpu::RenderPipeline,
    pub rt_scene_layout: wgpu::BindGroupLayout,
    pub rt_accum_layout: wgpu::BindGroupLayout,
}

pub(crate) fn create_raytrace_pipeline(device: &wgpu::Device) -> RaytracePipeline {
    let vf = hdr_format();

    let rt_scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rt_scene"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let rt_accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rt_accum"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let rt_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rt_pl"),
        bind_group_layouts: &[&rt_scene_layout, &rt_accum_layout],
        push_constant_ranges: &[],
    });
    let shader_rt = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ray_trace"),
        source: wgpu::ShaderSource::Wgsl(gpu::ray_trace::WGSL.into()),
    });
    let pipeline_raytrace = fullscreen_pipeline(
        device,
        &rt_pl_layout,
        &shader_rt,
        "fs_trace",
        &[Some(wgpu::ColorTargetState {
            format: vf,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        None,
    );

    RaytracePipeline {
        pipeline_raytrace,
        rt_scene_layout,
        rt_accum_layout,
    }
}

// ── Mascot pipeline ─────────────────────────────────────────────────────────

pub(crate) struct MascotPipelineResult {
    pub mascot_pipeline: wgpu::RenderPipeline,
    pub mascot_bind_layout: wgpu::BindGroupLayout,
}

pub(crate) fn create_mascot_pipeline(
    device: &wgpu::Device,
    sdr_format: wgpu::TextureFormat,
) -> MascotPipelineResult {
    let mascot_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mascot_uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let mascot_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mascot"),
        bind_group_layouts: &[&mascot_bind_layout],
        push_constant_ranges: &[],
    });
    let shader_mascot = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mascot"),
        source: wgpu::ShaderSource::Wgsl(gpu::mascot::WGSL.into()),
    });
    let mascot_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mascot"),
        layout: Some(&mascot_pl_layout),
        vertex: wgpu::VertexState {
            module: &shader_mascot,
            entry_point: Some("vs_mascot"),
            buffers: &[vertex_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_mascot,
            entry_point: Some("fs_mascot"),
            targets: &[Some(wgpu::ColorTargetState {
                format: sdr_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    MascotPipelineResult {
        mascot_pipeline,
        mascot_bind_layout,
    }
}

// ── Speech bubble pipeline ──────────────────────────────────────────────────

pub(crate) struct SpeechBubblePipelineResult {
    pub speech_bubble_pipeline: wgpu::RenderPipeline,
    pub speech_bubble_bind_layout: wgpu::BindGroupLayout,
}

pub(crate) fn create_speech_bubble_pipeline(
    device: &wgpu::Device,
    sdr_format: wgpu::TextureFormat,
) -> SpeechBubblePipelineResult {
    let speech_bubble_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("speech_bubble_uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let speech_bubble_pl_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("speech_bubble"),
            bind_group_layouts: &[&speech_bubble_bind_layout],
            push_constant_ranges: &[],
        });
    let shader_speech_bubble = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("speech_bubble"),
        source: wgpu::ShaderSource::Wgsl(gpu::speech_bubble::WGSL.into()),
    });
    let speech_bubble_pipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("speech_bubble"),
            layout: Some(&speech_bubble_pl_layout),
            vertex: wgpu::VertexState {
                module: &shader_speech_bubble,
                entry_point: Some("vs_bubble"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_speech_bubble,
                entry_point: Some("fs_bubble"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: sdr_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    SpeechBubblePipelineResult {
        speech_bubble_pipeline,
        speech_bubble_bind_layout,
    }
}
