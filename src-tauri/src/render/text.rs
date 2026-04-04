//! Text rendering: GPU peer labels, ping labels, gizmo delta labels.

use super::*;

/// Screen-space peer label for GPU text rendering.
pub struct GpuPeerLabel {
    pub name: String,
    pub color_rgb: u32,
    /// Pixel X in viewport space.
    pub x: f32,
    /// Pixel Y in viewport space.
    pub y: f32,
}

impl WgpuViewer {
    /// Replace the set of peer labels to render as GPU text this frame.
    pub fn upload_peer_labels(&mut self, labels: Vec<GpuPeerLabel>) {
        self.peer_label_data = labels;
    }

    pub fn clear_peer_labels(&mut self) {
        self.peer_label_data.clear();
    }

    pub fn upload_ping_label(&mut self, label: GpuPeerLabel) {
        self.ping_label_data = Some(label);
    }

    pub fn clear_ping_label(&mut self) {
        self.ping_label_data = None;
    }

    pub fn upload_gizmo_delta_label(&mut self, label: Option<GpuPeerLabel>) {
        self.gizmo_delta_label = label;
    }
}
