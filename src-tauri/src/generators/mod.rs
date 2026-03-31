//! Procedural generators (web parity). Face placement uses ray hit + inward normal.

mod ashlar_gen;
mod cloth_gen;
mod fauna_gen;
mod flora_gen;
mod grass_gen;
mod insecta_gen;
mod piscina_gen;
mod rocks;
mod roof_gen;
mod rope_gen;
mod squishy_gen;
mod squishy_gizmo;
mod squishy_pick_rings;
mod squishy_session;

pub use ashlar_gen::generator_ashlar_at_screen;
pub use cloth_gen::{generator_cloth_from_pins, preview_cloth_voxels, ClothSimOptions};
pub use fauna_gen::generator_fauna_at_screen;
pub use flora_gen::generator_flora_at_screen;
pub use grass_gen::{generator_grass_at_screen, preview_grass_at_screen};
pub use insecta_gen::generator_insecta_at_screen;
pub use piscina_gen::generator_piscina_at_screen;
pub use rocks::{generator_rocks_at_screen, preview_rock_at_screen};
pub use roof_gen::generate_roof_from_pins;
pub use rope_gen::{generator_rope_between_screens, preview_rope_voxels_between_screens};
pub use squishy_gen::squishy_metaball_at_screen;
pub use squishy_gizmo::{
    append_squishy_gizmo_wire, pick_squishy_gizmo_handle, squishy_gizmo_apply_drag,
    squishy_gizmo_begin_drag, SquishyGizmoDrag,
};
pub use squishy_pick_rings::append_squishy_metaball_pick_rings;
pub use squishy_session::{
    pick_metaball_at_screen, squishy_add_ball_at_screen, squishy_commit_session,
    voxel_coords_for_session_with_limit, Metaball, SquishyMode, SquishySession,
};
