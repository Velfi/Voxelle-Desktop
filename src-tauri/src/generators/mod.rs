//! Procedural generators (web parity). Face placement uses ray hit + inward normal.

mod grass_gen;
mod rocks;
mod rope_gen;
mod squishy_gen;
mod squishy_session;

pub use grass_gen::generator_grass_at_screen;
pub use rocks::generator_rocks_at_screen;
pub use rope_gen::generator_rope_between_screens;
pub use squishy_gen::squishy_metaball_at_screen;
pub use squishy_session::{
    pick_metaball_at_screen, squishy_add_ball_at_screen, squishy_commit_session, SquishyMode,
    SquishySession,
};
