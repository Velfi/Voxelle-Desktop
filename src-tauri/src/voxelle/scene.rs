//! Scene graph: hierarchical transforms for [`super::format::SceneObject`].

use super::format::SceneObject;
use glam::{Mat4, Quat, Vec3};

/// World matrix for `id`: root→leaf chain (parent × local).
pub fn object_world_matrix(objects: &[SceneObject], id: u32) -> Mat4 {
    let mut chain: Vec<&SceneObject> = Vec::new();
    let mut cur = Some(id);
    while let Some(cid) = cur {
        let Some(obj) = objects.iter().find(|o| o.id == cid) else {
            break;
        };
        chain.push(obj);
        cur = obj.parent_id;
    }
    chain.reverse();
    let mut m = Mat4::IDENTITY;
    for o in chain {
        let rot = Quat::from_xyzw(o.rotation[0], o.rotation[1], o.rotation[2], o.rotation[3]).normalize();
        let t = Vec3::from_array(o.translation);
        let s = Vec3::from_array(o.scale);
        m *= Mat4::from_scale_rotation_translation(s, rot, t);
    }
    m
}

pub fn is_object_visible(objects: &[SceneObject], id: u32) -> bool {
    objects
        .iter()
        .find(|o| o.id == id)
        .map(|o| o.visible)
        .unwrap_or(true)
}
