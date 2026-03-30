//! Scene graph: hierarchical transforms for [`super::format::SceneObject`].

use std::collections::HashMap;

use super::format::SceneObject;
use glam::{Mat4, Quat, Vec3};

/// Build `id → SceneObject` for O(1) parent-chain walks (avoids repeated linear scans).
fn objects_by_id(objects: &[SceneObject]) -> HashMap<u32, &SceneObject> {
    objects.iter().map(|o| (o.id, o)).collect()
}

/// World matrix for `id`: root→leaf chain (parent × local). Uses one map lookup per hierarchy level.
pub fn world_matrix_for_id(objects_by_id: &HashMap<u32, &SceneObject>, id: u32) -> Mat4 {
    let mut chain: Vec<&SceneObject> = Vec::new();
    let mut cur = Some(id);
    while let Some(cid) = cur {
        let Some(obj) = objects_by_id.get(&cid) else {
            break;
        };
        chain.push(*obj);
        cur = obj.parent_id;
    }
    chain.reverse();
    let mut m = Mat4::IDENTITY;
    for o in chain {
        let rot =
            Quat::from_xyzw(o.rotation[0], o.rotation[1], o.rotation[2], o.rotation[3]).normalize();
        let t = Vec3::from_array(o.translation);
        let s = Vec3::from_array(o.scale);
        m *= Mat4::from_scale_rotation_translation(s, rot, t);
    }
    m
}

/// World matrix for `id`: root→leaf chain (parent × local).
pub fn object_world_matrix(objects: &[SceneObject], id: u32) -> Mat4 {
    let by_id = objects_by_id(objects);
    world_matrix_for_id(&by_id, id)
}

/// One world matrix per object in `objects` (for voxel iteration: O(1) lookup per voxel vs rebuilding chains).
pub fn object_world_matrices_by_id(objects: &[SceneObject]) -> HashMap<u32, Mat4> {
    let by_id = objects_by_id(objects);
    let mut out = HashMap::with_capacity(objects.len());
    for o in objects {
        out.insert(o.id, world_matrix_for_id(&by_id, o.id));
    }
    out
}

pub fn object_visibility_by_id(objects: &[SceneObject]) -> HashMap<u32, bool> {
    objects.iter().map(|o| (o.id, o.visible)).collect()
}

pub fn is_object_visible(objects: &[SceneObject], id: u32) -> bool {
    objects
        .iter()
        .find(|o| o.id == id)
        .map(|o| o.visible)
        .unwrap_or(true)
}

/// True when every object is identity, so voxel integer coordinates match world bounds used for incremental edits.
pub fn scene_objects_identity_for_bounds_fast_path(objects: &[SceneObject]) -> bool {
    objects.iter().all(|o| {
        o.translation == [0.0, 0.0, 0.0]
            && o.rotation[0].abs() + o.rotation[1].abs() + o.rotation[2].abs() <= 1e-5
            && o.rotation[3] >= 0.9999
            && o.scale == [1.0, 1.0, 1.0]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn matrices_by_id_matches_legacy_per_id() {
        let mut a = SceneObject::default();
        a.id = 0;
        let mut b = SceneObject::default();
        b.id = 1;
        b.parent_id = Some(0);
        b.translation = [1.0, 0.0, 0.0];
        let objects = vec![a, b];
        let map = object_world_matrices_by_id(&objects);
        for o in &objects {
            let expected = object_world_matrix(&objects, o.id);
            let got = *map.get(&o.id).expect("id in map");
            let p = Vec3::new(1.0, 2.0, 3.0);
            assert_eq!(
                expected.transform_point3(p),
                got.transform_point3(p),
                "object {}",
                o.id
            );
        }
    }
}
