//! Scene graph: hierarchical transforms for [`super::format::SceneObject`].

use std::collections::HashMap;

use super::format::{default_scene_objects, SceneObject, Voxel};
use glam::{Mat4, Quat, Vec3};

/// Voxels that should contribute to the opaque mesh given scene visibility.
///
/// Uses the same [`SceneObject`] slice resolution as [`crate::greedy_mesh::build_greedy_mesh`]: when
/// `objects` is empty, [`default_scene_objects`] applies. Visibility follows [`is_object_visible`]:
/// unknown `object_id` values are treated as visible (legacy voxels that do not match any row yet).
pub fn visible_voxels_for_meshing(voxels: &[Voxel], objects: &[SceneObject]) -> Vec<Voxel> {
    let default_objs = default_scene_objects();
    let objs: &[SceneObject] = if objects.is_empty() {
        default_objs.as_slice()
    } else {
        objects
    };
    voxels
        .iter()
        .copied()
        .filter(|v| is_object_visible(objs, v.object_id))
        .collect()
}

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
    use crate::voxelle::MaterialId;
    use glam::Vec3;

    #[test]
    fn visible_voxels_for_meshing_respects_hidden_and_is_object_visible() {
        let mut a = SceneObject::default();
        a.id = 1;
        a.visible = false;
        let mut b = SceneObject::default();
        b.id = 2;
        b.visible = true;
        let objects = vec![a, b];
        let voxels = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 0xff0000,
                material: MaterialId::Plastic,
                object_id: 1,
            },
            Voxel {
                x: 1,
                y: 0,
                z: 0,
                color: 0x00ff00,
                material: MaterialId::Plastic,
                object_id: 2,
            },
            Voxel {
                x: 2,
                y: 0,
                z: 0,
                color: 0x0000ff,
                material: MaterialId::Plastic,
                object_id: 99,
            },
        ];
        let got = visible_voxels_for_meshing(&voxels, &objects);
        // id 1 hidden; id 2 visible; id 99 unknown → is_object_visible defaults to true
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|v| v.object_id == 2));
        assert!(got.iter().any(|v| v.object_id == 99));
    }

    #[test]
    fn visible_voxels_for_meshing_keeps_legacy_voxels_when_object_id_not_in_list() {
        let mut o = SceneObject::default();
        o.id = 1;
        o.visible = true;
        let objects = vec![o];
        let voxels = vec![Voxel {
            x: 0,
            y: 0,
            z: 0,
            color: 0xff0000,
            material: MaterialId::Plastic,
            object_id: 0,
        }];
        let got = visible_voxels_for_meshing(&voxels, &objects);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].object_id, 0);
    }

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
