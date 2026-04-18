---
category: fixed
---

Rotating a selection whose bounding box has an even width, height, or depth no longer splits or drops voxels. Rotation now uses a consistent lattice-snapping rule in doubled-integer arithmetic, so a quarter turn lands every voxel on exactly one destination cell and the selection stays contiguous, even when rotating an even-sized axis onto an odd-sized one.
