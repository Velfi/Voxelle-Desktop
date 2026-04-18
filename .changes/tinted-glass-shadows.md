---
category: fixed
---

Glass and water now cast correctly coloured shadows and render in their true colour. In both the rasterizer and the path tracer, shadow rays were treating transparent media as either fully solid or fully invisible — so a red glass pane cast either a black shadow or no shadow at all, and its refraction whitened out the material colour. Shadow attenuation is now per-channel Beer-Lambert against each voxel's RGB (red glass transmits red, absorbs green and blue), and refraction uses the same physics, so red glass looks red instead of picking up the sky reflection.
