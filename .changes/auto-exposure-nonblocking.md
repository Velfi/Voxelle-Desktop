---
category: fixed
---

Auto-exposure metering no longer stalls the render thread waiting for the GPU; luminance readback is now asynchronous, eliminating periodic frame hitches on scenes with auto-exposure enabled.
