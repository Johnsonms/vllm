# Status: Hopper node (role `hopper`)

The Hopper node writes this file. Newest entries at the bottom.

## 2026-11-28 — H1 ack
- H1 ack, running. Node: 8x H100 80GB HBM3 (SM90), all pairs NV18 (full NVLink mesh), driver 595.91.07 (CUDA 13.2 driver). No system CUDA toolkit `bin/` (no nvcc) on this node; relying on pip-provided CUDA libs.
- Building env on base `cfbc42099a` now; will start H2 directly after H1 is pushed.
