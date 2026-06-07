# vlm_ggen_interp — Documentation

Pointcloud + grasp visualization pipeline for a multi-camera RealSense setup with a Franka Panda arm. Renders AnyGrasp results overlaid on real camera images using calibrated extrinsics.

---

## File Overview

### `viz.py` — Shared rendering primitives

Core library imported by all other scripts. Provides loaders, geometry builders, and three render backends.

**Key functions:**

| Function | Description |
|---|---|
| `load_npz(path)` | Load a pointcloud from `.npz` with `points` + `colors` arrays |
| `load_grasps(path, top_k=None)` | Load grasps JSON, sorted by score descending |
| `grasp_to_T(g)` | Extract 4×4 pose matrix from a grasp dict |
| `make_gripper(T, width, depth, height)` | Build parallel-jaw gripper mesh (3 TriangleMesh: 2 red fingers + blue base) |
| `make_robot_base_marker(T_world_from_base)` | Coordinate frame + orange sphere at robot base origin |
| `render_to_array(geoms, w, h, K, T_cam)` | Render world-frame geometries → `(H, W, 3)` uint8 array |
| `render_and_composite(geoms, photo_path, w, h, K, T_cam)` | Render and composite over a real photo; non-black pixels replace photo |
| `render_to_file(geoms, path, w, h, K, T_cam)` | Render and save to PNG |
| `render_interactive(geoms)` | Open blocking interactive Open3D viewer |

**Example:**
```python
from viz import load_npz, load_grasps, make_gripper, grasp_to_T, render_and_composite
import numpy as np, json

pcd = load_npz("anygrasp_observation_only.npz")
grasps = load_grasps("anygrasp_results/grasps.json", top_k=5)
cam = json.load(open("Calibration_results/extrinsic_results/realsense_0_extrinsic.json"))

geoms = [pcd]
for g in grasps:
    geoms += make_gripper(grasp_to_T(g), g["width"], g["depth"], g["height"])

K = np.array(cam["K"])
T = np.array(cam["T_cam_from_world"])
w, h = cam["image_size"]

composite = render_and_composite(geoms, "Rope_1/color/0/0.png", w, h, K, T)
```

---

### `viz_with_arm.py` — Franka Panda arm visualization

Forward kinematics from standard DH parameters → Open3D stick-figure (spheres at joints, cylinders for links). Joint angles default to the ready pose since capture-time angles are unavailable.

**Key exports:**

| Symbol | Description |
|---|---|
| `PANDA_HOME_Q` | Default ready-pose joint angles `[0, -π/4, 0, -3π/4, 0, π/2, π/4]` |
| `franka_fk(q=None)` | Returns 9 × (4×4) link-frame transforms in base frame |
| `make_franka_geometry(T_world_from_base, q=None)` | Returns list of TriangleMesh in world frame |

**Example:**
```python
from viz_with_arm import make_franka_geometry, PANDA_HOME_Q
import json, numpy as np

calib = json.load(open("Calibration_results/robot_calib_results/robot_calib_result.json"))
T = np.array(calib["T_world_from_base"])

arm_geoms = make_franka_geometry(T)                   # ready pose
arm_geoms = make_franka_geometry(T, q=PANDA_HOME_Q)   # same thing, explicit
```

---

### `pov_viz.py` — **Recommended** multi-camera renderer

Fixes the double-transform bug present in `mcliv.py`/`mpcliv.py` (see [bugs.md](bugs.md)). Keeps all geometry in world frame and lets Open3D's camera extrinsic handle projection — single transform, correct output.

**Key functions:**

| Function | Description |
|---|---|
| `build_world_geometries(pcd, grasps, top_k, robot_geoms)` | Assemble all scene geometry in world frame |
| `render_pov(pcd, cam, grasps, top_k, robot_geoms)` | Render a single camera's POV → `(H, W, 3)` uint8 |

**CLI — save renders (pure pointcloud):**
```bash
python pov_viz.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
         Calibration_results/extrinsic_results/realsense_1_extrinsic.json \
         Calibration_results/extrinsic_results/realsense_2_extrinsic.json \
  --out_dir renders_pov
```

**CLI — composite over real photos:**
```bash
python pov_viz.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_{0,1,2}_extrinsic.json \
  --color_dir Rope_1/color \
  --out_dir renders_pov
```

**CLI — with robot arm overlay:**
```bash
python pov_viz.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
  --robot_calib Calibration_results/robot_calib_results/robot_calib_result.json \
  --out_dir renders_pov
```

**CLI — interactive viewer:**
```bash
python pov_viz.py --npz ... --grasp_json ... --cams ... --sim
```

---

### `dataloader.py` — Batched dataset builder for VLM pipelines

Generates a structured dataset under `data/data/` — one folder per batch containing overlay images for every camera and a `metadata.json` with full grasp info and projected 2D keypoints. Intended as the primary entry point for building VLM training/eval data.

Accepts the same arguments as `pc_to_image_overlay.py`, replacing `--k` with `--batch_size`.

**CLI:**
```bash
python dataloader.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
         Calibration_results/extrinsic_results/realsense_1_extrinsic.json \
         Calibration_results/extrinsic_results/realsense_2_extrinsic.json \
  --color_dir Rope_1/color \
  --batch_size 3 \
  --mode 3d
```

**Output structure:**
```
data/
  metadata_format.md
  data/
    batch_000/
      overlay_realsense_0.png
      overlay_realsense_1.png
      overlay_realsense_2.png
      metadata.json
    batch_001/ …
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--npz` | required | Path to pointcloud `.npz` |
| `--grasp_json` | required | Path to `grasps.json` |
| `--cams` | required | One or more calibration JSON paths |
| `--color_dir` | `Rope_1/color` | Root of real photos (`<dir>/<cam_idx>/0.png`) |
| `--batch_size` | `3` | Grasps per batch / output image |
| `--top_k` | all | Use only the top-k highest-scoring grasps |
| `--mode` | `3d` | `3d`: thick extruded rectangles; `2d`: thin wireframe |

See [data/metadata_format.md](data/metadata_format.md) for the full `metadata.json` schema.

---

### `pc_to_image_overlay.py` — Batched per-camera overlay renderer

Splits top-K grasps into batches of `k` and saves one composite image per batch per camera. Useful for VLM prompting where you want to show a small number of grasps at a time.

Uses `pov_viz.build_world_geometries` + `viz.render_and_composite` (correct world-frame path).

**CLI:**
```bash
python pc_to_image_overlay.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
         Calibration_results/extrinsic_results/realsense_1_extrinsic.json \
  --color_dir Rope_1/color \
  --out_dir renders_overlay \
  --k 3 \
  --top_k 21
```

Output structure: `renders_overlay/<camera_name>/00.png`, `01.png`, … (3 grasps each).

---

### `mcliv.py` — Multi-camera live/file viewer (legacy)

**Contains BUG-001 (double world-to-camera transform) — use `pov_viz.py` instead.**

Transforms all geometries into each camera's frame before rendering, then also sets the extrinsic — causing the W2C transform to be applied twice. Left here as reference.

**CLI:**
```bash
# interactive viewer per camera
python mcliv.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
  --sim

# save renders
python mcliv.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_{0,1,2}_extrinsic.json \
  --out_dir renders
```

---

### `mpcliv.py` — Multi-camera photo composite (legacy)

**Contains BUG-001 and BUG-002 — use `pov_viz.py` instead.**

Same double-transform issue as `mcliv.py`, plus unnecessarily inverts `T_world_from_cam` when `T_cam_from_world` is available directly. Kept as reference for the compositing logic that was ported (correctly) to `viz.render_and_composite`.

**CLI:**
```bash
python mpcliv.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
  --color_dir Rope_1/color \
  --out_dir renders_with_photo \
  --top_k 10
```

---

## Data Layout

```
Rope_1/
  color/<cam_idx>/0.png          # real camera photos (0=rs0, 1=rs1, 2=rs2)
  depth/<cam_idx>/0.npy          # depth frames
  metadata.json

Calibration_results/
  extrinsic_results/
    realsense_<N>_extrinsic.json # K, T_world_from_cam, T_cam_from_world, image_size, dist
  robot_calib_results/
    robot_calib_result.json      # T_world_from_base

anygrasp_results/
  grasps.json                    # list of {score, pose_matrix_4x4, width, depth, height, ...}
  best_grasp.json
  grasp_pose_matrices.npy

anygrasp_observation_only.npz   # scene pointcloud: keys "points" (N,3) and "colors" (N,3)
```

**Calibration JSON fields used by renderers:**

| Field | Description |
|---|---|
| `K` | 3×3 intrinsic matrix |
| `T_cam_from_world` | 4×4 world→camera extrinsic (use this directly) |
| `T_world_from_cam` | 4×4 camera→world transform (inverse of above) |
| `image_size` | `[width, height]` |
| `camera_name` | e.g. `"realsense_0"` |

---

## Known Bugs

See [bugs.md](bugs.md) for full details.

| ID | Severity | Files | Status |
|---|---|---|---|
| BUG-001 | Critical | `mcliv.py`, `mpcliv.py`, `viz.py` (old) | Fixed in `pov_viz.py` |
| BUG-002 | Minor | `mcliv.py`, `mpcliv.py` | Fixed in `pov_viz.py` |
| BUG-003 | Significant | all composite paths | Open — undistort with `cv2.undistort` before compositing |
| BUG-004 | Needs verification | calibration JSONs, `load_npz` | Verify NPZ is in charuco board frame |

---

## Typical Workflow

```bash
# 1. Collect scene data → Rope_1/
# 2. Run AnyGrasp → anygrasp_results/grasps.json

# 3. Render all 3 cameras, 3 grasps per image, for VLM selection
python pc_to_image_overlay.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_{0,1,2}_extrinsic.json \
  --color_dir Rope_1/color \
  --k 3 --top_k 21 \
  --out_dir renders_overlay

# 4. Feed renders_overlay/<cam>/*.png to VLM to rank/select grasps
```
