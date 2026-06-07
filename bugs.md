# Bug Log — pointcloud camera projection

---

## 2026-06-01

### BUG-001: Double world-to-camera transform (critical)
**Files:** `mcliv.py:16`, `mpcliv.py:66`, `viz.py:111`

Geometries were pre-transformed into camera frame via `.transform(T_cam_from_world)`, then
`params.extrinsic = T_cam_from_world` was also set. Since Open3D's `PinholeCameraParameters.extrinsic`
is the world-to-camera matrix that the renderer applies to every point, the W2C transform was
applied twice:

```
P_rendered = T_cam_from_world @ (T_cam_from_world @ P_world)
```

The virtual camera ended up "looking" at world-frame coordinates while the geometry sat at
camera-frame coordinates — producing completely wrong views.

**Fix in `pov_viz.py`:** geometries are kept in world frame; Open3D's extrinsic alone handles
the projection.

---

### BUG-002: Redundant inversion of T_world_from_cam (minor)
**Files:** `mcliv.py:13`, `mpcliv.py:63`

```python
T_cam_from_world = np.linalg.inv(np.array(cam["T_world_from_cam"], dtype=np.float64))
```

`T_cam_from_world` is already stored as a top-level key in the calibration JSON alongside
`T_world_from_cam`, so this inversion is unnecessary and obscures intent.

**Fix in `pov_viz.py`:** reads `cam["T_cam_from_world"]` directly.

---

### BUG-003: Lens distortion not accounted for (significant for overlays)
**Files:** `mpcliv.py:83-88`, all rendering paths

The calibration JSONs store non-trivial distortion parameters, e.g. for realsense_0:

```
dist = [k1=0.205, k2=-0.588, p1=-0.010, p2=-0.002, k3=0.329]
```

Open3D's `PinholeCameraIntrinsic` accepts only `(fx, fy, cx, cy)` — it renders as a pure
pinhole with no distortion. The real camera images are distorted. In `mpcliv.py` the rendered
overlay is composited directly over the distorted real photo, causing systematic spatial
misalignment (worst near image edges).

**Status: not yet fixed.** Mitigation: undistort real images with `cv2.undistort(img, K, dist)`
before compositing, so both the render and the photo match the pinhole model.

---

### BUG-004: World-frame assumption — verify pointcloud frame (needs verification)
**Files:** calibration JSONs, `load_npz` in `viz.py`

The calibration JSONs declare `"world_frame": "charuco_board"`, meaning extrinsics transform
between camera frame and the charuco board frame. The NPZ pointcloud must be expressed in that
same charuco board frame for the extrinsics to be meaningful. If the pointcloud is in any other
frame (depth sensor, robot base, etc.) the projections will be wrong regardless of other fixes.

**Status: unverified.** Confirm that the NPZ pointcloud origin and axes match the charuco board world frame.
