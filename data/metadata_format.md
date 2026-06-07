# Metadata Format

Each `data/data/batch_NNN/metadata.json` describes one batch of grasps.

---

## Top-level fields

| Field | Type | Description |
|---|---|---|
| `batch_index` | int | Zero-based index of this batch |
| `grasps` | list | One entry per grasp in the batch (ordered by score descending) |

---

## Per-grasp fields

### Identity / visualisation

| Field | Type | Description |
|---|---|---|
| `label` | str | Short display label — `"G0"`, `"G1"`, `"G2"` — matches the text drawn on the overlay images |
| `color_rgb` | [r, g, b] | Float values in [0, 1]. The color used for this grasp in all overlay images for this batch |

### AnyGrasp outputs

| Field | Type | Description |
|---|---|---|
| `rank` | int | Global rank across all grasps (0 = highest scoring) |
| `score` | float | AnyGrasp confidence score |
| `width` | float | Jaw opening distance in metres |
| `depth` | float | Finger reach distance in metres |
| `height` | float | Finger height in metres |
| `translation` | [x, y, z] | Grasp origin in world frame (metres). World frame = charuco board frame |
| `rotation_matrix` | [[r00..r02], [r10..r12], [r20..r22]] | 3×3 rotation matrix. Columns are the local x, y, z axes of the gripper frame in world coordinates |
| `pose_matrix_4x4` | [[...], [...], [...], [0,0,0,1]] | 4×4 homogeneous transform combining rotation_matrix and translation |
| `object_id` | int or null | AnyGrasp object cluster ID |

### Camera projections

`camera_projections` is a dict keyed by camera name (e.g. `"realsense_0"`).

Each entry has:

| Field | Type | Description |
|---|---|---|
| `in_frame` | bool | True if at least one keypoint projected in front of this camera |
| `keypoints_px` | dict | 2D pixel coordinates [u, v] of each gripper keypoint, or `null` if behind the camera |

#### Keypoints

The gripper frame has its origin at the wrist (palm centre). Local axes:
- **x**: finger reach direction (fingertips are at `x = depth`)
- **y**: jaw axis (fingers are at `y = ±width/2`)
- **z**: finger height axis

| Keypoint | Local position | Description |
|---|---|---|
| `lf_tip` | [depth, -width/2, 0] | Left fingertip |
| `rf_tip` | [depth, +width/2, 0] | Right fingertip |
| `lf_base` | [0, -width/2, 0] | Left finger base (where finger meets palm bar) |
| `rf_base` | [0, +width/2, 0] | Right finger base |
| `wrist` | [0, 0, 0] | Gripper origin / palm centre |

Pixel coordinates are in standard image space: `u` increases rightward, `v` increases downward, origin at top-left. They are computed using the pinhole model (no lens distortion correction — see `bugs.md` BUG-003).

---

## Example

```json
{
  "batch_index": 0,
  "grasps": [
    {
      "label": "G0",
      "color_rgb": [0.95, 0.09, 0.09],
      "rank": 0,
      "score": 0.2603,
      "width": 0.0537,
      "depth": 0.03,
      "height": 0.03,
      "translation": [0.0425, 0.0112, -0.0407],
      "rotation_matrix": [[-0.388, 0.265, -0.883], [-0.729, 0.497, 0.470], [0.563, 0.826, 0.0]],
      "pose_matrix_4x4": [[-0.388, 0.265, -0.883, 0.0425], [-0.729, 0.497, 0.470, 0.0112], [0.563, 0.826, 0.0, -0.0407], [0.0, 0.0, 0.0, 1.0]],
      "object_id": 0,
      "camera_projections": {
        "realsense_0": {
          "in_frame": true,
          "keypoints_px": {
            "lf_tip":  [423.1, 318.7],
            "rf_tip":  [401.5, 334.2],
            "lf_base": [440.3, 302.1],
            "rf_base": [418.8, 317.6],
            "wrist":   [429.5, 309.8]
          }
        },
        "realsense_1": { "..." : "..." },
        "realsense_2": { "..." : "..." }
      }
    }
  ]
}
```
