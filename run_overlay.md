# Running pc_to_image_overlay

Run from the repo root (`/home/adhil/vlm_ggen_interp`), using the Windows `graspgen` conda environment via WSL:

```bash
/mnt/c/Users/adhil/miniconda3/envs/graspgen/python.exe pc_to_image_overlay.py \
  --npz anygrasp_observation_only.npz \
  --grasp_json anygrasp_results/grasps.json \
  --cams Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
         Calibration_results/extrinsic_results/realsense_1_extrinsic.json \
         Calibration_results/extrinsic_results/realsense_2_extrinsic.json \
  --color_dir Rope_1/color \
  --out_dir renders_overlay
```

Output is written to `renders_overlay/<camera_name>/<batch_index>.png` (3 grasps per image by default).
