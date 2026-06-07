# Running pc_to_image_overlay

Run from the repo root (`/home/adhil/vlm_ggen_interp`), using the Windows `graspgen` conda environment via WSL:

```bash
/mnt/c/Users/adhil/miniconda3/envs/graspgen/python.exe rendering/pc_to_image_overlay.py \
  --npz scene_data/anygrasp_observation_only.npz \
  --grasp_json scene_data/anygrasp_results/grasps.json \
  --cams scene_data/Calibration_results/extrinsic_results/realsense_0_extrinsic.json \
         scene_data/Calibration_results/extrinsic_results/realsense_1_extrinsic.json \
         scene_data/Calibration_results/extrinsic_results/realsense_2_extrinsic.json \
  --color_dir scene_data/Rope_1/color \
  --out_dir renders/renders_overlay
```

Output is written to `renders/renders_overlay/<camera_name>/<batch_index>.png` (3 grasps per image by default).
