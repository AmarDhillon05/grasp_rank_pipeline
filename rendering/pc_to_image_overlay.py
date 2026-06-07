import os
import re
import shutil
import json
import argparse
import numpy as np
from PIL import Image

from pov_viz import build_world_geometries
from viz import load_npz, load_grasps, assign_grasp_colors, draw_grasp_labels, draw_grasp_keypoints, draw_grasp_schematic, render_and_composite


def cam_index_from_name(camera_name):
    m = re.search(r"(\d+)$", camera_name)
    return int(m.group(1)) if m else 0


def main(args):
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    pcd = load_npz(args.npz)
    grasps = load_grasps(args.grasp_json)  # already sorted by score desc

    top_grasps = grasps[:args.top_k] if args.top_k else grasps
    batches = [top_grasps[i:i + args.k] for i in range(0, len(top_grasps), args.k)]
    print(f"{len(top_grasps)} grasps -> {len(batches)} batches of up to {args.k}")

    # Assign colors once per batch (shared across cameras so each image is consistent)
    batch_color_assignments = [assign_grasp_colors(batch) for batch in batches]

    # Print the global color legend
    print("\nGrasp color assignments:")
    for i, (batch, cas) in enumerate(zip(batches, batch_color_assignments)):
        print(f"  Batch {i}:")
        for g, ca in zip(batch, cas):
            r, g_c, b = ca["color"]
            print(f"    {ca['label']}  rgb=({r:.2f}, {g_c:.2f}, {b:.2f})  score={g['score']:.4f}")
    print()

    for cam_path in args.cams:
        cam = json.load(open(cam_path))
        cam_idx = cam_index_from_name(cam["camera_name"])
        photo_path = os.path.join(args.color_dir, str(cam_idx), "0.png")

        cam_dir = os.path.join(args.out_dir, cam["camera_name"])
        os.makedirs(cam_dir)

        width, height = cam["image_size"]
        K_mat = np.array(cam["K"], dtype=np.float64)
        T_cam_from_world = np.array(cam["T_cam_from_world"], dtype=np.float64)

        photo = np.array(Image.open(photo_path).convert("RGB"))
        if photo.shape[:2] != (height, width):
            photo = np.array(Image.fromarray(photo).resize((width, height), Image.LANCZOS))

        for i, (batch, cas) in enumerate(zip(batches, batch_color_assignments)):
            lw = 8 if args.mode == "3d" else 3
            composite = draw_grasp_schematic(photo.copy(), batch, cas, K_mat, T_cam_from_world, line_width=lw)
            out_path = os.path.join(cam_dir, f"{i:02d}.png")
            Image.fromarray(composite).save(out_path)
            print(f"  saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--grasp_json", required=True)
    parser.add_argument("--cams", nargs="+", required=True)
    parser.add_argument("--color_dir", default="scene_data/Rope_1/color",
                        help="root of real photos: <color_dir>/<cam_idx>/0.png")
    parser.add_argument("--out_dir", default="renders/renders_overlay")
    parser.add_argument("--k", type=int, default=3,
                        help="grasps shown per image")
    parser.add_argument("--top_k", type=int, default=None,
                        help="total grasps to use (default: all)")
    parser.add_argument("--mode", choices=["3d", "2d"], default="3d",
                        help="3d: Open3D gripper meshes composited over photo; "
                             "2d: projected 2D schematic (dots + lines) on photo, no Open3D")
    main(parser.parse_args())
