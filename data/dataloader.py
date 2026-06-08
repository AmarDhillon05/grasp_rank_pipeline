import os
import re
import sys
import json
import shutil
import random
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rendering'))
from viz import load_npz, load_grasps, assign_grasp_colors, draw_grasp_schematic, grasp_to_T


def cam_index_from_name(camera_name):
    m = re.search(r"(\d+)$", camera_name)
    return int(m.group(1)) if m else 0


def project_keypoints(g, K_mat, T_cam_from_world):
    """
    Project the 5 gripper keypoints into pixel coordinates for one camera.

    Returns a dict with 'keypoints_px' (name → [u, v] or null if behind camera)
    and 'in_frame' (True if at least one keypoint projected in front of the camera).
    """
    T = grasp_to_T(g).astype(np.float64)
    Rg, tg = T[:3, :3], T[:3, 3]
    w, d = g["width"], g["depth"]

    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    R_ext = T_cam_from_world[:3, :3]
    t_ext = T_cam_from_world[:3, 3]

    def project(p_world):
        p = R_ext @ p_world + t_ext
        if p[2] <= 0:
            return None
        return [round(float(fx * p[0] / p[2] + cx), 2),
                round(float(fy * p[1] / p[2] + cy), 2)]

    keypoints = {
        "lf_tip":  project(tg + Rg @ np.array([d,    -w / 2, 0.0])),
        "rf_tip":  project(tg + Rg @ np.array([d,     w / 2, 0.0])),
        "lf_base": project(tg + Rg @ np.array([0.0,  -w / 2, 0.0])),
        "rf_base": project(tg + Rg @ np.array([0.0,   w / 2, 0.0])),
        "wrist":   project(tg),
    }
    return {
        "keypoints_px": keypoints,
        "in_frame": any(v is not None for v in keypoints.values()),
    }


def build_batch_metadata(batch_idx, batch, color_assignments, cameras):
    grasp_records = []
    for g, ca in zip(batch, color_assignments):
        record = {
            "label":              ca["label"],
            "color_rgb":          ca["color"],
            "rank":               g.get("rank"),
            "score":              g["score"],
            "width":              g["width"],
            "depth":              g["depth"],
            "height":             g["height"],
            "translation":        g["translation"],
            "rotation_matrix":    g["rotation_matrix"],
            "pose_matrix_4x4":    g["pose_matrix_4x4"],
            "object_id":          g.get("object_id"),
            "camera_projections": {},
        }
        for cam in cameras:
            cam_name = cam["camera_name"]
            K_mat = np.array(cam["K"], dtype=np.float64)
            T_cam_from_world = np.array(cam["T_cam_from_world"], dtype=np.float64)
            record["camera_projections"][cam_name] = project_keypoints(g, K_mat, T_cam_from_world)
        grasp_records.append(record)

    return {"batch_index": batch_idx, "grasps": grasp_records}


def main(args):
    out_root = os.path.join("data", "data")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root)

    grasps = load_grasps(args.grasp_json)
    top_grasps = grasps[:args.top_k] if args.top_k else grasps
    batches = [top_grasps[i:i + args.batch_size]
               for i in range(0, len(top_grasps), args.batch_size)]
    print(f"{len(top_grasps)} grasps -> {len(batches)} batches of up to {args.batch_size}")

    global_cas = assign_grasp_colors(top_grasps)
    colors = [ca["color"] for ca in global_cas]
    random.seed(0)
    random.shuffle(colors)
    global_cas = [{"label": ca["label"], "color": c} for ca, c in zip(global_cas, colors)]

    cameras = [json.load(open(p)) for p in args.cams]

    # Pre-load and resize photos once per camera
    photos = {}
    for cam in cameras:
        cam_idx = cam_index_from_name(cam["camera_name"])
        photo_path = os.path.join(args.color_dir, str(cam_idx), "0.png")
        w, h = cam["image_size"]
        photo = np.array(Image.open(photo_path).convert("RGB"))
        if photo.shape[:2] != (h, w):
            photo = np.array(Image.fromarray(photo).resize((w, h), Image.LANCZOS))
        photos[cam["camera_name"]] = (
            photo,
            np.array(cam["K"], dtype=np.float64),
            np.array(cam["T_cam_from_world"], dtype=np.float64),
        )

    lw = 8 if args.mode == "3d" else 3

    for batch_idx, batch in enumerate(batches):
        batch_dir = os.path.join(out_root, f"batch_{batch_idx:03d}")
        os.makedirs(batch_dir)

        start = batch_idx * args.batch_size
        cas = global_cas[start:start + len(batch)]

        for cam in cameras:
            cam_name = cam["camera_name"]
            photo, K_mat, T_cam_from_world = photos[cam_name]
            composite = draw_grasp_schematic(photo.copy(), batch, cas, K_mat, T_cam_from_world, line_width=lw)
            Image.fromarray(composite).save(os.path.join(batch_dir, f"overlay_{cam_name}.png"))

        meta = build_batch_metadata(batch_idx, batch, cas, cameras)
        with open(os.path.join(batch_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  batch {batch_idx:03d}: {len(batch)} grasps saved")

    print(f"\nDone. Output in data/data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",        required=True)
    parser.add_argument("--grasp_json", required=True)
    parser.add_argument("--cams",       nargs="+", required=True)
    parser.add_argument("--color_dir",  default="scene_data/Rope_1/color")
    parser.add_argument("--batch_size", type=int, default=3,
                        help="number of grasps per batch / output image")
    parser.add_argument("--top_k",      type=int, default=None,
                        help="use only the top-k highest-scoring grasps")
    parser.add_argument("--mode",       choices=["3d", "2d"], default="3d",
                        help="3d: thick extruded rectangles; 2d: thin wireframe")
    main(parser.parse_args())
