"""
viz_with_arm.py — Franka Panda arm visualization utilities.

Computes forward kinematics from the official Franka Panda DH parameters
and builds an Open3D stick-figure (spheres at joints, cylinders for links).

Because joint angles at capture time are unavailable, the arm is rendered
at a configurable pose — defaulting to the standard ready configuration
q = [0, -π/4, 0, -3π/4, 0, π/2, π/4]. This gives correct scale and
reach relative to the scene but does not represent the actual arm pose.

Usage:
    from viz_with_arm import make_franka_geometry, PANDA_HOME_Q

    geoms = make_franka_geometry(T_world_from_base)             # ready pose
    geoms = make_franka_geometry(T_world_from_base, q=my_q)    # custom joints
"""

import numpy as np
import open3d as o3d


# Franka Panda DH parameters (standard DH): (a_i, d_i, alpha_i) per joint
_PANDA_DH = [
    (0,       0.333,  0       ),   # joint 1
    (0,       0,     -np.pi/2 ),   # joint 2
    (0,       0.316,  np.pi/2 ),   # joint 3
    (0.0825,  0,      np.pi/2 ),   # joint 4
    (-0.0825, 0.384, -np.pi/2 ),   # joint 5
    (0,       0,      np.pi/2 ),   # joint 6
    (0.088,   0.107,  np.pi/2 ),   # joint 7 + flange
]

_EE_D = 0.1034  # flange-to-TCP offset along local z (m)

PANDA_HOME_Q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])


def _dh(a, d, alpha, theta):
    """Single standard DH transform matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ],
    ])


def franka_fk(q=None):
    """
    Compute 4x4 link frame transforms for the Franka Panda in base frame.

    Args:
        q: (7,) joint angles in radians. Defaults to PANDA_HOME_Q.

    Returns:
        List of 9 × (4, 4) numpy arrays:
        [base_frame, j1, j2, j3, j4, j5, j6, j7, ee_tcp]
    """
    if q is None:
        q = PANDA_HOME_Q

    T = np.eye(4)
    frames = [T.copy()]

    for i, (a, d, alpha) in enumerate(_PANDA_DH):
        T = T @ _dh(a, d, alpha, q[i])
        frames.append(T.copy())

    T_ee = T.copy()
    T_ee[:3, 3] += T[:3, :3] @ np.array([0.0, 0.0, _EE_D])
    frames.append(T_ee)

    return frames


def make_franka_geometry(T_world_from_base, q=None):
    """
    Build Open3D stick-figure geometry for the Franka Panda arm.

    Renders:
      - Blue-grey spheres at each joint origin
      - Grey cylinders for each link
      - Green sphere at the TCP / end-effector

    All geometries are returned in world frame (T_world_from_base applied).

    Args:
        T_world_from_base: (4, 4) robot base → world transform
                           (from robot_calib_result.json "T_world_from_base").
        q: (7,) joint angles in radians. Defaults to PANDA_HOME_Q.

    Returns:
        List of open3d.geometry.TriangleMesh objects in world frame.
    """
    frames = franka_fk(q)
    positions = [f[:3, 3] for f in frames]

    JOINT_COLOR = [0.35, 0.35, 0.75]
    LINK_COLOR  = [0.65, 0.65, 0.65]
    EE_COLOR    = [0.2,  0.8,  0.2 ]
    JOINT_R     = 0.035
    LINK_R      = 0.022

    geoms = []

    for i, pos in enumerate(positions):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=JOINT_R)
        s.compute_vertex_normals()
        s.paint_uniform_color(EE_COLOR if i == len(positions) - 1 else JOINT_COLOR)
        s.translate(pos)
        geoms.append(s)

    for p0, p1 in zip(positions[:-1], positions[1:]):
        vec = p1 - p0
        length = np.linalg.norm(vec)
        if length < 1e-4:
            continue

        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=LINK_R, height=length)
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(LINK_COLOR)

        direction = vec / length
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-6:
            angle = np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / axis_len * angle)
            cyl.rotate(R, center=(0, 0, 0))

        cyl.translate((p0 + p1) / 2)
        geoms.append(cyl)

    for g in geoms:
        g.transform(T_world_from_base)

    return geoms
