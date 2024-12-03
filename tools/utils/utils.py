import open3d as o3d
import numpy as np
import sys
import torch
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_open3d_point_cloud(xyz, color=None):
    """construct point cloud from coordinates and colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def load_npy_file(filename):
    return np.load(filename)


def load_npy_files(files):
    out = []
    for file in files:
        out.append(load_npy_file(file))
    return np.array(out)


def load_pcd(filename):
    pc = o3d.io.read_point_cloud(filename)
    return np.asarray(pc.points)


def load_pc_file(filename,
                 coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                 div_n=[256, 256, 32],
                 is_pcd=False):
    if is_pcd:
        pc = load_pcd(filename)
    else:
        pc = np.fromfile(filename, dtype="float32").reshape(-1, 4)[:, :3]

    pc = torch.from_numpy(pc).to(device)
    ids = load_voxel(pc,
                     coords_range_xyz=coords_range_xyz,
                     div_n=div_n)
    voxel_out = torch.zeros(div_n)
    voxel_out[ids[:, 0], ids[:, 1], ids[:, 2]] = 1
    return voxel_out


def load_pc_files(files,
                  coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                  div_n=[256, 256, 32],
                  is_pcd=False):
    out = []
    for file in files:
        out.append(load_pc_file(file, coords_range_xyz, div_n, is_pcd=is_pcd))
    return torch.stack(out)


def load_voxel(data,
               coords_range_xyz=[-50., -50, -4, 50, 50, 3],
               div_n=[256, 256, 32]):
    div = [(coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0],
           (coords_range_xyz[4] - coords_range_xyz[1]) / div_n[1],
           (coords_range_xyz[5] - coords_range_xyz[2]) / div_n[2]]
    id_x = (data[:, 0] - coords_range_xyz[0]) / div[0]
    id_y = (data[:, 1] - coords_range_xyz[1]) / div[1]
    id_z = (data[:, 2] - coords_range_xyz[2]) / div[2]
    all_id = torch.cat(
        [id_x.reshape(-1, 1), id_y.reshape(-1, 1), id_z.reshape(-1, 1)], axis=1).long()

    mask = (all_id[:, 0] >= 0) & (all_id[:, 1] >= 0) & (all_id[:, 2] >= 0) & (
        all_id[:, 0] < div_n[0]) & (all_id[:, 1] < div_n[1]) & (all_id[:, 2] < div_n[2])
    all_id = all_id[mask]
    data = data[mask]
    ids, _, _ = torch.unique(
        all_id, return_inverse=True, return_counts=True, dim=0)

    return ids


def read_poses(file, dx=2, st=100):
    stamp = None
    pose = None
    delta_d = dx**2
    with open(file) as f:
        lines = f.readlines()[st:]
    for line in lines:
        line = line.strip().split()
        stampi = line[0]
        posei = [float(line[i]) for i in range(1, len(line))]
        if pose is None:
            pose = [posei]
            stamp = [stampi]
        else:
            diffx = posei[0] - pose[-1][0]
            diffy = posei[1] - pose[-1][1]
            if diffx**2 + diffy**2 > delta_d:
                pose.append(posei)
                stamp.append(stampi)
    pose = np.array(pose, dtype='float32')
    return stamp, pose


def rot3d(axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m


def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts


def occ_pcd(points, state_st=6, max_range=np.pi):
    rand_state = random.randint(state_st, 10)
    if rand_state > 9:
        rand_start = random.uniform(-np.pi, np.pi)
        rand_end = random.uniform(rand_start, min(np.pi,
                                                  rand_start + max_range))
        angles = np.arctan2(points[:, 1], points[:, 0])
        return points[(angles < rand_start) | (angles > rand_end)]
    else:
        return points
