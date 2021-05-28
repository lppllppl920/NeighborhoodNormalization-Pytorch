import torch
import MinkowskiEngine as ME
import numpy as np
import scipy
import psutil
from collections import defaultdict


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta))


def sample_random_rotation(randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    return T


def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts


def apply_transform_pose(pts, normals, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    normals = normals @ R.T
    return pts, normals


def connected_vertices_per_vertex_from_faces(faces):
    edges = defaultdict(set)
    for i in range(len(faces)):
        edges[faces[i, 0]].update(faces[i, (1, 2)])
        edges[faces[i, 1]].update(faces[i, (0, 2)])
        edges[faces[i, 2]].update(faces[i, (0, 1)])

    edge_list = []

    for vertex_id in range(len(edges)):
        connected_vertices = edges[vertex_id]
        edge_list.append(list(connected_vertices))

    return edge_list


def has_handle(fpath):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass
    return False


def read_atlas_file(file_path):
    fp = open(str(file_path), "r")
    line_count = 0
    num_modes = 0
    num_vertices = 0

    mesh_mode_vertices_list = list()
    mesh_mode_vertex_mean_displacement_list = list()
    mesh_vertex_array = None
    while True:
        line = fp.readline()
        if line is None or line == "":
            break
        if line_count > 0 and "Mode" not in line:
            words = line.split(",")
            mesh_vertex_array.append([float(words[0]), float(words[1]), float(words[2])])
        elif line_count > 0 and "Mode" in line:
            ind = line.find("Mode ")
            ind2 = line.find(" :")
            mode_idx = int(line[ind + len("Mode "): ind2])
            if mode_idx != 0:
                mesh_mode_vertices_list.append(np.asarray(mesh_vertex_array).reshape((-1, 3)))
                # Get mean displacement value
                ind3 = line.find("Displacements ")
                vertex_displacement = float(line[ind3 + len("Displacements "):])
                mesh_mode_vertex_mean_displacement_list.append(vertex_displacement)
            else:
                mesh_mode_vertex_mean_displacement_list.append(0.0)
            mesh_vertex_array = list()
        elif line_count == 0:
            ind = line.find("Nvertices=")
            ind2 = line.find(" Nmodes=")
            num_vertices = int(line[ind + len("Nvertices="):ind2])
            num_modes = int(line[ind2 + len(" Nmodes="):])

    mesh_mode_vertices_list.append(np.asarray(mesh_vertex_array).reshape((-1, 3)))
    mesh_mode_vertices_list = np.asarray(mesh_mode_vertices_list).reshape((num_modes + 1, num_vertices, 3))
    mesh_mode_vertex_mean_displacement_list = np.asarray(mesh_mode_vertex_mean_displacement_list).reshape((-1,))

    return {"vertices": mesh_mode_vertices_list, "vertex_mean_displacement": mesh_mode_vertex_mean_displacement_list}


def collate_pair_fn(list_data):
    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, scale_ratios = list(
        zip(*list_data))
    xyz_batch0, xyz_batch1 = [], []
    matching_inds_batch, trans_batch, len_batch = [], [], []
    ratio_batch = []
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]

        xyz_batch0.append(to_tensor(xyz0[batch_id]))
        xyz_batch1.append(to_tensor(xyz1[batch_id]))

        trans_batch.append(to_tensor(trans[batch_id]).unsqueeze(dim=0))
        ratio_batch.append(torch.tensor(scale_ratios[batch_id]))

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
        len_batch.append([N0, N1])

        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
    return {
        'pcd0': xyz_batch0,
        'pcd1': xyz_batch1,
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0.float(),
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1.float(),
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch,
        'scale_ratio': ratio_batch,
    }


def separate_collate_pair_fn(list_data):
    temp = list(
        zip(*list_data))
    if len(temp) == 9:
        xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, scale_ratios = temp
    else:
        mesh0_vertices, mesh1_vertices, mesh0_faces, mesh1_faces, \
        xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, scale_ratios = temp

    xyz_batch0, xyz_batch1 = [], []
    matching_inds_batch, matching_inds_batch1, trans_batch, len_batch = [], [], [], []
    ratio_batch = []
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]

        xyz_batch0.append(to_tensor(xyz0[batch_id]))
        xyz_batch1.append(to_tensor(xyz1[batch_id]))

        trans_batch.append(to_tensor(trans[batch_id]).unsqueeze(dim=0))
        ratio_batch.append(torch.tensor(scale_ratios[batch_id]))

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id])).int())
        matching_inds_batch1.append(
            torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))

        len_batch.append([N0, N1])

        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch1 = torch.cat(matching_inds_batch1, 0).int()
    # ratio_batch = torch.cat(ratio_batch, 0).float()
    if len(temp) == 9:
        return {
            'pcd0': xyz_batch0,
            'pcd1': xyz_batch1,
            'sinput0_C': coords_batch0,
            'sinput0_F': feats_batch0.float(),
            'sinput1_C': coords_batch1,
            'sinput1_F': feats_batch1.float(),
            'correspondences': matching_inds_batch,
            'correspondences_group': matching_inds_batch1,
            'T_gt': trans_batch,
            'len_batch': len_batch,
            'scale_ratio': ratio_batch,
        }
    else:
        return {
            'mesh0_vertices': mesh0_vertices,
            'mesh1_vertices': mesh1_vertices,
            'mesh0_faces': mesh0_faces,
            'mesh1_faces': mesh1_faces,
            'pcd0': xyz_batch0,
            'pcd1': xyz_batch1,
            'sinput0_C': coords_batch0,
            'sinput0_F': feats_batch0.float(),
            'sinput1_C': coords_batch1,
            'sinput1_F': feats_batch1.float(),
            'correspondences': matching_inds_batch,
            'correspondences_group': matching_inds_batch1,
            'T_gt': trans_batch,
            'len_batch': len_batch,
            'scale_ratio': ratio_batch,
        }


def sample_random_transform(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


def generate_trans(rotation_axis, rotation_angle):
    T = np.eye(4)
    R = M(rotation_axis, rotation_angle * np.pi / 180.0)
    T[:3, :3] = R
    return T


def M(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta))
