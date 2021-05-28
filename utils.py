import time
import numpy as np
import torch
from skimage.measure import ransac
import textwrap
import tqdm
from collections import defaultdict
import open3d as o3d
import os
import re
from sklearn.neighbors import BallTree
import cv2
from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial import cKDTree
import umap
from torch_geometric.data import Data
import MinkowskiEngine as ME
from pathlib import Path
import logging


def load_full_model(model, trained_model_path):
    if not Path(trained_model_path).exists():
        raise IOError(f"No trained weights detected at {trained_model_path}")

    logging.info("Loading {:s} ...".format(str(trained_model_path)))
    state = torch.load(str(trained_model_path))
    step = state['step']
    epoch = state['epoch'] + 1
    model.load_state_dict(state["model"])
    logging.info('Restored model, epoch {}, step {}'.format(epoch, step))

    return epoch, step


def tuplify(val, length):
    if type(val) in [tuple, list] and len(val) == length:
        return tuple(val)
    else:
        return tuple(val for _ in range(length))


def compute_alpha_stats(state_dict):
    mean_att_list = list()
    std_att_list = list()
    for key in state_dict.keys():
        if "mean_att" in key:
            mean_att_list.append(state_dict[key].item())

        if "std_att" in key:
            std_att_list.append(state_dict[key].item())

    mean_atts = np.asarray(mean_att_list)
    std_atts = np.asarray(std_att_list)

    print(f"alpha 1 mean: {np.mean(mean_atts)} std: {np.std(mean_atts)}, "
          f"alpha 2 mean: {np.mean(std_atts)} std: {np.std(std_atts)} \n")
    return


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def vis_config():
    camera_config = {
        'cls': 'PerspectiveCamera',
        'fov': 90,
        'aspect': 16.0 / 9.0,
        'near': 1.0e-4,
        'far': 1.0e5
    }

    light_config = {
        'cls': 'DirectionalLight',
        'color': 0xffffff,
        'intensity': 1.0,
    }

    material_config = {
        'cls': 'MeshLambertMaterial',
        'side': 2,  # 2 means rendering both sides of the object
    }

    config_dict = {'camera': camera_config, "light": light_config, "material": material_config}
    return config_dict


def visualize_curvature(est_curvature_0, gt_curvature_0, sample_0):
    coords_0 = sample_0.pos.detach().cpu().numpy()

    gt_curvature_0 = gt_curvature_0.cpu().numpy()
    est_curvature_0 = est_curvature_0.cpu().numpy()

    max_value = np.maximum(np.max(est_curvature_0), np.max(gt_curvature_0))
    min_value = np.minimum(np.min(est_curvature_0), np.min(gt_curvature_0))

    normalized_est_curvature_0 = np.asarray((est_curvature_0 - min_value) / (max_value - min_value) * 255,
                                            dtype=np.uint8)
    normalized_gt_curvature_0 = np.asarray((gt_curvature_0 - min_value) / (max_value - min_value) * 255, dtype=np.uint8)

    normalized_est_curvature_0 = cv2.cvtColor(cv2.applyColorMap(normalized_est_curvature_0, cv2.COLORMAP_JET),
                                              cv2.COLOR_BGR2RGB)
    normalized_gt_curvature_0 = cv2.cvtColor(cv2.applyColorMap(normalized_gt_curvature_0, cv2.COLORMAP_JET),
                                             cv2.COLOR_BGR2RGB)

    gt_mesh_0 = o3d.geometry.TriangleMesh()
    gt_mesh_0.vertices = o3d.utility.Vector3dVector(coords_0)
    gt_mesh_0.vertex_colors = o3d.utility.Vector3dVector(normalized_gt_curvature_0.reshape((-1, 3)) / 255.0)
    gt_mesh_0.triangles = o3d.utility.Vector3iVector(sample_0.faces.cpu().numpy())

    est_mesh_0 = o3d.geometry.TriangleMesh()
    est_mesh_0.vertices = o3d.utility.Vector3dVector(coords_0)
    est_mesh_0.vertex_colors = o3d.utility.Vector3dVector(normalized_est_curvature_0.reshape((-1, 3)) / 255.0)
    est_mesh_0.triangles = o3d.utility.Vector3iVector(sample_0.faces.cpu().numpy())

    return gt_mesh_0, est_mesh_0


def visualize_heatmap(feat_response_0, feat_response_1, selected_pos_index_pair_0_1, sample_0, sample_1):
    selected_pos_index_pair_0_1 = selected_pos_index_pair_0_1.detach().cpu().numpy()
    pos_index_0 = selected_pos_index_pair_0_1[0]
    pos_index_1 = selected_pos_index_pair_0_1[1]

    coords_0 = sample_0.pos.detach().cpu().numpy()
    coords_1 = sample_1.pos.detach().cpu().numpy()

    ball_tree_0 = BallTree(coords_0)
    neighbor_indexes_0 = ball_tree_0.query_radius(X=coords_0[pos_index_0].reshape(-1, 3),
                                                  r=3.0 * sample_0.edge_length.item())[0]

    ball_tree_1 = BallTree(coords_1)
    neighbor_indexes_1 = ball_tree_1.query_radius(X=coords_1[pos_index_1].reshape(-1, 3),
                                                  r=3.0 * sample_1.edge_length.item())[0]

    gt_heatmap_0 = np.zeros((coords_0.shape[0], 1), dtype=np.uint8)
    gt_heatmap_0[neighbor_indexes_0] = 255

    gt_heatmap_1 = np.zeros((coords_1.shape[0], 1), dtype=np.uint8)
    gt_heatmap_1[neighbor_indexes_1] = 255

    est_heatmap_0 = feat_response_0.reshape(-1, 1).cpu().numpy()
    est_heatmap_1 = feat_response_1.reshape(-1, 1).cpu().numpy()

    est_heatmap_0 = np.asarray(est_heatmap_0 / np.max(est_heatmap_0) * 255, dtype=np.uint8)
    est_heatmap_1 = np.asarray(est_heatmap_1 / np.max(est_heatmap_1) * 255, dtype=np.uint8)

    gt_heatmap_0 = cv2.cvtColor(cv2.applyColorMap(gt_heatmap_0, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    gt_heatmap_1 = cv2.cvtColor(cv2.applyColorMap(gt_heatmap_1, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    est_heatmap_0 = cv2.cvtColor(cv2.applyColorMap(est_heatmap_0, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    est_heatmap_1 = cv2.cvtColor(cv2.applyColorMap(est_heatmap_1, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    mesh_0 = o3d.geometry.TriangleMesh()
    mesh_0.vertices = o3d.utility.Vector3dVector(coords_0)
    mesh_0.vertex_colors = o3d.utility.Vector3dVector(gt_heatmap_0.reshape((-1, 3)) / 255.0)
    mesh_0.triangles = o3d.utility.Vector3iVector(sample_0.faces.cpu().numpy())

    mesh_1 = o3d.geometry.TriangleMesh()
    mesh_1.vertices = o3d.utility.Vector3dVector(coords_1)
    mesh_1.vertex_colors = o3d.utility.Vector3dVector(gt_heatmap_1.reshape((-1, 3)) / 255.0)
    mesh_1.triangles = o3d.utility.Vector3iVector(sample_1.faces.cpu().numpy())

    est_mesh_0 = o3d.geometry.TriangleMesh()
    est_mesh_0.vertices = o3d.utility.Vector3dVector(coords_0)
    est_mesh_0.vertex_colors = o3d.utility.Vector3dVector(est_heatmap_0.reshape((-1, 3)) / 255.0)
    est_mesh_0.triangles = o3d.utility.Vector3iVector(sample_0.faces.cpu().numpy())

    est_mesh_1 = o3d.geometry.TriangleMesh()
    est_mesh_1.vertices = o3d.utility.Vector3dVector(coords_1)
    est_mesh_1.vertex_colors = o3d.utility.Vector3dVector(est_heatmap_1.reshape((-1, 3)) / 255.0)
    est_mesh_1.triangles = o3d.utility.Vector3iVector(sample_1.faces.cpu().numpy())

    return mesh_0, mesh_1, est_mesh_0, est_mesh_1


def colorize_mesh_with_descriptor(F0, F1, pos0, pos1, faces0, faces1, T_est=None):
    # tSNE learning
    tsne_learner = TSNE(n_components=3, perplexity=30, early_exaggeration=12.0, learning_rate=200.0,
                        n_iter=500,
                        n_iter_without_progress=200, min_grad_norm=1.0e-7, metric="euclidean", init="pca",
                        verbose=1, random_state=0, method="barnes_hut", angle=0.5, n_jobs=2
                        )
    color_coded_features = tsne_learner.fit_transform(torch.cat([F0, F1], dim=0).cpu().numpy())
    color_max = np.amax(color_coded_features, axis=0, keepdims=True)
    color_min = np.amin(color_coded_features, axis=0, keepdims=True)
    color_coded_features = (color_coded_features - color_min) / (color_max - color_min)

    mesh_0 = o3d.geometry.TriangleMesh()
    mesh_1 = o3d.geometry.TriangleMesh()

    pos0 = pos0.cpu().numpy()
    mesh_0.vertices = o3d.utility.Vector3dVector(pos0)
    mesh_0.vertex_colors = o3d.utility.Vector3dVector(color_coded_features[:F0.shape[0]])
    mesh_0.triangles = o3d.utility.Vector3iVector(faces0.cpu().numpy())

    pos1 = pos1.cpu().numpy()
    mesh_1.vertices = o3d.utility.Vector3dVector(pos1)
    mesh_1.vertex_colors = o3d.utility.Vector3dVector(color_coded_features[F0.shape[0]:])
    mesh_1.triangles = o3d.utility.Vector3iVector(faces1.cpu().numpy())

    if T_est is not None:
        pos1_est = pos1 @ T_est[:3, :3].T + T_est[:3, 3]
        transformed_mesh_1 = o3d.geometry.TriangleMesh()
        transformed_mesh_1.vertices = o3d.utility.Vector3dVector(pos1_est)
        transformed_mesh_1.vertex_colors = o3d.utility.Vector3dVector(color_coded_features[F0.shape[0]:])
        transformed_mesh_1.triangles = o3d.utility.Vector3iVector(faces1.cpu().numpy())

        return mesh_0, mesh_1, transformed_mesh_1
    else:
        return mesh_0, mesh_1


def colorize_points_with_descriptor_and_display_matches(F0, F1, pos0, pos1, nn_indices_in_1):
    # tSNE learning
    tsne_learner = TSNE(n_components=3, perplexity=30, early_exaggeration=12.0, learning_rate=200.0,
                        n_iter=500,
                        n_iter_without_progress=200, min_grad_norm=1.0e-7, metric="euclidean", init="pca",
                        verbose=1, random_state=0, method="barnes_hut", angle=0.5, n_jobs=2
                        )
    if isinstance(F0, torch.Tensor):
        feats = torch.cat([F0, F1], dim=0).cpu().numpy()
    elif isinstance(F0, np.ndarray):
        feats = np.concatenate([F0, F1], axis=0)
    else:
        raise NotImplementedError("not supported type for F0 and F1")

    color_coded_features = tsne_learner.fit_transform(feats)
    color_max = np.amax(color_coded_features, axis=0, keepdims=True)
    color_min = np.amin(color_coded_features, axis=0, keepdims=True)
    color_coded_features = (color_coded_features - color_min) / (color_max - color_min)

    mesh_0 = o3d.geometry.PointCloud()
    mesh_1 = o3d.geometry.PointCloud()

    if isinstance(pos0, torch.Tensor):
        pos0 = pos0.cpu().numpy()
    mesh_0.points = o3d.utility.Vector3dVector(pos0)
    mesh_0.colors = o3d.utility.Vector3dVector(color_coded_features[:F0.shape[0]])

    if isinstance(pos0, torch.Tensor):
        pos1 = pos1.cpu().numpy()
    mesh_1.points = o3d.utility.Vector3dVector(pos1)
    mesh_1.colors = o3d.utility.Vector3dVector(color_coded_features[F0.shape[0]:])

    replaced_mesh_0 = o3d.geometry.PointCloud()
    replaced_mesh_0.points = o3d.utility.Vector3dVector(pos0)
    replaced_mesh_0.colors = o3d.utility.Vector3dVector(color_coded_features[F0.shape[0]:][nn_indices_in_1])

    return mesh_0, mesh_1, replaced_mesh_0


def interpolate_color_from_point_to_mesh(mesh, point_cloud):
    mesh_vertices = np.asarray(mesh.vertices)
    pc_points = np.asarray(point_cloud.points)
    pc_coded_feats = np.asarray(point_cloud.colors)
    tree = cKDTree(pc_points)
    dists, indexes = tree.query(x=mesh_vertices, k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(pc_coded_feats[indexes].astype(np.float64))
    return mesh


def process_sample(dict_data, input_features):
    coords = dict_data['vertices']
    edges = dict_data['edges']
    traces = dict_data['traces']

    sample = Data(x=torch.from_numpy(input_features).float(),
                  pos=coords[0],
                  edge_index=edges[0].t().contiguous())

    nested_meshes = []
    for level in range(1, len(edges)):
        data = Data(edge_index=edges[level].t().contiguous())
        data.trace_index = traces[level - 1]
        nested_meshes.append(data)

    sample.num_vertices = [torch.tensor(coords[0].shape[0])]
    for level, nested_mesh in enumerate(nested_meshes):
        setattr(
            sample, f"hierarchy_edge_index_{level + 1}", nested_mesh.edge_index)
        setattr(
            sample, f"hierarchy_trace_index_{level + 1}", nested_mesh.trace_index)
        sample.num_vertices.append(torch.tensor(coords[level + 1].shape[0]))

    return sample


def generate_and_save_hierarchy_mesh(vertices, num_level, edge_length, grid_factor):
    edge_output_list = list()
    traces_list = list()
    coords_list = list()

    coords_list.append(vertices)
    tree = cKDTree(vertices)
    radius_indexes = tree.query_ball_tree(other=tree, r=grid_factor * edge_length)
    edge_output_list.append(obtain_edge_pairs(vertices=vertices, radius_indexes=radius_indexes))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)

    prev_point_cloud = point_cloud
    for level in range(num_level):
        edge_length = edge_length * 2.0
        point_cloud = prev_point_cloud.voxel_down_sample(voxel_size=edge_length)
        tree = cKDTree(np.asarray(point_cloud.points))
        # TODO: The query operation will fill in the rest of neighbors with the total number of samples in
        #  the tree if it cannot find anymore neighbors. This behavior is not warned!!!
        radius_indexes = tree.query_ball_tree(other=tree, r=grid_factor * edge_length)
        _, trace_scatter = tree.query(x=np.asarray(prev_point_cloud.points), k=1)
        edge_output_list.append(obtain_edge_pairs(vertices=np.asarray(point_cloud.points),
                                                  radius_indexes=radius_indexes))
        coords_list.append(np.asarray(point_cloud.points))
        traces_list.append(trace_scatter)
        prev_point_cloud = point_cloud

    pt_data = dict()
    pt_data['vertices'] = [torch.from_numpy(coords_list[i]).float() for i in range(len(coords_list))]
    pt_data['edges'] = [torch.from_numpy(
        edge_output_list[i]).long() for i in range(0, len(edge_output_list))]
    pt_data['traces'] = [torch.from_numpy(x).long() for x in traces_list]

    return pt_data


def obtain_edge_pairs(vertices, radius_indexes):
    temp = list()
    for i in range(vertices.shape[0]):
        indexes = radius_indexes[i]
        for j in range(len(indexes)):
            temp.append([i, indexes[j]])
    return np.asarray(temp)


def colorize_points_with_descriptor(F0, F1, pos0, pos1, color_map=cv2.COLORMAP_HOT, T_est=None):
    fit = umap.UMAP(
        n_neighbors=20,
        min_dist=0.01,
        n_components=1,
        metric='euclidean'
    )
    if isinstance(F0, torch.Tensor):
        feats = torch.cat([F0, F1], dim=0).cpu().numpy().astype(np.float32)
    elif isinstance(F0, np.ndarray):
        feats = np.concatenate([F0, F1], axis=0).astype(np.float32)
    else:
        raise NotImplementedError("not supported type for F0 and F1")

    if isinstance(pos0, torch.Tensor):
        pos0 = pos0.cpu().numpy()
        pos1 = pos1.cpu().numpy()

    color_coded_features = fit.fit_transform(feats)
    color_max = np.amax(color_coded_features, axis=0, keepdims=True)
    color_min = np.amin(color_coded_features, axis=0, keepdims=True)
    color_coded_features = (color_coded_features - color_min) / (color_max - color_min)
    color_coded_features = cv2.applyColorMap(src=(color_coded_features * 255).astype(np.uint8), colormap=color_map)
    color_coded_features = cv2.cvtColor(src=color_coded_features, code=cv2.COLOR_BGR2RGB)
    color_coded_features = (color_coded_features / 255.0).astype(np.float64).reshape((-1, 3))

    mesh_0 = o3d.geometry.PointCloud()
    mesh_1 = o3d.geometry.PointCloud()

    if isinstance(pos0, torch.Tensor):
        pos0 = pos0.cpu().numpy()
    mesh_0.points = o3d.utility.Vector3dVector(pos0)
    mesh_0.colors = o3d.utility.Vector3dVector(color_coded_features[:F0.shape[0]])

    if isinstance(pos0, torch.Tensor):
        pos1 = pos1.cpu().numpy()
    mesh_1.points = o3d.utility.Vector3dVector(pos1)
    mesh_1.colors = o3d.utility.Vector3dVector(color_coded_features[F0.shape[0]:])

    if T_est is not None:
        pos1_est = pos1 @ T_est[:3, :3].T + T_est[:3, 3]
        transformed_mesh_1 = o3d.geometry.PointCloud()
        transformed_mesh_1.points = o3d.utility.Vector3dVector(pos1_est)
        transformed_mesh_1.colors = o3d.utility.Vector3dVector(color_coded_features[F0.shape[0]:])

        return mesh_0, mesh_1, transformed_mesh_1
    else:
        return mesh_0, mesh_1


def corr_dist(est, gth, xyz0):
    xyz0_est = xyz0 @ est[:3, :3].T + est[:3, 3]
    xyz0_gth = xyz0 @ gth[:3, :3].T + gth[:3, 3]
    dists = np.sqrt(np.sum((xyz0_est - xyz0_gth) ** 2, axis=1))
    return dists.mean(), dists.std()


def pair_wise_matching(xyz0, xyz1, F0, F1, nn_max_n, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]


def find_nn_gpu(F0, F1, nn_max_n=-1, return_distance=False, dist_type='L2'):
    # Too much memory if F0 or F1 are large. Divide the F0
    if nn_max_n > 1:
        N = len(F0)
        C = int(np.ceil(N / nn_max_n))
        stride = nn_max_n
        dists, inds = [], []
        for i in range(C):
            dist = pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        if C * stride < N:
            dist = pdist(F0[C * stride:], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        dists = torch.cat(dists)
        inds = torch.cat(inds)
        assert len(inds) == N
    else:
        dist = pdist(F0, F1, dist_type=dist_type)
        min_dist, inds = dist.min(dim=1)
        dists = min_dist.detach().unsqueeze(1).cpu()
        inds = inds.cpu()
    if return_distance:
        return inds, dists
    else:
        return inds


def find_nn_cpu(F0, F1):
    F0 = F0.cpu().numpy()
    F1 = F1.cpu().numpy()
    tree = cKDTree(F1)
    _, inds = tree.query(F0, k=1)
    return inds


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        l, c = A.size()
        A = A.reshape(l, 1, c)
        l, c = B.size()
        B = B.reshape(1, l, c)
        D2 = torch.sum((A - B).pow(2), dim=2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        l, c = A.size()
        A = A.reshape(l, 1, c)
        l, c = B.size()
        B = B.reshape(1, l, c)
        D2 = torch.sum((A - B).pow(2), dim=2)
        return D2
    else:
        raise NotImplementedError('Not implemented')


def from_graph_data_list(data_list, follow_batch=[]):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    # XL: Assuming there are two samples in a single iteration in each of the data in mini-batch
    # XL: data_list is a list of mini-batch sample pairs
    batch_0 = _batch_processing(data_list, 0, follow_batch)
    batch_1 = _batch_processing(data_list, 1, follow_batch)

    return batch_0, batch_1


def __cumsum__(key, value):
    return bool(re.search('(index|face)', key))


def __cat_dim__(key, value):
    return -1 if bool(re.search('(index|face)', key)) else 0


def _batch_processing(data_list, idx, follow_batch):
    keys = [set(data[idx].keys) for data in data_list]
    keys = list(set.union(*keys))
    assert 'batch' not in keys

    batch = dataset.Batch()

    for key in keys:
        batch[key] = []
    for key in follow_batch:
        batch['{}_batch'.format(key)] = []
    batch.batch = []
    cumsum = 0

    hierarchy_cumsum = [0 for _ in range(len(data_list[0][idx].num_vertices))]
    valid_cumsum = 0

    for i, data in enumerate(data_list):
        data = data[idx]
        num_nodes = data.num_nodes
        batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
        for key in data.keys:
            item = data[key]

            if 'hierarchy' in key:
                # TODO: XL: this assumes the number of levels should not exceed 9
                level = int(key[-1]) - 1
                item = item + hierarchy_cumsum[level] if __cumsum__(key, item) else item
            elif 'quantized_pos' in key:
                item = torch.cat([(i * torch.ones(item.shape[0], 1)).int(), item], dim=1)
            elif key.startswith('original_index'):
                if type(item) != list:
                    item = item + valid_cumsum if __cumsum__(key, item) else item
                else:
                    item[0] = item[0] + valid_cumsum if __cumsum__(key, item[0]) else item[0]
                    for i in range(1, len(item)):
                        item[i] = item[i] + hierarchy_cumsum[i - 1] if __cumsum__(key, item[i]) else item[i]
            elif key.startswith('full_mesh_index_recover'):
                pass
            elif key.startswith('full_mesh_labels'):
                pass
            else:
                item = item + cumsum if __cumsum__(key, item) else item

            batch[key].append(item)

        for key in follow_batch:
            size = data[key].size(__cat_dim__(key, data[key]))
            item = torch.full((size,), i, dtype=torch.long)
            batch['{}_batch'.format(key)].append(item)

        cumsum += num_nodes

        for i in range(len(data.num_vertices)):
            hierarchy_cumsum[i] += data.num_vertices[i]

        valid_cumsum += num_nodes

    for key in keys:
        item = batch[key][0]
        if key.startswith('full_mesh_index_recover'):
            pass
        elif key.startswith('full_mesh_labels'):
            pass
        elif torch.is_tensor(item):
            batch[key] = torch.cat(
                batch[key], dim=__cat_dim__(key, item))
        elif isinstance(item, int) or isinstance(item, float):
            batch[key] = torch.tensor(batch[key])
        else:
            pass
    batch.batch = torch.cat(batch.batch, dim=-1)
    return batch.contiguous()


def clear_folder(folder: str):
    """create temporary empty folder.
    If it already exists, all containing files will be removed.

    Arguments:
        folder {[str]} -- Path to the empty folder
    """
    if not os.path.exists(os.path.dirname(folder)):
        os.makedirs(os.path.dirname(folder))

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


class SimilarityTransform(object):
    def __init__(self, matrix=None):
        if matrix is not None:
            if matrix.shape == (4, 4):
                self.params = matrix
            else:
                raise ValueError("Invalid shape of transformation matrix.")
        else:
            # default to identity matrix
            self.params = np.eye(4)

    def estimate(self, src, dst):
        self.params = _umeyama(src, dst, True)
        return True

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/3)
        return np.linalg.det(self.params) ** (1 / 3)

    @property
    def rotation(self):
        return self.params[0:3, 0:3] / np.linalg.det(self.params) ** (1 / 3)

    @property
    def translation(self):
        return self.params[0:3, 3]

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y, z = np.transpose(coords)
        src = np.vstack((x, y, z, np.ones_like(x)))
        dst = src.T @ matrix.T

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        dst[dst[:, 3] == 0, 3] = np.finfo(float).eps
        # rescale to homogeneous coordinates
        dst[:, :3] /= dst[:, 3:4]

        return dst[:, :3]

    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.
        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.
        Parameters
        ----------
        src : (N, 3) array
            Source coordinates.
        dst : (N, 3) array
            Destination coordinates.
        Returns
        -------
        residuals : (N, ) array
            Residual for coordinate.
        """
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))

    def __call__(self, coords):
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        return self._apply_mat(coords, self._inv_matrix)

    def __nice__(self):
        npstring = np.array2string(self.params, separator=', ')
        paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
        return paramstr

    def __repr__(self):
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return '<{}({}) at {}>'.format(classstr, paramstr, hex(id(self)))

    def __str__(self):
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return '<{}({})>'.format(classstr, paramstr)


def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts


def find_corr(xyz0, xyz1, F0, F1, nn_max_n, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]


def find_corr_with_indexes_no_cycle(xyz0, xyz1, F0, F1):
    # Find the nearest neighbor of points in point cloud 0 in point cloud 1
    nn_inds = find_nn_cpu(F0, F1)
    return xyz0, xyz1[nn_inds], nn_inds


def find_corr_with_indexes(xyz0, xyz1, F0, F1, nn_max_n, cycle_threshold):
    # Find the nearest neighbor of points in point cloud 0 in point cloud 1
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
    cycle_nn_inds = find_nn_gpu(F1[nn_inds], F0, nn_max_n=nn_max_n)
    xyz0_np = xyz0.cpu().numpy()
    inlier_inds = np.argwhere(
        np.sqrt(np.sum((xyz0_np - xyz0_np[cycle_nn_inds]) ** 2, axis=1)) < cycle_threshold).reshape((-1,))

    return xyz0[inlier_inds], xyz1[nn_inds[inlier_inds]], inlier_inds, nn_inds[inlier_inds].numpy().reshape((-1,))


def evaluate_hit_ratio(xyz0, xyz1, T_gth, thresh):
    xyz0 = apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1) ** 2).sum(1) + 1e-6)
    return np.mean((dist < thresh).astype(np.float32))


def rot_x(x):
    out = torch.zeros((3, 3))
    c = torch.cos(x)
    s = torch.sin(x)
    out[0, 0] = 1
    out[1, 1] = c
    out[1, 2] = -s
    out[2, 1] = s
    out[2, 2] = c
    return out


def rot_y(x):
    out = torch.zeros((3, 3))
    c = torch.cos(x)
    s = torch.sin(x)
    out[0, 0] = c
    out[0, 2] = s
    out[1, 1] = 1
    out[2, 0] = -s
    out[2, 2] = c
    return out


def rot_z(x):
    out = torch.zeros((3, 3))
    c = torch.cos(x)
    s = torch.sin(x)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    out[2, 2] = 1
    return out


def get_trans(x):
    trans = torch.eye(4)
    trans[:3, :3] = rot_z(x[2]).mm(rot_y(x[1])).mm(rot_x(x[0]))
    trans[:3, 3] = x[3:, 0]
    return trans


def update_pcd(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = torch.t(R @ torch.t(pts)) + T
    return pts


def build_linear_system(pts0, pts1, weight):
    npts0 = pts0.shape[0]
    A0 = torch.zeros((npts0, 6))
    A1 = torch.zeros((npts0, 6))
    A2 = torch.zeros((npts0, 6))
    A0[:, 1] = pts0[:, 2]
    A0[:, 2] = -pts0[:, 1]
    A0[:, 3] = 1
    A1[:, 0] = -pts0[:, 2]
    A1[:, 2] = pts0[:, 0]
    A1[:, 4] = 1
    A2[:, 0] = pts0[:, 1]
    A2[:, 1] = -pts0[:, 0]
    A2[:, 5] = 1
    ww1 = weight.repeat(3, 6)
    ww2 = weight.repeat(3, 1)
    A = ww1 * torch.cat((A0, A1, A2), 0)
    b = ww2 * torch.cat(
        (pts1[:, 0] - pts0[:, 0], pts1[:, 1] - pts0[:, 1], pts1[:, 2] - pts0[:, 2]),
        0,
    ).unsqueeze(1)
    return A, b


def solve_linear_system(A, b):
    temp = torch.inverse(A.t().mm(A))
    return temp.mm(A.t()).mm(b)


def compute_weights(pts0, pts1, par):
    return par / (torch.norm(pts0 - pts1, dim=1).unsqueeze(1) + par)


def est_quad_linear_robust(pts0, pts1, weight=None):
    pts0_curr = pts0
    trans = torch.eye(4)

    par = 1.0  # Todo: need to decide
    if weight is None:
        weight = torch.ones(pts0.size()[0], 1)

    for i in range(20):
        if i > 0 and i % 5 == 0:
            par /= 2.0

        # compute weights
        A, b = build_linear_system(pts0_curr, pts1, weight)
        x = solve_linear_system(A, b)
        trans_curr = get_trans(x)
        pts0_curr = update_pcd(pts0_curr, trans_curr)
        weight = compute_weights(pts0_curr, pts1, par)
        trans = trans_curr.mm(trans)

    return trans


def pose_estimation(model,
                    device,
                    xyz0,
                    xyz1,
                    coord0,
                    coord1,
                    feats0,
                    feats1,
                    return_corr=False):
    sinput0 = ME.SparseTensor(feats0, coordinates=coord0, device=device)
    F0 = model(sinput0).F

    sinput1 = ME.SparseTensor(feats1, coordinates=coord1, device=device)
    F1 = model(sinput1).F

    corr = F0.mm(F1.t())
    weight, inds = corr.max(dim=1)
    weight = weight.unsqueeze(1).cpu()
    xyz1_corr = xyz1[inds, :]

    trans = est_quad_linear_robust(xyz0, xyz1_corr, weight)

    if return_corr:
        return trans, weight, corr
    else:
        return trans, weight


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.avg = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls
        if average:
            return self.avg
        else:
            return self.diff


def make_open3d_feature(data, dim, npts):
    feature = o3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def save_checkpoint(epoch, step, model, config, path):
    state = {
        'epoch': epoch,
        'step': step,
        'model': model.state_dict(),
        'config': config
    }
    torch.save(state, str(path))


def validation(model, data_loader, max_iter, hit_ratio_thresh, nn_max_n, cur_epoch, writer, device):
    model.eval()
    data_loader.dataset.randg.seed(1)
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, rse_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    tot_num_data = len(data_loader.dataset)
    tot_num_data = min(max_iter, tot_num_data)
    data_loader_iter = data_loader.__iter__()

    tq = tqdm.tqdm(total=len(data_loader))
    tq.set_description('Validation - Epoch {}'.format(cur_epoch))
    for batch_idx in range(tot_num_data):
        input_dict = data_loader_iter.next()

        with torch.no_grad():
            sinput0 = ME.SparseTensor(
                input_dict['sinput0_F'], coordinates=input_dict['sinput0_C'], device=device)
            F0 = model(sinput0).F

            sinput1 = ME.SparseTensor(
                input_dict['sinput1_F'], coordinates=input_dict['sinput1_C'], device=device)
            F1 = model(sinput1).F

        xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
        xyz1_corr, xyz0_corr = find_corr(xyz1, xyz0, F1, F0, nn_max_n=nn_max_n, subsample_size=-1)
        xyz0_corr = xyz0_corr.cpu().numpy()
        xyz1_corr = xyz1_corr.cpu().numpy()

        tq.set_description('Validation - RANSAC')
        registration_model, inliers = ransac(data=(xyz0_corr, xyz1_corr),
                                             model_class=SimilarityTransform, min_samples=30,
                                             residual_threshold=5.0,
                                             max_trials=200)
        T_est = registration_model.params
        T_gt = T_gt.cpu().numpy()

        tq.set_description('Validation - Loss calculation')
        # Transform point cloud 0 to est and gt pose, and calculate point-wise distance as an evaluation metric here
        loss = corr_dist(T_est, T_gt, xyz0, weight=None)
        loss_meter.update(loss)

        rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        rte_meter.update(rte)

        rre = np.arccos(
            (np.trace(registration_model.rotation.T @ (T_gt[:3, :3] / np.linalg.det(T_gt) ** (1 / 3))) - 1) / 2)
        if not np.isnan(rre):
            rre_meter.update(rre)

        rse = registration_model.scale / np.linalg.det(T_gt) ** (1 / 3)
        rse_meter.update(rse)

        hit_ratio = evaluate_hit_ratio(
            xyz0_corr, xyz1_corr, T_gt, thresh=hit_ratio_thresh)
        hit_ratio_meter.update(hit_ratio)

        torch.cuda.empty_cache()

        tq.update(1)
        tq.set_postfix(loss='avg: {:.3f}, cur: {:.3f}'.format(loss_meter.avg, loss),
                       rte='avg: {:.3f}, cur: {:.3f}'.format(rte_meter.avg, rte),
                       rre='avg: {:.3f}, cur: {:.3f}'.format(rre_meter.avg, rre),
                       rse='avg: {:.3f}, cur: {:.3f}'.format(rse_meter.avg, rse),
                       hit_ratio='avg: {:.3f}, cur: {:.3f}'.format(hit_ratio_meter.avg, hit_ratio),
                       )

    writer.add_scalar('validation/loss', loss_meter.avg, cur_epoch)
    writer.add_scalar('validation/rte', rte_meter.avg, cur_epoch)
    writer.add_scalar('validation/rre', rre_meter.avg, cur_epoch)
    writer.add_scalar('validation/rse', rse_meter.avg, cur_epoch)
    writer.add_scalar('validation/hit_ratio', hit_ratio_meter.avg, cur_epoch)
    tq.close()
    return loss_meter.avg, rte_meter.avg, rre_meter.avg, rse_meter.avg, hit_ratio_meter.avg


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def calc_correspondence_deviation(gt_indexes, est_indexes, xyz, inlier_threshold):
    # Assume all points in xyz0 have tried to find the best matches in xyz1
    # Assume the order of elements in gt_pairs and est_pairs are the same
    assert (gt_indexes.shape[0] == est_indexes.shape[0])
    point_wise_matching_deviation = np.sqrt(np.sum((xyz[gt_indexes] - xyz[est_indexes]) ** 2, axis=1))

    fm_recall = np.sum((point_wise_matching_deviation.reshape((-1,)) < inlier_threshold).astype(np.float32)) / \
                gt_indexes.shape[0]
    matching_deviation = np.sum(point_wise_matching_deviation) / gt_indexes.shape[0]
    return matching_deviation, fm_recall


def validation_point_matching_evaluation(model, data_loader, max_iter, hit_ratio_thresh, nn_max_n, cur_epoch,
                                         writer, device):
    model.eval()
    data_loader.dataset.randg.seed(1)
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, rse_meter, md_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    tot_num_data = len(data_loader.dataset)
    tot_num_data = min(max_iter, tot_num_data)
    data_loader_iter = data_loader.__iter__()

    tq = tqdm.tqdm(total=len(data_loader))
    tq.set_description('Validation - Epoch {}'.format(cur_epoch))
    for batch_idx in range(tot_num_data):
        input_dict = data_loader_iter.next()

        with torch.no_grad():
            sinput0 = ME.SparseTensor(
                input_dict['sinput0_F'], coordinates=input_dict['sinput0_C'], device=device)
            F0 = model(sinput0).F

            sinput1 = ME.SparseTensor(
                input_dict['sinput1_F'], coordinates=input_dict['sinput1_C'], device=device)
            F1 = model(sinput1).F

        xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
        gt_pairs = input_dict['correspondences'].numpy()
        gt_pairs = gt_pairs[np.argsort(gt_pairs[:, 1])]
        # inlier_indexes here is the point index in the point cloud
        # inlier_indexes_1 is in ascending order
        xyz1_corr, xyz0_corr, inlier_indexes_1, est_inlier_indexes_0 = find_corr_with_indexes(xyz1, xyz0, F1, F0,
                                                                                              cycle_threshold=5.0,
                                                                                              nn_max_n=nn_max_n)

        inlier_indexes_of_gt_indexes = []
        est_inlier_indexes_0_with_gt = []
        last_pos = 0
        for i in range(gt_pairs[:, 1].shape[0]):
            ind = gt_pairs[i, 1]

            while last_pos < inlier_indexes_1.shape[0]:
                if inlier_indexes_1[last_pos] == ind:
                    inlier_indexes_of_gt_indexes.append(i)
                    est_inlier_indexes_0_with_gt.append(est_inlier_indexes_0[last_pos])
                    last_pos += 1
                    break
                elif inlier_indexes_1[last_pos] < ind:
                    last_pos += 1
                else:
                    break

        est_inlier_indexes_0_with_gt = np.asarray(est_inlier_indexes_0_with_gt).reshape((-1,))
        matching_deviation = calc_correspondence_deviation(
            gt_pairs[inlier_indexes_of_gt_indexes, 0].reshape((-1,)),
            est_inlier_indexes_0_with_gt,
            xyz0.numpy())
        md_meter.update(matching_deviation)
        xyz0_corr = xyz0_corr.numpy().reshape((-1, 3))
        xyz1_corr = xyz1_corr.numpy().reshape((-1, 3))

        tq.set_description('Validation - RANSAC')
        registration_model, inliers = ransac(data=(xyz0_corr, xyz1_corr),
                                             model_class=SimilarityTransform, min_samples=4,
                                             residual_threshold=5.0,
                                             max_trials=200)
        T_est = registration_model.params
        T_gt = T_gt.numpy()

        tq.set_description('Validation - Loss calculation')
        # Transform point cloud 0 to est and gt pose, and calculate point-wise distance as an evaluation metric here
        loss = corr_dist(T_est, T_gt, xyz0, weight=None)
        loss_meter.update(loss)

        rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        rte_meter.update(rte)

        rre = np.arccos(
            (np.trace(registration_model.rotation.T @ (T_gt[:3, :3] / np.linalg.det(T_gt) ** (1 / 3))) - 1) / 2)
        if not np.isnan(rre):
            rre_meter.update(rre)

        rse = registration_model.scale / np.linalg.det(T_gt) ** (1 / 3)
        rse_meter.update(rse)

        hit_ratio = evaluate_hit_ratio(
            xyz0_corr, xyz1_corr, T_gt, thresh=hit_ratio_thresh)
        hit_ratio_meter.update(hit_ratio)

        torch.cuda.empty_cache()

        tq.update(1)
        tq.set_postfix(loss='avg: {:.3f}, cur: {:.3f}'.format(loss_meter.avg, loss),
                       rte='avg: {:.3f}, cur: {:.3f}'.format(rte_meter.avg, rte),
                       rre='avg: {:.3f}, cur: {:.3f}'.format(rre_meter.avg, rre),
                       rse='avg: {:.3f}, cur: {:.3f}'.format(rse_meter.avg, rse),
                       hit_ratio='avg: {:.3f}, cur: {:.3f}'.format(hit_ratio_meter.avg, hit_ratio),
                       md='avg: {:.3f}, cur: {:.3f}'.format(md_meter.avg, matching_deviation),
                       )

    writer.add_scalar('validation/loss', loss_meter.avg, cur_epoch)
    writer.add_scalar('validation/rte', rte_meter.avg, cur_epoch)
    writer.add_scalar('validation/rre', rre_meter.avg, cur_epoch)
    writer.add_scalar('validation/rse', rse_meter.avg, cur_epoch)
    writer.add_scalar('validation/hit_ratio', hit_ratio_meter.avg, cur_epoch)
    writer.add_scalar('validation/matching_deviation', md_meter.avg, cur_epoch)

    tq.close()
    return loss_meter.avg, rte_meter.avg, rre_meter.avg, rse_meter.avg, hit_ratio_meter.avg, md_meter.avg


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
