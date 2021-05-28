from torch.utils.data import Dataset
import torch
import MinkowskiEngine as ME
from sklearn.neighbors import BallTree
from scipy.stats import loguniform
import copy
import numpy as np
import open3d as o3d
import pyvista, pyacvd
import pickle
from scipy.spatial import cKDTree

from .dataset_utils import sample_random_rotation


class NasalDataset(Dataset):
    def __init__(self, mean_mesh_model_path, partial_mean_mesh_model_path,
                 atlas_mode_weights_path, atlas_mode_weights_std_range, atlas_mode_range,
                 use_rotation, use_remesh,
                 sampling_size, use_crop, default_edge_length,
                 edge_length_range, max_select_trial, phase, oversampling_factor,
                 num_iter, batch_size, crop_ratio_range=None, min_crop_remained_portion=None,
                 scale_range=None, rotate_range=None, subdivide_factor=None,
                 ):
        with open(str(atlas_mode_weights_path), "rb") as f:
            self._atlas_mode_weights = pickle.load(f)

        self._atlas_mode_weights_std_range = atlas_mode_weights_std_range
        self._atlas_mode_range = atlas_mode_range
        self._num_atlas_modes = atlas_mode_range[1] - atlas_mode_range[0]
        self._mean_mesh_model_path = mean_mesh_model_path
        self._mean_mesh_model = o3d.io.read_triangle_mesh(str(mean_mesh_model_path))
        self._partial_mean_mesh_model = o3d.io.read_triangle_mesh(str(partial_mean_mesh_model_path))
        self._atlas_mode_stds = self._atlas_mode_weights["mode_stds"]
        ball_tree = BallTree(np.asarray(self._mean_mesh_model.vertices))
        self._partial_model_indexes = \
            ball_tree.query(X=np.asarray(self._partial_mean_mesh_model.vertices), k=1, return_distance=False)

        self.randg = np.random.RandomState()
        self._use_rotation = use_rotation
        self._use_remesh = use_remesh
        self._scale_range = scale_range
        self._sampling_size = sampling_size
        self._rotate_range = rotate_range
        self._num_iter = num_iter
        self._subdivide_factor = subdivide_factor
        self._use_crop = use_crop
        self._crop_ratio_range = crop_ratio_range
        self._min_crop_remained_portion = min_crop_remained_portion
        self._max_select_trial = max_select_trial
        self._oversampling_factor = oversampling_factor
        self._default_edge_length = default_edge_length
        self._edge_length_range = edge_length_range
        self._phase = phase
        self._batch_size = batch_size

    def __len__(self):
        return self._num_iter * self._batch_size

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            id = str(0)
        else:
            id = str(worker_info.id)
        self.id = id
        while True:
            mesh_model_0 = self.model_deforming(mesh_model=self._mean_mesh_model,
                                                mode_vertices=self._atlas_mode_weights[
                                                    "mode_components"])
            mesh_model_1 = self.partial_model_handling(mesh_model=mesh_model_0)

            area_0 = self.calc_total_surface_area(mesh_model_0)
            area_1 = self.calc_total_surface_area(mesh_model_1)
            if self._use_remesh:
                # rv_continuous.random_state = self.randg
                tgt_edge_length_0 = loguniform(a=self._edge_length_range[0],
                                               b=self._edge_length_range[1]).rvs(size=1)[0]

                # number of cluster is actually the number of faces instead of points!!!!
                num_cluster_0 = area_0 / (np.sqrt(3) / 4 * tgt_edge_length_0 ** 2)
                tgt_edge_length_1 = loguniform(a=self._edge_length_range[0],
                                               b=self._edge_length_range[1]).rvs(size=1)[0]
                num_cluster_1 = area_1 / (np.sqrt(3) / 4 * tgt_edge_length_1 ** 2)

                try:
                    mesh_model_0, edge_length_0, scale_0 = \
                        self.model_remeshing(o3d_mesh=mesh_model_0,
                                             num_cluster=int(num_cluster_0),
                                             tgt_edge_length=tgt_edge_length_0)

                    mesh_model_1, edge_length_1, scale_1 = \
                        self.model_remeshing(o3d_mesh=mesh_model_1,
                                             num_cluster=int(num_cluster_1),
                                             tgt_edge_length=tgt_edge_length_1)
                except (ArithmeticError, KeyError, ValueError) as err:
                    print(err)
                    idx = np.random.randint(low=0, high=self._num_iter)
                    continue

                voxel_size_0 = edge_length_0
                voxel_size_1 = edge_length_1
            else:
                tgt_edge_length_0 = tgt_edge_length_1 = self._default_edge_length
                scale_0 = scale_1 = 1.0
                voxel_size_0 = self._default_edge_length
                voxel_size_1 = self._default_edge_length
                mesh_model_0 = mesh_model_0.compute_vertex_normals(normalized=True)
                mesh_model_1 = mesh_model_1.compute_vertex_normals(normalized=True)

            transform0 = np.eye(4)
            transform1 = np.eye(4)

            if self._use_rotation:
                T0 = sample_random_rotation(self.randg, 360)
                T1 = sample_random_rotation(self.randg, 360)

                mesh_model_0.transform(T0)
                mesh_model_1.transform(T1)

                transform0 = T0 @ transform0
                transform1 = T1 @ transform1

            transform0[:3, :3] = transform0[:3, :3] * scale_0
            transform1[:3, :3] = transform1[:3, :3] * scale_1

            if self._use_crop:
                mesh_model_1 = self.crop_points(mesh_model_1,
                                                axis_aligned_ratio_range=self._crop_ratio_range,
                                                min_remained_portion=self._min_crop_remained_portion)

            xyz0 = np.asarray(mesh_model_0.vertices)
            xyz1 = np.asarray(mesh_model_1.vertices)

            xyz0 = torch.from_numpy(xyz0)
            xyz1 = torch.from_numpy(xyz1)

            sel0 = ME.utils.sparse_quantize(xyz0, return_index=True, quantization_size=voxel_size_0)[1]
            sel1 = ME.utils.sparse_quantize(xyz1, return_index=True, quantization_size=voxel_size_1)[1]

            unique_xyz0 = xyz0[sel0]
            unique_xyz1 = xyz1[sel1]

            relative_transform = transform0 @ np.linalg.inv(transform1)

            transformed_vertices_0 = np.concatenate([unique_xyz0, np.ones((unique_xyz0.shape[0], 1))],
                                                    axis=1) @ np.linalg.inv(transform0).T
            transformed_vertices_0 = transformed_vertices_0[:, :3] / transformed_vertices_0[:, 3].reshape((-1, 1))

            transformed_vertices_1 = np.concatenate([unique_xyz1, np.ones((unique_xyz1.shape[0], 1))],
                                                    axis=1) @ np.linalg.inv(transform1).T
            transformed_vertices_1 = transformed_vertices_1[:, :3] / transformed_vertices_1[:, 3].reshape((-1, 1))

            # Voxelize the coordinates to integers
            coords0 = torch.floor(unique_xyz0 / voxel_size_0)
            coords1 = torch.floor(unique_xyz1 / voxel_size_1)

            count, matches = self.correspondence_generation(1.5 * voxel_size_0 / scale_0, 1.5 * voxel_size_1 / scale_1,
                                                            transformed_vertices_0, transformed_vertices_1)

            feats0 = torch.ones((unique_xyz0.shape[0], 1)).float()
            feats1 = torch.ones((unique_xyz1.shape[0], 1)).float()

            scale_ratio = scale_0 / scale_1

            if count >= self._max_select_trial:
                idx = np.random.randint(low=0, high=self._num_iter)
                continue
            else:
                break
        if self._phase == "train" or self._phase == "val":
            if np.random.random() > 0.5:
                return (unique_xyz0.float(), unique_xyz1.float(), coords0.int(),
                        coords1.int(), feats0.float(), feats1.float(), matches, relative_transform, scale_ratio)
            else:
                return (unique_xyz1.float(), unique_xyz0.float(), coords1.int(),
                        coords0.int(), feats1.float(), feats0.float(),
                        torch.cat([matches[:, 1:2], matches[:, 0:1]], dim=1), np.linalg.inv(relative_transform),
                        1.0 / scale_ratio)
        elif self._phase == "test":
            return (np.asarray(mesh_model_0.vertices), np.asarray(mesh_model_1.vertices),
                    np.asarray(mesh_model_0.triangles), np.asarray(mesh_model_1.triangles),
                    unique_xyz0.float(), unique_xyz1.float(), coords0.int(),
                    coords1.int(), feats0.float(), feats1.float(), matches, relative_transform,
                    max(tgt_edge_length_0, tgt_edge_length_1)
                    )
        else:
            raise AttributeError(f"Phase {self._phase} not supported")

    def partial_model_handling(self, mesh_model):
        partial_mesh_model = copy.deepcopy(self._partial_mean_mesh_model)
        partial_mesh_model.vertices = \
            o3d.utility.Vector3dVector(np.asarray(mesh_model.vertices)[self._partial_model_indexes.reshape((-1,)), :])
        partial_mesh_model = partial_mesh_model.compute_vertex_normals(normalized=True)
        return partial_mesh_model

    def model_deforming(self, mesh_model, mode_vertices):
        deformed_mesh_model = copy.deepcopy(mesh_model)
        mode_weights = np.random.uniform(low=self._atlas_mode_weights_std_range[0],
                                         high=self._atlas_mode_weights_std_range[1],
                                         size=self._num_atlas_modes)
        # TODO: Check the norm of vertex displacements
        deformed_mesh_model.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_model.vertices) + np.sum(
            mode_weights.reshape((-1, 1, 1))[self._atlas_mode_range[0]:self._atlas_mode_range[1]] *
            self._atlas_mode_stds.reshape((-1, 1, 1))[self._atlas_mode_range[0]:self._atlas_mode_range[1]] *
            mode_vertices[self._atlas_mode_range[0]:self._atlas_mode_range[1]],
            axis=0).reshape((-1, 3)))
        deformed_mesh_model = deformed_mesh_model.compute_vertex_normals(normalized=True)
        return deformed_mesh_model

    def crop_points(self, mesh, axis_aligned_ratio_range, min_remained_portion):
        bounding_box = mesh.get_axis_aligned_bounding_box()
        num_vertices = np.asarray(mesh.vertices).shape[0]
        while True:
            min_corner = bounding_box.get_min_bound()
            box_lengths = bounding_box.get_extent()
            mesh_to_crop = copy.deepcopy(mesh)
            # Randomly pick one axis and choose two points to do the cropping
            remained_length = np.random.uniform(low=axis_aligned_ratio_range[0], high=axis_aligned_ratio_range[1])
            offset_pos = np.random.uniform(low=0.0, high=1.0 - remained_length)
            axis_to_crop = np.random.randint(low=0, high=3)

            shifted_min_corner = copy.deepcopy(min_corner)
            shifted_min_corner[axis_to_crop] = min_corner[axis_to_crop] + offset_pos * box_lengths[axis_to_crop]

            shifted_max_corner = min_corner + box_lengths
            shifted_max_corner[axis_to_crop] = shifted_max_corner[axis_to_crop] - (1.0 - remained_length - offset_pos) * \
                                               box_lengths[axis_to_crop]
            crop_bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=shifted_min_corner.reshape((3, 1)),
                                                                    max_bound=shifted_max_corner.reshape((3, 1)))
            cropped_mesh = mesh_to_crop.crop(bounding_box=crop_bounding_box)

            if np.asarray(cropped_mesh.vertices).shape[0] >= min_remained_portion * num_vertices:
                break
            else:
                continue

        return cropped_mesh

    def correspondence_generation(self, edge_length_0, edge_length_1, vertices_0, vertices_1):
        count = 0
        overlap_indexes = list()

        if edge_length_0 <= edge_length_1:
            tree_0 = cKDTree(vertices_0)

            while count < self._max_select_trial:
                sampled_indexes = np.sort(
                    np.random.choice(np.arange(start=0, stop=vertices_1.shape[0]),
                                     size=int(self._sampling_size * self._oversampling_factor),
                                     replace=True))

                counts = tree_0.query_ball_point(x=vertices_1[sampled_indexes], r=edge_length_1,
                                                 return_length=True)

                overlap_indexes.extend(sampled_indexes[np.argwhere(counts > 0).flatten()])

                if len(overlap_indexes) <= self._sampling_size:
                    count += 1
                    continue
                else:
                    overlap_indexes = np.random.choice(list(overlap_indexes), size=self._sampling_size)

                    distances, valid_indexes = tree_0.query(x=vertices_1[overlap_indexes], k=1)

                    assert (np.all(distances <= edge_length_1))
                    break

            if count >= self._max_select_trial:
                return count, []

            valid_indexes_1 = overlap_indexes
            valid_indexes_0 = valid_indexes.flatten()
            matches = torch.from_numpy(
                np.concatenate(
                    [np.asarray(valid_indexes_0).reshape((-1, 1)), np.asarray(valid_indexes_1).reshape((-1, 1))],
                    axis=1))
        else:
            tree_1 = cKDTree(vertices_1)

            while count < self._max_select_trial:
                sampled_indexes = np.sort(
                    np.random.choice(np.arange(start=0, stop=vertices_0.shape[0]),
                                     size=int(self._sampling_size * self._oversampling_factor),
                                     replace=True))

                counts = tree_1.query_ball_point(x=vertices_0[sampled_indexes], r=edge_length_0,
                                                 return_length=True)

                overlap_indexes.extend(sampled_indexes[np.argwhere(counts > 0).flatten()])

                if len(overlap_indexes) <= self._sampling_size:
                    count += 1
                    continue
                else:
                    overlap_indexes = np.random.choice(list(overlap_indexes), size=self._sampling_size)
                    distances, valid_indexes = tree_1.query(x=vertices_0[overlap_indexes], k=1)

                    assert (np.all(distances <= edge_length_0))

                    break

            if count >= self._max_select_trial:
                return count, []

            valid_indexes_0 = overlap_indexes
            valid_indexes_1 = valid_indexes.flatten()

            matches = torch.from_numpy(
                np.concatenate(
                    [np.asarray(valid_indexes_0).reshape((-1, 1)), np.asarray(valid_indexes_1).reshape((-1, 1))],
                    axis=1))

        return count, matches

    @staticmethod
    def from_o3d_to_pyvista_mesh(o3d_mesh):
        mesh = pyvista.PolyData()
        mesh.points = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        mesh.faces = np.concatenate([3 * np.ones((faces.shape[0], 1)), faces], axis=1).astype(np.int32)
        return mesh

    @staticmethod
    def from_pyvista_to_o3d_mesh(vista_mesh):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vista_mesh.points))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(vista_mesh.faces).reshape((-1, 4))[..., 1:])

        return o3d_mesh

    @staticmethod
    def calc_edge_length(o3d_mesh):
        faces = np.asarray(o3d_mesh.triangles)
        vertices = np.asarray(o3d_mesh.vertices)
        avg_edge_length = 1.0 / 3 * \
                          (np.mean(
                              np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 1], :]) ** 2, axis=1))) +
                           np.mean(
                               np.sqrt(np.sum((vertices[faces[:, 1], :] - vertices[faces[:, 2], :]) ** 2, axis=1))) +
                           np.mean(np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 2], :]) ** 2, axis=1))))
        return avg_edge_length

    @staticmethod
    def calc_edge_length_vista(vista_mesh):
        faces = np.asarray(vista_mesh.faces).reshape((-1, 4))[:, 1:]
        vertices = np.asarray(vista_mesh.points)
        avg_edge_length = 1.0 / 3 * \
                          (np.mean(
                              np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 1], :]) ** 2, axis=1))) +
                           np.mean(
                               np.sqrt(np.sum((vertices[faces[:, 1], :] - vertices[faces[:, 2], :]) ** 2, axis=1))) +
                           np.mean(np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 2], :]) ** 2, axis=1))))
        return avg_edge_length

    @staticmethod
    def calc_total_surface_area(o3d_mesh):
        faces = np.asarray(o3d_mesh.triangles)
        vertices = np.asarray(o3d_mesh.vertices)
        side_0 = np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 1], :]) ** 2, axis=1))
        side_1 = np.sqrt(np.sum((vertices[faces[:, 1], :] - vertices[faces[:, 2], :]) ** 2, axis=1))
        side_2 = np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 2], :]) ** 2, axis=1))
        p = 0.5 * (side_0 + side_1 + side_2)
        surface_area = np.sum(np.sqrt(p * (p - side_0) * (p - side_1) * ((p - side_2))))
        return surface_area

    def model_remeshing(self, o3d_mesh, num_cluster, tgt_edge_length):
        bin = 5000
        cluster_id = int(np.ceil(num_cluster / bin))
        mesh = self.from_o3d_to_pyvista_mesh(o3d_mesh)
        mesh_cluster = pyacvd.Clustering(mesh)

        # Subdivide the surface so that it reaches the maximum number of faces in this bin
        ratio = int(np.ceil(self._subdivide_factor * cluster_id * bin / (4.0 * mesh.n_faces)))
        mesh_cluster.subdivide(nsub=ratio)
        # This should not be smaller than the requirement anymore
        assert mesh_cluster.mesh.n_faces >= self._subdivide_factor * num_cluster, \
            f"ori {mesh.n_faces}, src {mesh_cluster.mesh.n_faces}, tgt {self._subdivide_factor * num_cluster}"

        mesh_cluster.cluster(num_cluster)
        remeshed_model = mesh_cluster.create_mesh(flipnorm=False)
        # Recalibrate to try to get the target edge length
        temp_length = self.calc_edge_length_vista(remeshed_model)
        correct_ratio = (tgt_edge_length / temp_length) ** 2

        if np.isnan(num_cluster / correct_ratio):
            raise ArithmeticError("NAN found in number_cluster / ratio")

        mesh_cluster.cluster(num_cluster / correct_ratio)
        remeshed_model = mesh_cluster.create_mesh(flipnorm=True)
        o3d_mesh = self.from_pyvista_to_o3d_mesh(remeshed_model)

        # Do the cleaning and repairing here!
        o3d_mesh = o3d_mesh.remove_non_manifold_edges()
        o3d_mesh = o3d_mesh.remove_degenerate_triangles()
        o3d_mesh = o3d_mesh.remove_duplicated_triangles()
        o3d_mesh = o3d_mesh.remove_duplicated_vertices()
        o3d_mesh = o3d_mesh.remove_unreferenced_vertices()
        o3d_mesh = o3d_mesh.compute_vertex_normals(normalized=True)
        avg_edge_length = self.calc_edge_length(o3d_mesh)

        # Rescale the o3d mesh to default edge length by scaling its vertex values
        scale = self._default_edge_length / avg_edge_length
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(o3d_mesh.vertices) * scale)
        avg_edge_length = self.calc_edge_length(o3d_mesh)

        return o3d_mesh, avg_edge_length, scale

    def augment_mesh(self, mesh_model):
        # o3d transform ONLY transform geometry coordinates. normals need to be handled separately
        mesh_model_0 = copy.deepcopy(mesh_model)
        transform = np.eye(4)
        if self._use_rotation:
            transform = sample_random_rotation(self.randg, self._rotate_range)
            mesh_model_0.vertex_normals = o3d.utility.Vector3dVector(
                np.asarray(mesh_model_0.vertex_normals) @ transform[:3, :3].T)

        vertices = np.asarray(mesh_model_0.vertices)
        vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
        vertices = vertices @ transform.T
        mesh_model_0.vertices = o3d.utility.Vector3dVector(vertices[:, :3])
        return mesh_model_0, transform
