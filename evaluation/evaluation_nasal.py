import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d
from multiprocessing import Process, Queue
from .eval_utils import extract_features, calc_total_surface_area, extract_mesh_features
from pathlib import Path
import torch
import copy
from scipy.spatial import cKDTree
import pickle
import json
import tqdm
from sklearn.neighbors import BallTree
import scipy
import random

import utils
import models
from eval_utils import summarize_resolution_mismatch_results


class FeatureMatchEvaluation(object):
    def __init__(self, output_root,
                 default_inlier_dist_thresh,
                 default_voxel_size, num_rand_keypoints, voxel_size_list,
                 vary_scale, use_remesh, overwrite_result,
                 atlas_mode_weights_path,
                 atlas_mode_weights_std_range, atlas_mode_range,
                 mean_mesh_model_path, partial_mean_mesh_model_path, num_run,
                 crop_ratio_range, device, inlier_ratio_thresholds):
        self._crop_ratio_range = crop_ratio_range
        self._output_root = output_root
        self._default_voxel_size = default_voxel_size
        self._default_inlier_dist_thresh = default_inlier_dist_thresh
        self._voxel_size_list = voxel_size_list
        self._num_rand_keypoints = num_rand_keypoints
        self._vary_scale = vary_scale
        self._overwrite_result = overwrite_result
        self._remesh_surface = use_remesh
        self._num_run = num_run
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
        self._device = device
        self._inlier_ratio_thresholds = inlier_ratio_thresholds
        self.randg = np.random.RandomState()
        if not self._vary_scale:
            # If not varying scale, each run should only have a single scale for the sample pair
            assert (len(voxel_size_list) == 1)

    def find_nn_cpu(self, feat0, feat1):
        feat1tree = cKDTree(feat1)
        dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
        putative_match_indexes = None
        return nn_inds, putative_match_indexes

    def valid_feat_ratio(self, coord0, coord1, feat0, feat1, trans_gth, thresh):
        coord0_copy = copy.deepcopy(coord0)
        coord0_copy = coord0_copy.transform(trans_gth)
        nn_inds, putative_inds_of_0 = self.find_nn_cpu(feat0, feat1)
        dist = np.sqrt(((np.array(coord0_copy.points) - np.array(coord1.points)[nn_inds]) ** 2).sum(1))
        return np.mean(dist < thresh)

    # Rotation matrix along axis with angle theta
    @staticmethod
    def M(axis, theta):
        return scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta))

    @staticmethod
    def sample_random_trans(randg, rotation_range=360):
        T = np.eye(4)
        R = FeatureMatchEvaluation.M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
        T[:3, :3] = R
        return T

    @staticmethod
    def make_open3d_point_cloud(xyz, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd

    @staticmethod
    def apply_transform(pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def do_single_pair_evaluation(self, queue, coord_i, coord_j, feat_i, feat_j,
                                  gt_pose, inlier_dist_thresh):
        Ni, Nj = len(coord_i), len(coord_j)

        # Only subsample points that serve as the query instead of the target
        if self._num_rand_keypoints > 0:
            num_sample_j = min(Nj, self._num_rand_keypoints)
            inds_j = np.random.choice(Nj, num_sample_j, replace=False)
            coord_j, feat_j = coord_j[inds_j], feat_j[inds_j]

        coord_i = self.make_open3d_point_cloud(coord_i)
        coord_j = self.make_open3d_point_cloud(coord_j)

        try:
            temp = self.valid_feat_ratio(coord0=coord_j, coord1=coord_i, feat0=feat_j, feat1=feat_i,
                                         trans_gth=gt_pose, thresh=inlier_dist_thresh)
            hit_ratio = temp
            queue.put(hit_ratio)
            return

        except IndexError or ZeroDivisionError:
            print(f"number of features in sample i and j: {Ni} {Nj} ")
            queue.put(0.0)
            return

    def feature_evaluation(self, model):
        if not self._overwrite_result and (self._output_root / f"eval_result.npy").exists():
            return

        hit_ratio_list = []
        hit_ratio_meter = utils.AverageMeter()
        tq = tqdm.tqdm(total=self._num_run)
        self.randg.seed(0)
        np.random.seed(0)

        for i in range(self._num_run):
            tq.update(1)
            ori_mesh_0 = self.model_deforming(mesh_model=self._mean_mesh_model,
                                              mode_vertices=self._atlas_mode_weights[
                                                  "mode_components"])
            ori_mesh_1 = self.partial_model_handling(mesh_model=ori_mesh_0)

            remeshed_0_list = list()
            remeshed_1_list = list()
            length_0_list = list()
            length_1_list = list()
            scale_0_list = list()
            scale_1_list = list()

            T0 = self.sample_random_trans(randg=self.randg, rotation_range=360)
            T1 = self.sample_random_trans(randg=self.randg, rotation_range=360)

            for m in range(len(self._voxel_size_list)):
                if self._remesh_surface:
                    # pre-process and extract features from mesh based on voxel size and required feature type
                    tgt_edge_length = self._voxel_size_list[m]
                    mesh_area = calc_total_surface_area(ori_mesh_0)
                    num_cluster_guess = mesh_area / (np.sqrt(3) / 4 * tgt_edge_length ** 2)

                    mesh_0 = None
                    avg_edge_length_0 = None
                    scale_0 = None
                    while True:
                        try:
                            mesh_0, avg_edge_length_0, scale_0 = \
                                extract_mesh_features(o3d_mesh=ori_mesh_0,
                                                      tgt_edge_length=tgt_edge_length,
                                                      num_cluster=num_cluster_guess,
                                                      subdivide_factor=2.0,
                                                      default_edge_length=self._default_voxel_size)
                            break
                        except ValueError as err:
                            print(err)
                            continue

                    xyz0 = np.asarray(mesh_0.vertices)
                    xyz0 = self.apply_transform(xyz0, T0)
                    remeshed_0_list.append(xyz0)
                    length_0_list.append(avg_edge_length_0)
                    scale_0_list.append(scale_0)
                    # pre-process and extract features from mesh based on voxel size and required feature type
                    tgt_edge_length = self._voxel_size_list[m]
                    mesh_area = calc_total_surface_area(ori_mesh_1)
                    num_cluster_guess = mesh_area / (np.sqrt(3) / 4 * tgt_edge_length ** 2)

                    mesh_1 = None
                    scale_1 = None
                    avg_edge_length_1 = None
                    while True:
                        try:
                            mesh_1, avg_edge_length_1, scale_1 = \
                                extract_mesh_features(o3d_mesh=ori_mesh_1,
                                                      tgt_edge_length=tgt_edge_length,
                                                      num_cluster=num_cluster_guess,
                                                      subdivide_factor=2.0,
                                                      default_edge_length=self._default_voxel_size)
                            break
                        except ValueError as err:
                            print(err)
                            continue
                    xyz1 = np.asarray(mesh_1.vertices)
                    xyz1 = self.apply_transform(xyz1, T1)
                    xyz1, _ = self.crop_points(xyz1, feats=None,
                                               axis_aligned_ratio_range=self._crop_ratio_range,
                                               min_remained_portion=self._crop_ratio_range[0])
                    remeshed_1_list.append(xyz1)
                    length_1_list.append(avg_edge_length_1)
                    scale_1_list.append(scale_1)

            voxel_size_num = len(self._voxel_size_list)
            # For a single pair of meshes, first extract xyz and features for all
            feats_0_list = list()
            feats_1_list = list()

            for m in range(voxel_size_num):
                if self._remesh_surface:
                    voxel_size = length_0_list[m]
                else:
                    voxel_size = self._voxel_size_list[m]

                while True:
                    try:
                        xyz_down, feature = extract_features(
                            model,
                            xyz=np.asarray(remeshed_0_list[m]),
                            voxel_size=voxel_size,
                            device=self._device,
                            skip_check=True)
                        break
                    except ValueError as err:
                        print(f"[extract_features] {err}")
                if self._remesh_surface:
                    feats_0_list.append((xyz_down, feature, scale_0_list[m]))
                else:
                    feats_0_list.append((xyz_down, feature))

            for m in range(voxel_size_num):
                if self._remesh_surface:
                    voxel_size = length_1_list[m]
                else:
                    voxel_size = self._voxel_size_list[m]

                while True:
                    try:
                        xyz_down, feature = extract_features(
                            model,
                            xyz=np.asarray(remeshed_1_list[m]),
                            voxel_size=voxel_size,
                            device=self._device,
                            skip_check=True)
                        break
                    except ValueError as err:
                        print(f"[extract_features] {err}")
                if self._remesh_surface:
                    feats_1_list.append((xyz_down, feature, scale_1_list[m]))
                else:
                    feats_1_list.append((xyz_down, feature))

            queue = Queue()
            processes = list()
            for m in range(voxel_size_num):
                for n in range(voxel_size_num):
                    if self._vary_scale:
                        if m == n:
                            continue
                    coord_0, feat_0 = feats_0_list[m][:2]
                    coord_1, feat_1 = feats_1_list[n][:2]

                    if self._remesh_surface:
                        scale_0 = feats_0_list[m][2]
                        scale_1 = feats_1_list[n][2]
                    else:
                        scale_0 = scale_1 = 1.0

                    transform0 = copy.deepcopy(T0)
                    transform1 = copy.deepcopy(T1)
                    transform0[:3, :3] = transform0[:3, :3] * scale_0
                    transform1[:3, :3] = transform1[:3, :3] * scale_1
                    gt_pose = transform0 @ np.linalg.inv(transform1)

                    inlier_dist_thresh = max(self._default_inlier_dist_thresh *
                                             self._voxel_size_list[m] / self._default_voxel_size,
                                             self._default_inlier_dist_thresh *
                                             self._voxel_size_list[n] / self._default_voxel_size)
                    work = Process(target=self.do_single_pair_evaluation,
                                   args=(queue, coord_0, coord_1, feat_0, feat_1,
                                         gt_pose, inlier_dist_thresh))
                    processes.append(work)
                    work.start()

            for _ in processes:
                hit_ratio = queue.get()
                if np.isnan(hit_ratio):
                    hit_ratio = 0.0
                hit_ratio_list.append(hit_ratio)
                hit_ratio_meter.update(hit_ratio)
                tq.set_postfix(hit_ratio='avg: {:.3f}, cur: {:.3f}'.format(hit_ratio_meter.avg, hit_ratio))
            for work in processes:
                work.join()

        results = np.asarray(hit_ratio_list).reshape((-1, 1))
        np.save(str(self._output_root / f"fmr_evaluation_result.npy"), results)

        result_string = f"{self._voxel_size_list[0]} {self._voxel_size_list[1]} FMR: "
        for inlier_ratio_thresh in self._inlier_ratio_thresholds:
            result_string = result_string + f"{inlier_ratio_thresh}: {np.mean(results > inlier_ratio_thresh)}, "

        with open(str(self._output_root.parent / "summary_fmr.txt"), "a") as fp:
            fp.write(result_string + "\n")
        logging.info(result_string)

        tq.close()

        return

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
        deformed_mesh_model.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_model.vertices) + np.sum(
            mode_weights.reshape((-1, 1, 1))[self._atlas_mode_range[0]:self._atlas_mode_range[1]] *
            self._atlas_mode_stds.reshape((-1, 1, 1))[self._atlas_mode_range[0]:self._atlas_mode_range[1]] *
            mode_vertices[self._atlas_mode_range[0]:self._atlas_mode_range[1]],
            axis=0).reshape((-1, 3)))
        deformed_mesh_model = deformed_mesh_model.compute_vertex_normals(normalized=True)
        return deformed_mesh_model

    def crop_points(self, points, axis_aligned_ratio_range, min_remained_portion, feats=None):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if feats is not None:
            point_cloud.normals = o3d.utility.Vector3dVector(feats)

        bounding_box = point_cloud.get_axis_aligned_bounding_box()
        num_vertices = np.asarray(point_cloud.points).shape[0]
        while True:
            min_corner = bounding_box.get_min_bound()
            box_lengths = bounding_box.get_extent()
            point_cloud_to_crop = copy.deepcopy(point_cloud)
            # Randomly pick one axis and choose two points to do the cropping
            remained_length = self.randg.uniform(low=axis_aligned_ratio_range[0], high=axis_aligned_ratio_range[1])
            offset_pos = self.randg.uniform(low=0.0, high=1.0 - remained_length)
            axis_to_crop = self.randg.randint(low=0, high=3)

            shifted_min_corner = min_corner
            shifted_min_corner[axis_to_crop] = min_corner[axis_to_crop] + offset_pos * box_lengths[axis_to_crop]

            shifted_max_corner = min_corner + box_lengths
            shifted_max_corner[axis_to_crop] = shifted_max_corner[axis_to_crop] - (1.0 - remained_length - offset_pos) * \
                                               box_lengths[axis_to_crop]
            crop_bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=shifted_min_corner.reshape((3, 1)),
                                                                    max_bound=shifted_max_corner.reshape((3, 1)))
            cropped_point_cloud = point_cloud_to_crop.crop(bounding_box=crop_bounding_box)

            if np.asarray(cropped_point_cloud.points).shape[0] >= min_remained_portion * num_vertices:
                break
            else:
                continue
        if feats is not None:
            return np.asarray(cropped_point_cloud.points), np.asarray(cropped_point_cloud.normals)
        else:
            return np.asarray(cropped_point_cloud.points), None


def main():
    parser = argparse.ArgumentParser(
        description='Geometric Feature Evaluation on Nasal Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', type=str, required=True, help='config file')
    args = parser.parse_args()
    if args.config_path is None or not Path(args.config_path).exists():
        print(f"specified config path does not exist {args.config_path}")
        exit()

    with open(args.config_path, 'r') as f:
        args.__dict__ = json.load(f)

    output_root = Path(args.output_root)
    if not output_root.exists():
        output_root.mkdir(parents=True)

    with open(str(output_root / 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    ch = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(str(output_root / "log.txt"))
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(
        format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch, fh])

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    model = models.MinkResUNet(in_channels=1,
                               out_channels=32,
                               down_channels=(None, 32, 64, 128, 256),
                               up_channels=(None, 64, 64, 64, 128),
                               bn_momentum=0.05,
                               pre_conv_num=3,
                               after_pre_channels=1,
                               conv1_kernel_size=7,
                               norm_type=args.net_norm_type,
                               upsample_type=args.net_upsample_type,
                               epsilon=1.0e-8,
                               D=3)
    checkpoint = torch.load(args.trained_model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(args.device)

    for voxel_size_pair in args.voxel_size_pair_list:
        voxel_size_0 = voxel_size_pair[0]
        voxel_size_1 = voxel_size_pair[1]

        logging.info(
            f"Processing voxel size pair {voxel_size_0} {voxel_size_1}")
        run_output_root = output_root / f"voxel_size_pair_{voxel_size_0}_" \
                                        f"{voxel_size_1}"
        if not run_output_root.exists():
            run_output_root.mkdir(parents=True)
        else:
            npy_list = list(run_output_root.glob("*eval_result.npy"))
            if len(npy_list) >= 1 and not args.overwrite_result:
                continue

        batch_voxel_size_list = [voxel_size_0, voxel_size_1]
        evaluation = FeatureMatchEvaluation(default_inlier_dist_thresh=args.inlier_dist_thresh,
                                            default_voxel_size=args.default_voxel_size,
                                            num_rand_keypoints=args.num_rand_keypoints,
                                            voxel_size_list=batch_voxel_size_list,
                                            output_root=run_output_root,
                                            vary_scale=args.vary_scale,
                                            use_remesh=args.use_remesh,
                                            overwrite_result=args.overwrite_result,
                                            atlas_mode_weights_path=Path(args.atlas_mode_weights_path),
                                            atlas_mode_weights_std_range=args.atlas_mode_weights_std_range,
                                            atlas_mode_range=args.atlas_mode_range,
                                            mean_mesh_model_path=Path(args.mean_mesh_model_path),
                                            partial_mean_mesh_model_path=Path(
                                                args.partial_mean_mesh_model_path),
                                            num_run=args.num_run_per_size_pair,
                                            crop_ratio_range=args.crop_ratio_range,
                                            device=args.device,
                                            inlier_ratio_thresholds=args.inlier_ratio_thresholds
                                            )
        evaluation.feature_evaluation(model=model)

    # Summarize results
    result_dict = summarize_resolution_mismatch_results(root=output_root,
                                                        inlier_ratio_threshold_list=args.inlier_ratio_thresholds)
    # Save average feature match rate over scenes per voxel size pair as a dictionary
    with open(str(output_root / 'fmr_per_pair.pkl'), 'w') as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    with torch.no_grad():
        main()
