import sys
import numpy as np
import argparse
import logging
import open3d as o3d

from multiprocessing import Process, Queue
from .eval_utils import extract_features, remesh_surface, calc_total_surface_area, read_trajectory
from pathlib import Path
import torch
import copy
from scipy.spatial import cKDTree
import pickle
import json
import tqdm
import random

import utils
import models
from eval_utils import summarize_resolution_mismatch_results


class FeatureMatchEvaluation(object):
    def __init__(self, output_root,
                 default_inlier_dist_thresh,
                 default_voxel_size, num_rand_keypoints, voxel_size_list, use_gt_pose,
                 vary_scale, use_remesh, overwrite_result, device):
        self._output_root = output_root
        self._default_voxel_size = default_voxel_size
        self._default_inlier_dist_thresh = default_inlier_dist_thresh
        self._voxel_size_list = voxel_size_list
        self._num_rand_keypoints = num_rand_keypoints
        self._device = device
        self._use_gt_pose = use_gt_pose
        self._vary_scale = vary_scale
        self._overwrite_result = overwrite_result
        self._remesh_surface = use_remesh
        if not self._vary_scale:
            assert (len(voxel_size_list) == 1)

    def find_nn_cpu(self, feat0, feat1):
        feat1tree = cKDTree(feat1)
        dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
        return nn_inds

    def valid_feat_ratio(self, coord0, coord1, feat0, feat1, trans_gth, thresh):
        coord0_copy = copy.deepcopy(coord0)
        coord0_copy = coord0_copy.transform(trans_gth)

        nn_inds = self.find_nn_cpu(feat0, feat1)
        dist = np.sqrt(((np.array(coord0_copy.points) - np.array(coord1.points)[nn_inds]) ** 2).sum(1))
        return np.mean(dist < thresh)

    @staticmethod
    def make_open3d_point_cloud(xyz, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd

    def do_single_pair_evaluation(self, queue, coord_i, coord_j, feat_i, feat_j,
                                  traj, inlier_dist_thresh):
        trans_gth = np.linalg.inv(traj.pose)
        Ni, Nj = len(coord_i), len(coord_j)

        # Only sample the one that has fewer points in it originally
        if self._num_rand_keypoints > 0:
            if Ni < Nj:
                num_sample_i = min(Ni, self._num_rand_keypoints)
                inds_i = np.random.choice(Ni, num_sample_i, replace=False)
                coord_i, feat_i = coord_i[inds_i], feat_i[inds_i]
            else:
                num_sample_j = min(Nj, self._num_rand_keypoints)
                inds_j = np.random.choice(Nj, num_sample_j, replace=False)
                coord_j, feat_j = coord_j[inds_j], feat_j[inds_j]

        coord_i = self.make_open3d_point_cloud(coord_i)
        coord_j = self.make_open3d_point_cloud(coord_j)

        try:
            if Ni < Nj:
                hit_ratio = self.valid_feat_ratio(coord0=coord_i, coord1=coord_j, feat0=feat_i, feat1=feat_j,
                                                  trans_gth=trans_gth, thresh=inlier_dist_thresh)

            else:
                hit_ratio = self.valid_feat_ratio(coord0=coord_j, coord1=coord_i, feat0=feat_j, feat1=feat_i,
                                                  trans_gth=np.linalg.inv(trans_gth), thresh=inlier_dist_thresh)

            queue.put(hit_ratio)
            return
        except IndexError or ZeroDivisionError as err:
            print(f"number of features in sample i and j: {Ni} {Nj}, err: {err}")
            queue.put(0.0)
            return

    def feature_evaluation(self, model, set_list):
        result_dict = dict()
        hit_ratio_meter = utils.AverageMeter()
        avg_edge_length = None

        for set_idx, set_path in enumerate(set_list):
            logging.info(f"Processing {set_path}...")
            # Fixed numpy random seed for reproducibility
            np.random.seed(set_idx)
            set_name = set_path.name

            if not self._overwrite_result:
                if (self._output_root / f"eval_result_{set_name}.npy").exists():
                    continue

            traj = read_trajectory(str(set_path.parent / (set_name + "-evaluation") / "gt.log"))

            results = []
            tq = tqdm.tqdm(total=len(traj))
            tq.set_description(f"{set_path.name}")
            xyz_feat_dict = dict()

            for i in range(len(traj)):

                tq.update(1)
                traj_item = traj[i]
                idx_0 = traj_item.metadata[0]
                idx_1 = traj_item.metadata[1]

                input_path_0 = set_path / "seq-01" / "cloud_bin_{:d}.ply".format(idx_0)
                input_path_1 = set_path / "seq-01" / "cloud_bin_{:d}.ply".format(idx_1)

                if not input_path_0.exists():
                    input_path_0 = set_path / "cloud_bin_{:d}.ply".format(idx_0)
                    input_path_1 = set_path / "cloud_bin_{:d}.ply".format(idx_1)

                mesh_0 = o3d.io.read_triangle_mesh(str(input_path_0))
                mesh_1 = o3d.io.read_triangle_mesh(str(input_path_1))

                if not self._use_gt_pose:
                    traj_item.pose = np.eye(4)

                voxel_size_num = len(self._voxel_size_list)
                # For a single pair of meshes, first extract xyz and features for all
                for m in range(voxel_size_num):
                    if idx_0 not in xyz_feat_dict or m not in xyz_feat_dict[idx_0]:
                        if self._remesh_surface:
                            # pre-process and extract features from mesh based on voxel size and required feature type
                            tgt_edge_length = self._voxel_size_list[m]
                            mesh_area = calc_total_surface_area(mesh_0)
                            num_cluster_guess = mesh_area / (np.sqrt(3) / 4 * tgt_edge_length ** 2)
                            mesh_0, avg_edge_length = \
                                remesh_surface(o3d_mesh=mesh_0,
                                               tgt_edge_length=tgt_edge_length,
                                               num_cluster=num_cluster_guess,
                                               subdivide_factor=2.0)

                            if np.asarray(mesh_0.vertices).shape[0] == 0:
                                print(f"mesh {idx_0} at tgt length {tgt_edge_length} has no vertices left")
                                break

                        if self._remesh_surface:
                            voxel_size = avg_edge_length
                        else:
                            voxel_size = self._voxel_size_list[m]

                        while True:
                            try:
                                xyz_down, feature = extract_features(
                                    model,
                                    xyz=np.asarray(mesh_0.vertices),
                                    voxel_size=voxel_size,
                                    device=self._device,
                                    skip_check=True)
                                break
                            except ValueError as err:
                                print(f"[extract_features] {err}")

                        if idx_0 not in xyz_feat_dict:
                            xyz_feat_dict[idx_0] = dict()
                        if self._remesh_surface:
                            xyz_feat_dict[idx_0][m] = (xyz_down, feature, avg_edge_length)
                        else:
                            xyz_feat_dict[idx_0][m] = (xyz_down, feature)

                for n in range(voxel_size_num):
                    if idx_1 not in xyz_feat_dict or n not in xyz_feat_dict[idx_1]:
                        if self._remesh_surface:
                            # pre-process and extract features from mesh based on voxel size and required feature type
                            tgt_edge_length = self._voxel_size_list[n]
                            mesh_area = calc_total_surface_area(mesh_1)
                            num_cluster_guess = mesh_area / (np.sqrt(3) / 4 * tgt_edge_length ** 2)
                            mesh_1, avg_edge_length = \
                                remesh_surface(o3d_mesh=mesh_1,
                                               tgt_edge_length=tgt_edge_length,
                                               num_cluster=num_cluster_guess,
                                               subdivide_factor=2.0)
                            if np.asarray(mesh_1.vertices).shape[0] == 0:
                                print(f"mesh {idx_1} at tgt length {tgt_edge_length} has no vertices left")
                                break

                        if self._remesh_surface:
                            voxel_size = avg_edge_length
                        else:
                            voxel_size = self._voxel_size_list[n]

                        while True:
                            try:
                                xyz_down, feature = extract_features(
                                    model,
                                    xyz=np.asarray(mesh_1.vertices),
                                    voxel_size=voxel_size,
                                    device=self._device,
                                    skip_check=True)
                                break
                            except ValueError as err:
                                print(f"[extract_features] {err}")

                        if idx_1 not in xyz_feat_dict:
                            xyz_feat_dict[idx_1] = dict()

                        if self._remesh_surface:
                            xyz_feat_dict[idx_1][n] = (xyz_down, feature, avg_edge_length)
                        else:
                            xyz_feat_dict[idx_1][n] = (xyz_down, feature)

                # Not evaluate pairs where the mesh is empty
                if np.asarray(mesh_0.vertices).shape[0] == 0 or np.asarray(mesh_1.vertices).shape[0] == 0:
                    continue

                queue = Queue()
                processes = list()
                for m in range(voxel_size_num):
                    for n in range(voxel_size_num):
                        if self._vary_scale and m == n:
                            continue
                        coord_0, feat_0 = xyz_feat_dict[idx_0][m][:2]
                        coord_1, feat_1 = xyz_feat_dict[idx_1][n][:2]

                        # Not evaluating sample pairs with too few vertices
                        if feat_0.shape[0] < 10 or feat_1.shape[0] < 10:
                            continue

                        inlier_dist_thresh = max(self._default_inlier_dist_thresh *
                                                 self._voxel_size_list[m] / self._default_voxel_size,
                                                 self._default_inlier_dist_thresh *
                                                 self._voxel_size_list[n] / self._default_voxel_size)
                        work = Process(target=self.do_single_pair_evaluation,
                                       args=(queue, coord_0, coord_1, feat_0, feat_1,
                                             traj_item, inlier_dist_thresh))
                        processes.append(work)
                        work.start()

                for _ in processes:
                    hit_ratio = queue.get()
                    if np.isnan(hit_ratio):
                        hit_ratio = 0.0
                    results.append([idx_0, idx_1, hit_ratio])
                    hit_ratio_meter.update(hit_ratio)
                    tq.set_postfix(hit_ratio='avg: {:.3f}, cur: {:.3f}'.format(hit_ratio_meter.avg, hit_ratio))

                for work in processes:
                    work.join()

            tq.close()
            result_dict[set_name] = np.asarray(results)

            np.save(str(self._output_root / f"eval_result_{set_name}.npy"), result_dict[set_name])

        with open(str(self._output_root / "eval_result.pkl"), "wb") as fp:
            pickle.dump(result_dict, fp)


def main():
    parser = argparse.ArgumentParser(
        description='Geometric Feature Evaluation on 3DMatch Dataset',
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

    set_list = sorted(list(Path(args.source_root).glob("*/")))
    evaluate_set_list = sorted(list(Path(args.source_root).glob("*-evaluation/")))
    set_list = sorted(list(set(set_list) - set(evaluate_set_list)))

    if args.vary_scale:
        for i in range(len(args.voxel_size_list) // 2):
            logging.info(
                f"Processing voxel size pair {args.voxel_size_list[2 * i]} {args.voxel_size_list[2 * i + 1]}")
            run_output_root = output_root / f"voxel_size_pair_{args.voxel_size_list[2 * i]}_" \
                                            f"{args.voxel_size_list[2 * i + 1]}"
            if not run_output_root.exists():
                run_output_root.mkdir(parents=True)

            batch_voxel_size_list = [args.voxel_size_list[2 * i], args.voxel_size_list[2 * i + 1]]
            evaluation = FeatureMatchEvaluation(default_inlier_dist_thresh=args.inlier_dist_thresh,
                                                default_voxel_size=args.default_voxel_size,
                                                num_rand_keypoints=args.num_rand_keypoints,
                                                voxel_size_list=batch_voxel_size_list,
                                                # the built mesh dataset already get meshes aligned
                                                use_gt_pose=False,
                                                vary_scale=True,
                                                use_remesh=True,
                                                output_root=run_output_root,
                                                overwrite_result=args.overwrite_result,
                                                device=args.device)
            evaluation.feature_evaluation(model=model, set_list=set_list)

        # Summarize results
        result_dict = summarize_resolution_mismatch_results(root=output_root,
                                                            inlier_ratio_threshold_list=args.inlier_ratio_thresholds)
        # Save average feature match rate over scenes per voxel size pair as a dictionary
        with open(str(output_root / 'fmr_per_pair.pkl'), 'w') as fp:
            pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        for i in range(len(args.voxel_size_list)):
            logging.info(
                f"Processing voxel size {args.voxel_size_list[i]}")
            run_output_root = output_root / f"voxel_size_{args.voxel_size_list[i]}_{args.voxel_size_list[i]}"
            if not run_output_root.exists():
                run_output_root.mkdir(parents=True)

            batch_voxel_size_list = [args.voxel_size_list[i]]
            evaluation = FeatureMatchEvaluation(default_inlier_dist_thresh=args.inlier_dist_thresh,
                                                default_voxel_size=args.default_voxel_size,
                                                num_rand_keypoints=args.num_rand_keypoints,
                                                voxel_size_list=batch_voxel_size_list,
                                                # the point cloud dataset does not have meshes aligned
                                                use_gt_pose=True,
                                                vary_scale=False,
                                                use_remesh=False,
                                                output_root=run_output_root,
                                                overwrite_result=args.overwrite_result,
                                                device=args.device)
            evaluation.feature_evaluation(model=model, set_list=set_list)

        # Summarize results
        result_dict = summarize_resolution_mismatch_results(root=output_root,
                                                            inlier_ratio_threshold_list=args.inlier_ratio_thresholds)
        # Save average feature match rate over scenes per voxel size pair as a dictionary
        with open(str(output_root / 'fmr_per_pair.pkl'), 'w') as fp:
            pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    with torch.no_grad():
        main()
