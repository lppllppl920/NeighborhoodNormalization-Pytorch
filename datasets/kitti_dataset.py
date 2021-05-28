import logging
import torch
import torch.utils.data
import numpy as np
import glob
import os
import pathlib
import copy
from scipy.spatial import cKDTree
import MinkowskiEngine as ME
import open3d as o3d
from scipy.stats import loguniform

import utils

from .dataset_utils import sample_random_rotation


class PairDataset(torch.utils.data.Dataset):
    def __init__(self,
                 phase,
                 transform=None,
                 use_rotation=True,
                 use_scale=True,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.use_scale = use_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.use_rotation = use_rotation
        self.rotation_range = config.rotation_range
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)


class KITTIPairDataset(PairDataset):
    DATA_FILES = {
        'train': [0, 1, 2, 3, 4, 5],
        'val': [6, 7],
        'test': [8, 9, 10],
    }

    def __init__(self,
                 phase,
                 transform=None,
                 use_rotation=True,
                 use_scale=True,
                 is_odometry=True,
                 manual_seed=False,
                 config=None):
        self.is_odometry = is_odometry
        # For evaluation, use the odometry dataset training following the 3DFeat eval method
        if self.is_odometry:
            self.root = root = config.kitti_root + '/dataset'
            use_rotation = use_rotation
            use_scale = use_scale
        else:
            self.date = config.kitti_date
            self.root = root = os.path.join(config.kitti_root, self.date)

        self.icp_path = os.path.join(config.kitti_root, 'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        PairDataset.__init__(self, phase, transform, use_rotation, use_scale,
                             manual_seed, config)

        logging.info(f"Loading the subset {phase} from {root}")
        # Use the kitti root
        self.max_time_diff = max_time_diff = config.kitti_max_time_diff
        if phase == "train":
            self.sampling_size = config.train_sampling_size
        elif phase == "val":
            self.sampling_size = config.val_sampling_size
        elif phase == "test":
            self.sampling_size = None  # test is evaluated on all gt correspondences
        else:
            raise NotImplementedError
        self.oversampling_factor = config.oversampling_factor
        self.max_sampling_trial = config.max_sampling_trial
        self.allow_repeat_sampling = config.allow_repeat_sampling
        self.same_scale = config.same_scale
        self.test_type = config.test_type
        self.positive_pair_search_voxel_size_multiplier = config.positive_pair_search_voxel_size_multiplier

        self.kitti_cache = {}
        self.kitti_icp_cache = {}

        subset_names = self.DATA_FILES[phase]
        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
            for start_time in inames:
                for time_diff in range(2, max_time_diff):
                    pair_time = time_diff + start_time
                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))

    def get_all_scan_ids(self, drive_id):
        if self.is_odometry:
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.is_odometry:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]
        else:
            data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in self.kitti_cache:
                    self.kitti_cache[filename] = np.genfromtxt(filename)
                odometry.append(self.kitti_cache[filename])

            odometry = np.array(odometry)
            return odometry

    def odometry_to_positions(self, odometry):
        if self.is_odometry:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def rot3d(self, axis, angle):
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(3)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

    def pos_transform(self, pos):
        x, y, z, rx, ry, rz, _ = pos[0]
        RT = np.eye(4)
        RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
        RT[:3, 3] = [x, y, z]
        return RT

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)

    def _get_velodyne_fn(self, drive, t):
        if self.is_odometry:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        else:
            fname = self.root + \
                    '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
                        drive, t)
        return fname

    def __getitem__(self, idx):
        sel0 = None
        sel1 = None
        pcd0 = None
        pcd1 = None
        while True:
            drive = self.files[idx][0]
            t0, t1 = self.files[idx][1], self.files[idx][2]
            all_odometry = self.get_video_odometry(drive, [t0, t1])
            positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
            fname0 = self._get_velodyne_fn(drive, t0)
            fname1 = self._get_velodyne_fn(drive, t1)

            # XYZ and reflectance
            xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
            xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

            xyz0 = xyzr0[:, :3]
            xyz1 = xyzr1[:, :3]

            key = '%d_%d_%d' % (drive, t0, t1)
            filename = self.icp_path + '/' + key + '.npy'
            if key not in self.kitti_icp_cache:
                if not os.path.exists(filename):
                    # work on the downsampled xyzs, 0.05m == 5cm
                    sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)[1]
                    sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)[1]

                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T
                    xyz0_t = self.apply_transform(xyz0[sel0], M)
                    pcd0 = utils.make_open3d_point_cloud(xyz0_t)
                    pcd1 = utils.make_open3d_point_cloud(xyz1[sel1])
                    reg = o3d.registration.registration_icp(
                        pcd0, pcd1, 0.2, np.eye(4),
                        o3d.registration.TransformationEstimationPointToPoint(),
                        o3d.registration.ICPConvergenceCriteria(max_iteration=200))
                    pcd0.transform(reg.transformation)
                    # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                    M2 = M @ reg.transformation
                    # o3d.draw_geometries([pcd0, pcd1])
                    # write to a file
                    np.save(filename, M2)
                else:
                    M2 = np.load(filename)
                self.kitti_icp_cache[key] = M2
            else:
                M2 = self.kitti_icp_cache[key]

            if self.use_rotation:
                T0 = sample_random_rotation(self.randg, self.rotation_range)
                T1 = sample_random_rotation(self.randg, self.rotation_range)
                trans = T1 @ M2 @ np.linalg.inv(T0)

                xyz0 = self.apply_transform(xyz0, T0)
                xyz1 = self.apply_transform(xyz1, T1)
            else:
                trans = M2

            if self.use_scale and self.same_scale:
                scale = loguniform(a=self.min_scale,
                                   b=self.max_scale).rvs(size=1)[0]
                xyz0 = scale * xyz0
                xyz1 = scale * xyz1
                scale0 = scale1 = scale
            elif self.use_scale and not self.same_scale:
                scale0 = loguniform(a=self.min_scale,
                                    b=self.max_scale).rvs(size=1)[0]
                scale1 = loguniform(a=self.min_scale,
                                    b=self.max_scale).rvs(size=1)[0]
                xyz0 = scale0 * xyz0
                xyz1 = scale1 * xyz1
            else:
                scale0 = scale1 = 1

            scale_trans_0 = np.eye(4)
            scale_trans_0[:3, :3] = scale_trans_0[:3, :3] * scale0
            scale_trans_1 = np.eye(4)
            scale_trans_1[:3, :3] = scale_trans_1[:3, :3] * scale1
            trans = scale_trans_1 @ trans @ np.linalg.inv(scale_trans_0)

            # Voxelization
            xyz0_th = torch.from_numpy(xyz0)
            xyz1_th = torch.from_numpy(xyz1)

            if self.phase == "train" or self.phase == "val" or \
                    (self.phase == "test" and self.test_type == "fixed_scale"):
                sel0 = ME.utils.sparse_quantize(xyz0_th, return_index=True, quantization_size=self.voxel_size)[1]
                sel1 = ME.utils.sparse_quantize(xyz1_th, return_index=True, quantization_size=self.voxel_size)[1]
                # Make point clouds using voxelized points
                pcd0 = utils.make_open3d_point_cloud(xyz0[sel0])
                pcd1 = utils.make_open3d_point_cloud(xyz1[sel1])
            elif self.phase == "test" and self.test_type == "scale_vary":
                # If the test type is scale_vary, we delay the quantization after data loading
                pcd0 = utils.make_open3d_point_cloud(xyz0)
                pcd1 = utils.make_open3d_point_cloud(xyz1)

            # Get matches
            if self.phase == "train" or self.phase == "val":
                count, matches = self.get_matching_indices(pcd0, pcd1, trans,
                                                           self.voxel_size *
                                                           self.positive_pair_search_voxel_size_multiplier)
                if count >= self.max_sampling_trial:
                    idx = self.randg.randint(low=0, high=len(self.files))
                    continue
                else:
                    break
            elif self.phase == "test":
                matches = torch.ones(2, 2).int()
                break

        if self.phase == "train" or self.phase == "val" or \
                (self.phase == "test" and self.test_type == "fixed_scale"):
            # Get features
            npts0 = len(sel0)
            npts1 = len(sel1)

            feats_train0, feats_train1 = [], []

            unique_xyz0_th = xyz0_th[sel0]
            unique_xyz1_th = xyz1_th[sel1]

            feats_train0.append(torch.ones((npts0, 1)))
            feats_train1.append(torch.ones((npts1, 1)))

            feats0 = torch.cat(feats_train0, 1)
            feats1 = torch.cat(feats_train1, 1)

            coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
            coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

            if self.transform:
                coords0, feats0 = self.transform(coords0, feats0)
                coords1, feats1 = self.transform(coords1, feats1)
            return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(),
                    coords1.int(), feats0.float(), feats1.float(), matches, trans, scale0 / scale1)

        elif self.phase == "test" and self.test_type == "scale_vary":
            feats0 = torch.ones(xyz0_th.shape[0], 1).float()
            feats1 = torch.ones(xyz1_th.shape[0], 1).float()
            # scale varying is delayed to data loading function when scale_vary eval type is used
            return (xyz0_th.float(), xyz1_th.float(), xyz0_th.int(),
                    xyz1_th.int(), feats0.float(), feats1.float(), matches, trans, 1)
        else:
            raise AttributeError(f"Phase {self.phase} not supported")

    def get_matching_indices(self, source, target, trans, search_voxel_size):
        valid_indexes = None
        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)
        source_copy.transform(trans)

        target_points = np.asarray(target_copy.points)
        source_points = np.asarray(source_copy.points)

        count = 0
        if not self.allow_repeat_sampling:
            overlap_indexes = set()
        else:
            overlap_indexes = list()
        ball_tree = cKDTree(target_points)
        while count < self.max_sampling_trial:
            sampled_indexes = np.sort(
                self.randg.choice(np.arange(start=0, stop=source_points.shape[0]),
                                  size=int(self.sampling_size * self.oversampling_factor),
                                  replace=True))

            counts = ball_tree.query_ball_point(x=source_points[sampled_indexes], r=search_voxel_size,
                                                return_length=True)
            if not self.allow_repeat_sampling:
                overlap_indexes.update(sampled_indexes[np.argwhere(counts > 0).flatten()])
            else:
                overlap_indexes.extend(sampled_indexes[np.argwhere(counts > 0).flatten()])

            if len(overlap_indexes) <= self.sampling_size:
                count += 1
                continue
            else:
                overlap_indexes = self.randg.choice(list(overlap_indexes), size=self.sampling_size)
                distances, valid_indexes = ball_tree.query(x=source_points[overlap_indexes], k=1)
                break

        if count >= self.max_sampling_trial:
            return count, []

        valid_indexes_0 = overlap_indexes
        valid_indexes_1 = valid_indexes.flatten()
        matches = torch.from_numpy(
            np.concatenate(
                [np.asarray(valid_indexes_0).reshape((-1, 1)), np.asarray(valid_indexes_1).reshape((-1, 1))],
                axis=1))

        return count, matches


class KITTINMPairDataset(KITTIPairDataset):
    r"""
    Generate KITTI pairs within N meter distance
    """

    def __init__(self,
                 phase,
                 transform=None,
                 use_rotation=True,
                 use_scale=True,
                 is_odometry=True,
                 manual_seed=False,
                 config=None):

        # self.is_odometry = is_odometry
        # if self.is_odometry:
        #     self.root = root = os.path.join(config.kitti_root, 'dataset')
        #     use_rotation = use_rotation
        #     use_scale = use_scale
        # else:
        #     self.date = config.kitti_date
        #     self.root = root = os.path.join(config.kitti_root, self.date)
        #
        # self.icp_path = os.path.join(config.kitti_root, 'icp')
        # pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        KITTIPairDataset.__init__(self, phase, transform, use_rotation, use_scale, is_odometry,
                                  manual_seed, config)
        logging.info(f"Loading the subset {phase} from {self.root}")
        subset_names = self.DATA_FILES[phase]
        self.min_dist = config.min_dist
        self.files.clear()
        if self.is_odometry:
            for dirname in subset_names:
                drive_id = int(dirname)
                fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
                assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, :3, 3]
                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))
                valid_pairs = pdist > self.min_dist
                curr_time = inames[0]
                while curr_time in inames:
                    # Find the min index
                    next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
                    if len(next_time) == 0:
                        curr_time += 1
                    else:
                        # Follow https://github.com/yewzijian/3DFeatNet/blob/master/
                        # scripts_data_processing/kitti/process_kitti_data.m#L44
                        next_time = next_time[0] + curr_time - 1

                    if next_time in inames:
                        self.files.append((drive_id, curr_time, next_time))
                        curr_time = next_time + 1
        else:
            for dirname in subset_names:
                drive_id = int(dirname)
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
                assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
                Ts = all_pos[:, 0, :3]

                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))

                for start_time in inames:
                    pair_time = np.where(
                        pdist[start_time][start_time:start_time + 100] > self.min_dist)[0]
                    if len(pair_time) == 0:
                        continue
                    else:
                        pair_time = pair_time[0] + start_time

                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))

        if self.is_odometry:
            # Remove problematic sequence
            for item in [
                (8, 15, 58),
            ]:
                if item in self.files:
                    self.files.pop(self.files.index(item))
