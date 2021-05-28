import MinkowskiEngine as ME
from pathlib import Path
import torch
import tqdm
import argparse
import numpy as np
import random
import json
import logging
import open3d as o3d
import sys
import warnings
from scipy.spatial import cKDTree
import copy
import pickle

warnings.filterwarnings("ignore")
import datasets
import models
import utils
from utils import AverageMeter, Timer
from eval_utils import summarize_resolution_mismatch_results


def main():
    parser = argparse.ArgumentParser(
        description='Geometric Feature Evaluation on KITTI Dataset -- Resolution Mismatch',
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

    args.allow_repeat_sampling = True
    test_dataset = datasets.KITTINMPairDataset(phase="test", use_rotation=args.use_rotation,
                                               use_scale=args.use_scale,
                                               manual_seed=False, config=args)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=datasets.separate_collate_pair_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)))

    model = models.MinkResUNet(in_channels=1,
                               out_channels=32,
                               down_channels=(None, 32, 64, 128, 256),
                               up_channels=(None, 64, 64, 64, 128),
                               bn_momentum=0.05,
                               pre_conv_num=3,
                               after_pre_channels=1,
                               conv1_kernel_size=5,
                               norm_type=args.net_norm_type,
                               upsample_type=args.net_upsample_type,
                               epsilon=1.0e-8,
                               D=3)
    checkpoint = torch.load(args.trained_model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(args.device)

    fp = open(str(output_root / "summary_fmr.txt"), "a")

    for voxel_size_pair in args.voxel_size_pair_list:
        hit_ratio_meter = AverageMeter()
        data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()
        feat_match_timer = Timer()
        test_iter = test_loader.__iter__()
        n_gpu_failures = 0
        hit_ratio_list = list()

        # In the default parameter settings, voxel_size_1 will be the base voxel size.
        # voxel_size_0 is the varying voxel size
        voxel_size_1 = voxel_size_pair[0]
        voxel_size_0 = voxel_size_pair[1]

        print(f"Processing voxel size pair {voxel_size_1} {voxel_size_0}")
        run_output_root = output_root / "voxel_size_pair_{:.3f}_{:.3f}".format(voxel_size_1, voxel_size_0)
        if not run_output_root.exists():
            run_output_root.mkdir(parents=True)

        if not args.overwrite_result:
            if (run_output_root / "fmr_evaluation_result.npy").exists():
                continue
        tq = tqdm.tqdm(total=len(test_iter))
        for i in range(len(test_iter)):
            data_timer.tic()
            try:
                data_dict = test_iter.next()
            except ValueError:
                n_gpu_failures += 1
                tq.update(1)
                continue
            data_timer.toc()

            xyz0, xyz1 = data_dict['pcd0'], data_dict['pcd1']
            T_gth = data_dict['T_gt']
            T_gth = T_gth.squeeze()

            # Conducting point cloud downsampling here
            sel0 = ME.utils.sparse_quantize(xyz0, return_index=True, quantization_size=voxel_size_0)[1]
            sel1 = ME.utils.sparse_quantize(xyz1, return_index=True, quantization_size=voxel_size_1)[1]
            # Make point clouds using voxelized points
            pcd1 = utils.make_open3d_point_cloud(xyz1[sel1])
            feats0 = torch.ones(len(sel0), 1).float()
            feats1 = torch.ones(len(sel1), 1).float()

            coords0 = torch.floor(xyz0[sel0] / voxel_size_0).int()
            coords0 = torch.cat([torch.zeros(len(sel0), 1).int(), coords0], dim=1)
            coords1 = torch.floor(xyz1[sel1] / voxel_size_1).int()
            coords1 = torch.cat([torch.zeros(len(sel1), 1).int(), coords1], dim=1)

            feat_timer.tic()
            sinput0 = ME.SparseTensor(
                feats0, coordinates=coords0, device=args.device)
            F0 = model(sinput0).F.detach()
            sinput1 = ME.SparseTensor(
                feats1, coordinates=coords1, device=args.device)
            F1 = model(sinput1).F.detach()
            feat_timer.toc()

            F0 = F0.cpu().numpy()
            F1 = F1.cpu().numpy()
            xyz0np = xyz0[sel0].cpu().numpy()
            N0 = len(xyz0np)

            feat_match_timer.tic()
            # Only subsample points that serve as the query instead of the target
            if args.num_rand_keypoints > 0:
                num_sample_0 = min(N0, args.num_rand_keypoints)
                inds_0 = np.random.choice(N0, num_sample_0, replace=False)
                xyz0np, F0 = xyz0np[inds_0], F0[inds_0]
                pcd0 = utils.make_open3d_point_cloud(xyz0np)
            else:
                pcd0 = utils.make_open3d_point_cloud(xyz0np)

            hit_ratio = valid_feat_ratio(coord0=pcd0, coord1=pcd1,
                                         feat0=F0, feat1=F1,
                                         trans_gth=T_gth, thresh=args.inlier_dist_threshold_factor *
                                                                 max(voxel_size_1, voxel_size_0))
            feat_match_timer.toc()
            hit_ratio_list.append(hit_ratio)
            hit_ratio_meter.update(hit_ratio)
            tq.update(1)
            tq.set_postfix(hit_ratio='avg: {:.4f}, cur: {:.4f}'.format(hit_ratio_meter.avg, hit_ratio),
                           n_below_1k='{:d}'.format(n_gpu_failures),
                           feat_time='{:.2f}'.format(feat_timer.avg),
                           data_time='{:.2f}'.format(data_timer.avg),
                           fm_time='{:.2f}'.format(feat_match_timer.avg),
                           )
            feat_timer.reset()
            feat_match_timer.reset()
            data_timer.reset()

        results = np.asarray(hit_ratio_list).reshape((-1, 1))
        np.save(str(run_output_root / "fmr_evaluation_result.npy"), results)

        result_string = f"{voxel_size_1} {voxel_size_0} FMR: "
        for inlier_ratio_thresh in args.inlier_ratio_thresholds:
            result_string = result_string + f"{inlier_ratio_thresh}: {np.mean(results > inlier_ratio_thresh)}, "
        logging.info(result_string)
        fp.write(result_string + "\n")

        tq.close()

    # Summarize results
    result_dict = summarize_resolution_mismatch_results(root=output_root,
                                                        inlier_ratio_threshold_list=args.inlier_ratio_thresholds)
    # Save average feature match rate over scenes per voxel size pair as a dictionary
    with open(str(output_root / 'fmr_per_pair.pkl'), 'w') as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def find_nn_cpu(feat0, feat1):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
    return nn_inds


def valid_feat_ratio(coord0, coord1, feat0, feat1, trans_gth, thresh):
    coord0_copy = copy.deepcopy(coord0)
    coord0_copy = coord0_copy.transform(trans_gth)
    nn_inds = find_nn_cpu(feat0, feat1)
    dist = np.sqrt(((np.array(coord0_copy.points) - np.array(coord1.points)[nn_inds]) ** 2).sum(1))
    return np.mean(dist < thresh)


if __name__ == "__main__":
    with torch.no_grad():
        main()
