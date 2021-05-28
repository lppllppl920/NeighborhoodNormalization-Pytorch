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

warnings.filterwarnings("ignore")
import datasets
import models
import utils
from utils import AverageMeter, Timer


def main():
    parser = argparse.ArgumentParser(
        description='Geometric Feature Evaluation on KITTI Dataset -- Standard',
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
    test_dataset = datasets.KITTINMPairDataset(phase="test",
                                               use_rotation=args.use_rotation,
                                               use_scale=args.use_scale,
                                               manual_seed=False, config=args)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=datasets.separate_collate_pair_fn,
        pin_memory=True,
        drop_last=False)

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

    success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
    hit_ratio_meter = AverageMeter()
    data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()
    feat_match_timer = Timer()

    test_iter = test_loader.__iter__()
    n_gpu_failures = 0

    rte_list = list()
    rre_list = list()
    hit_ratio_list = list()
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
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

        feat_timer.tic()
        sinput0 = ME.SparseTensor(
            data_dict['sinput0_F'], coordinates=data_dict['sinput0_C'], device=args.device)
        F0 = model(sinput0).F.detach()
        sinput1 = ME.SparseTensor(
            data_dict['sinput1_F'], coordinates=data_dict['sinput1_C'], device=args.device)
        F1 = model(sinput1).F.detach()
        feat_timer.toc()

        if args.eval_type == "reg":
            pcd0 = utils.make_open3d_point_cloud(xyz1np)
            pcd1 = utils.make_open3d_point_cloud(xyz1np)

            feat0 = utils.make_open3d_feature(F0, 32, F0.shape[0])
            feat1 = utils.make_open3d_feature(F1, 32, F1.shape[0])

            reg_timer.tic()
            distance_threshold = args.voxel_size
            ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
                pcd0, pcd1, feat0, feat1, distance_threshold,
                o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                    o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ], o3d.registration.RANSACConvergenceCriteria(4000000, 1000))
            T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
            reg_timer.toc()

            # Translation error
            rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
            rre = np.arccos((np.trace(T_ransac[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

            rte_list.append(rte)
            rre_list.append(rre)
            # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
            # http://openaccess.thecvf.com/content_ECCV_2018/papers/
            # Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf
            if rte < 2:
                rte_meter.update(rte)

            if not np.isnan(rre) and rre < np.pi / 180 * 5:
                rre_meter.update(rre)

            if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
                success_meter.update(1)
            else:
                success_meter.update(0)
                logging.info(f"Failed with RTE: {rte}, RRE: {rre}")

            tq.update(1)
            tq.set_postfix(rte='avg: {:.4f}, cur: {:.4f}'.format(rte_meter.avg, rte),
                           rre='avg: {:.4f}, cur: {:.4f}'.format(rre_meter.avg, rre),
                           success='{:.4f}'.format(success_meter.avg),
                           n_below_1k='{:d}'.format(n_gpu_failures),
                           feat_time='{:.2f}'.format(feat_timer.avg),
                           reg_time='{:.2f}'.format(reg_timer.avg),
                           data_time='{:.2f}'.format(data_timer.avg))
            data_timer.reset()
            feat_timer.reset()
            reg_timer.reset()
        elif args.eval_type == "fmr":
            pcd1 = utils.make_open3d_point_cloud(xyz1np)
            F0 = F0.cpu().numpy()
            F1 = F1.cpu().numpy()
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
                                         trans_gth=T_gth, thresh=args.inlier_dist_threshold)
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
            data_timer.reset()
            feat_timer.reset()
            feat_match_timer.reset()
        else:
            raise AttributeError(f"Evaluation type {args.eval_type} not supported")

    if args.eval_type == "reg":
        rte_list = np.asarray(rte_list).reshape((-1, 1))
        rre_list = np.asarray(rre_list).reshape((-1, 1))
        results = np.concatenate([rte_list, rre_list], axis=1)
        np.save(str(output_root / "reg_evaluation_result.npy"), results)
        result_string = f"RTE: {rte_meter.avg}, std: {np.sqrt(rte_meter.var)}," + \
                        f" RRE: {rre_meter.avg}, std: {np.sqrt(rre_meter.var)}, Success: {success_meter.sum} " + \
                        f"/ {success_meter.count} ({success_meter.avg * 100} %)"
        with open(str(output_root / "summary_reg.txt"), "w") as fp:
            fp.write(result_string + "\n")
        logging.info(result_string)
    elif args.eval_type == "fmr":
        hit_ratio_list = np.asarray(hit_ratio_list).reshape((-1, 1))
        np.save(str(output_root / "fmr_evaluation_result.npy"), hit_ratio_list)
        result_string = f"FMR: "
        for inlier_ratio_thresh in args.inlier_ratio_thresholds:
            result_string = result_string + f"{inlier_ratio_thresh}: {np.mean(hit_ratio_list > inlier_ratio_thresh)}, "
        with open(str(output_root / "summary_fmr.txt"), "w") as fp:
            fp.write(result_string + "\n")
        logging.info(result_string)
    else:
        raise AttributeError(f"Evaluation type {args.eval_type} not supported")


def find_nn_cpu(feat0, feat1):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
    putative_match_indexes = None
    return nn_inds, putative_match_indexes


def valid_feat_ratio(coord0, coord1, feat0, feat1, trans_gth, thresh):
    coord0_copy = copy.deepcopy(coord0)
    coord0_copy = coord0_copy.transform(trans_gth)
    nn_inds, putative_inds_of_0 = find_nn_cpu(feat0, feat1)
    dist = np.sqrt(((np.array(coord0_copy.points) - np.array(coord1.points)[nn_inds]) ** 2).sum(1))
    return np.mean(dist < thresh)


if __name__ == "__main__":
    with torch.no_grad():
        main()
