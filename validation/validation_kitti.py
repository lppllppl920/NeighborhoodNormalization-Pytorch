import torch
import tqdm
import numpy as np
import MinkowskiEngine as ME
import gc
import cv2
import logging

import utils


def validation_kitti(model, loader, epoch, loss_func, writer,
                     val_sampling_size, config_dict,
                     num_iter, vis_mesh, device, vis_mesh_freq=None):
    if "val_step" not in validation_kitti.__dict__:
        validation_kitti.val_step = 0
    loss_meter = utils.AverageMeter()

    tq = tqdm.tqdm(total=num_iter, dynamic_ncols=True)
    tq.set_description(f"Validation - Epoch {epoch}")

    for batch_idx, (input_dict) in enumerate(loader):
        tq.update(1)

        if batch_idx > num_iter:
            break

        try:
            sinput0 = ME.SparseTensor(
                input_dict['sinput0_F'], coordinates=input_dict['sinput0_C'], device=device)
            F0 = model(sinput0).F

            sinput1 = ME.SparseTensor(
                input_dict['sinput1_F'], coordinates=input_dict['sinput1_C'], device=device)
            F1 = model(sinput1).F

            gt_pairs = input_dict['correspondences'][0].cuda()

            loss_gt_pairs = gt_pairs[:val_sampling_size]
            loss_0 = loss_func(F0, F1, loss_gt_pairs)
            loss_1 = loss_func(F1, F0, torch.cat([loss_gt_pairs[:, 1].reshape(-1, 1),
                                                  loss_gt_pairs[:, 0].reshape(-1, 1)],
                                                 dim=1))

            if torch.isnan(loss_0) or torch.isnan(loss_1):
                continue

            loss = 0.5 * loss_0 + 0.5 * loss_1
            loss_meter.update(loss.item())
        except (ValueError, RuntimeError) as err:
            torch.cuda.empty_cache()
            logging.error(f"[validation_kitti] error {err}")
            continue

        tq.set_postfix(
            loss='avg: {:.3f}, cur: {:.3f}'.format(loss_meter.avg, loss.item()),
            remesh_ratio='cur: {:.3f}'.format(input_dict['scale_ratio'][0].item())
        )

        writer.add_scalar(
            'Validation/loss', loss.item(), global_step=validation_kitti.val_step)
        writer.add_scalar(
            'Validation/remesh_ratio', input_dict['scale_ratio'][0].item(), global_step=validation_kitti.val_step)

        if vis_mesh and batch_idx % vis_mesh_freq == 0:
            try:
                T_gt = input_dict['T_gt'].numpy()
                points_1, points_0, transformed_points_0 = utils.colorize_points_with_descriptor(
                    F1,
                    F0,
                    input_dict['pcd1'],
                    input_dict['pcd0'],
                    color_map=cv2.COLORMAP_JET,
                    T_est=T_gt[0])
            except ValueError as err:
                print(f"[colorize_points_with_descriptor] error at idx {batch_idx}: {err}")
                continue

            writer.add_mesh("Validation/mesh_0",
                            vertices=torch.from_numpy(np.asarray(transformed_points_0.points)).reshape(1, -1, 3),
                            colors=torch.from_numpy(np.asarray(transformed_points_0.colors) * 255).int().reshape(1, -1,
                                                                                                                 3),
                            global_step=validation_kitti.val_step, config_dict=config_dict)
            writer.add_mesh("Validation/mesh_1",
                            vertices=torch.from_numpy(np.asarray(points_1.points)).reshape(1, -1, 3),
                            colors=torch.from_numpy(np.asarray(points_1.colors) * 255).int().reshape(1, -1,
                                                                                                     3),
                            global_step=validation_kitti.val_step, config_dict=config_dict)

        validation_kitti.val_step += 1
        gc.collect()
        torch.cuda.empty_cache()

    log = {
        'loss': loss_meter.avg
    }

    tq.close()
    return log
