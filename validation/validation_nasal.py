import numpy as np
import open3d as o3d
import gc
import cv2
import tqdm
import torch
import MinkowskiEngine as ME
import logging

import utils


def validation_nasal(model, loader, epoch, loss_func, writer,
                     val_sampling_size, config_dict,
                     num_iter, device, vis_mesh, vis_mesh_freq):
    if "val_step" not in validation_nasal.__dict__:
        validation_nasal.val_step = 0
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
        except (ValueError, RuntimeError) as err:
            torch.cuda.empty_cache()
            logging.error(f"[validation_nasal] error {err}")
            continue

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

        T_gt = input_dict['T_gt'].numpy()

        tq.set_postfix(
            loss='avg: {:.3f}, cur: {:.3f}'.format(loss_meter.avg, loss.item()),
            remesh_ratio='cur: {:.3f}'.format(input_dict['scale_ratio'][0].item())
        )

        writer.add_scalar(
            'Validation/rr_loss', loss_meter.avg, global_step=validation_nasal.val_step)
        writer.add_scalar(
            'Validation/remesh_ratio', input_dict['scale_ratio'][0].item(), global_step=validation_nasal.val_step)

        if vis_mesh and batch_idx % vis_mesh_freq == 0:
            try:
                points_0, points_1, transformed_points_1 = utils.colorize_points_with_descriptor(
                    F0,
                    F1,
                    input_dict['pcd0'],
                    input_dict['pcd1'],
                    color_map=cv2.COLORMAP_JET,
                    T_est=T_gt[0])
            except ValueError as err:
                print(f"[colorize_points_with_descriptor] error at idx {batch_idx}: {err}")
                continue

            mesh0 = o3d.geometry.TriangleMesh()
            mesh0.vertices = o3d.utility.Vector3dVector(input_dict['mesh0_vertices'][0])
            mesh0.triangles = o3d.utility.Vector3iVector(input_dict['mesh0_faces'][0])

            mesh1 = o3d.geometry.TriangleMesh()
            mesh1.vertices = o3d.utility.Vector3dVector(input_dict['mesh1_vertices'][0])
            mesh1.triangles = o3d.utility.Vector3iVector(input_dict['mesh1_faces'][0])

            color_mesh_0 = utils.interpolate_color_from_point_to_mesh(mesh=mesh0, point_cloud=points_0)
            color_mesh_1 = utils.interpolate_color_from_point_to_mesh(mesh=mesh1, point_cloud=points_1)

            transformed_vertices1 = np.asarray(color_mesh_1.vertices) @ T_gt[0][:3, :3].T + T_gt[0][:3, 3]
            color_mesh_1.vertices = o3d.utility.Vector3dVector(transformed_vertices1)

            writer.add_mesh("visualization_mesh_0",
                            vertices=torch.from_numpy(np.asarray(color_mesh_0.vertices)).reshape(1, -1, 3),
                            colors=torch.from_numpy(np.asarray(color_mesh_0.vertex_colors) * 255).int().reshape(1, -1,
                                                                                                                3),
                            global_step=validation_nasal.val_step, config_dict=config_dict)
            writer.add_mesh("visualization_mesh_1",
                            vertices=torch.from_numpy(np.asarray(color_mesh_1.vertices)).reshape(1, -1, 3),
                            colors=torch.from_numpy(np.asarray(color_mesh_1.vertex_colors) * 255).int().reshape(1, -1,
                                                                                                                3),
                            global_step=validation_nasal.val_step, config_dict=config_dict)

        validation_nasal.val_step += 1
        gc.collect()
        torch.cuda.empty_cache()

    log = {
        'loss': loss_meter.avg
    }

    tq.close()
    return log
