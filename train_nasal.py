import MinkowskiEngine as ME
from pathlib import Path
import torch
import tqdm
import argparse
import numpy as np
import random
import datetime
import json
from tensorboardX import SummaryWriter
import gc
import logging
import open3d as o3d
import sys
import os
import signal
import warnings

warnings.filterwarnings("ignore")

from datasets import NasalDataset
import datasets
import utils
import models
import losses
import validation

import faulthandler

faulthandler.disable()
faulthandler.enable(all_threads=True)


def main():
    parser = argparse.ArgumentParser(
        description='Geometric Feature Learning on Nasal Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', type=str, required=True, help='config file')
    args = parser.parse_args()
    if args.config_path is None or not Path(args.config_path).exists():
        print(f"specified config path does not exist {args.config_path}")
        exit()

    with open(args.config_path, 'r') as f:
        args.__dict__ = json.load(f)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    date = datetime.datetime.now()

    if not args.continue_train or not args.load_trained_weights:
        log_root = Path(args.log_root) / "Nasal_train_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second)
        if not log_root.exists():
            log_root.mkdir(parents=True)
    else:
        log_root = None
        parents = Path(args.trained_model_path).parents
        for idx in range(len(parents)):
            if "Nasal_train" in str(parents[idx].name):
                log_root = parents[idx] / "{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.month, date.day, date.hour,
                                                                                      date.minute, date.second)
        if log_root is None:
            raise IOError("no proper continuation path found")
        if not log_root.exists():
            log_root.mkdir(parents=True)

    with open(str(log_root / 'commandline_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    ch = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(str(Path(log_root) / "log.txt"))

    if args.logging_mode.lower() == "info":
        logging_mode = logging.INFO
    elif args.logging_mode.lower() == "debug":
        logging_mode = logging.DEBUG
    else:
        raise AttributeError(f"logging mode {args.logging_mode} is not supported")

    logging.getLogger().setLevel(logging_mode)
    logging.basicConfig(
        format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch, fh])

    # atlas range train - [0.0, 2.5], val - [2.5, 3.0], test - [-3.0, 0.0]
    train_dataset = NasalDataset(
        mean_mesh_model_path=Path(args.mean_mesh_model_path),
        partial_mean_mesh_model_path=Path(args.partial_mean_mesh_model_path),
        atlas_mode_weights_path=Path(args.atlas_mode_weights_path),
        atlas_mode_weights_std_range=args.train_atlas_mode_weights_range,
        atlas_mode_range=args.atlas_mode_range,
        use_rotation=args.use_rotation,
        use_remesh=args.use_remesh,
        sampling_size=args.train_sampling_size,
        rotate_range=args.rotate_range,
        num_iter=args.train_num_iter,
        subdivide_factor=args.subdivide_factor,
        use_crop=args.use_crop,
        crop_ratio_range=args.crop_ratio_range,
        min_crop_remained_portion=args.crop_ratio_range[0],
        default_edge_length=args.default_edge_length,
        edge_length_range=args.edge_length_range,
        max_select_trial=args.max_sampling_trial,
        phase="train",
        oversampling_factor=args.oversampling_factor,
        batch_size=args.train_batch_size,
    )

    val_dataset = NasalDataset(
        mean_mesh_model_path=Path(args.mean_mesh_model_path),
        partial_mean_mesh_model_path=Path(args.partial_mean_mesh_model_path),
        atlas_mode_weights_path=Path(args.atlas_mode_weights_path),
        atlas_mode_weights_std_range=args.val_atlas_mode_weights_range,
        atlas_mode_range=args.atlas_mode_range,
        use_rotation=args.use_rotation,
        use_remesh=args.use_remesh,
        sampling_size=args.val_sampling_size,
        rotate_range=args.rotate_range,
        num_iter=args.val_num_iter,
        subdivide_factor=args.subdivide_factor,
        use_crop=args.use_crop,
        crop_ratio_range=args.crop_ratio_range,
        min_crop_remained_portion=args.crop_ratio_range[0],
        default_edge_length=args.default_edge_length,
        edge_length_range=args.edge_length_range,
        max_select_trial=args.max_sampling_trial,
        phase="val",
        oversampling_factor=args.oversampling_factor,
        batch_size=1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=datasets.separate_collate_pair_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
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
                               conv1_kernel_size=7,
                               norm_type=args.net_norm_type,
                               upsample_type=args.net_upsample_type,
                               epsilon=1.0e-8,
                               D=3)
    models.init_net(model, att_init_value=args.att_init_value,
                    type="kaiming", mode="fan_in", activation_mode="relu",
                    distribution="normal")
    models.count_parameters(model)

    trained_model_state = None
    model_state = None
    ignore_list = list()
    if args.load_trained_weights:
        if not Path(args.trained_model_path).exists():
            raise IOError("No pre-trained model detected")

        if args.partial_load:
            pre_trained_state = torch.load(str(args.trained_model_path))
            epoch = pre_trained_state['epoch'] + 1
            if 'step' in pre_trained_state:
                step = pre_trained_state['step']
            else:
                step = epoch * args.train_num_iter

            model_state = model.state_dict()
            if "model" in pre_trained_state:
                trained_state = pre_trained_state["model"]
            else:
                raise IOError(f"no state dict found in {args.trained_model_path}")

            trained_model_state = dict()
            for k, v in trained_state.items():
                if k in model_state:
                    shape = trained_state[k].shape
                    ori_k = k
                    if model_state[k].shape == shape:
                        trained_model_state[k] = v
                    else:
                        ignore_list.append(ori_k)
                else:
                    ignore_list.append(k)

            logging.info(
                f"Loading {len(trained_model_state.items())} chunks of parameters to the model to be trained which has "
                f"{len(model_state.items())}")
            model_state.update(trained_model_state)
            model.load_state_dict(model_state)
        else:
            logging.info("Loading {:s} ...".format(str(args.trained_model_path)))
            state = torch.load(str(args.trained_model_path))
            step = state['step']
            epoch = state['epoch'] + 1
            model.load_state_dict(state["model"])
            logging.info('Restored model, epoch {}, step {}'.format(epoch, step))

    else:
        epoch = 0
        step = 0

    loss_func = losses.RelativeResponseLoss(scale=args.rr_scale,
                                            standard_sample_size=5000)

    if args.partial_load and trained_model_state is not None and model_state is not None:
        if len(trained_model_state.items()) < len(model_state.items()):
            logging.info("Reset epoch to 0 when partial loading")
            epoch = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_range[1], momentum=0.9)
    lr_scheduler = models.CyclicLR(optimizer, base_lr=args.lr_range[0], max_lr=args.lr_range[1])

    config_dict = utils.vis_config()

    writer = SummaryWriter(log_dir=str(log_root))
    logging.info("Tensorboard visualization at {}".format(str(log_root)))

    # iterating through each instance of the proess
    for line in os.popen("ps ax | grep tensorboard | grep -v grep"):
        fields = line.split()
        # extracting Process ID from the output
        pid = fields[0]
        # terminating process
        os.kill(int(pid), signal.SIGKILL)

    os.system(f"tensorboard --logdir=\"{str(log_root)}\" --port=6006 --reload_multifile=true &")

    for cur_epoch in range(epoch, args.num_epochs + 1):
        # Set the seed correlated to cur_epoch for reproducibility
        torch.manual_seed(1 + cur_epoch)
        np.random.seed(1 + cur_epoch)
        random.seed(1 + cur_epoch)
        train_loader.dataset.randg.seed(1 + cur_epoch)
        train_loader.dataset.epoch = cur_epoch
        model.train()

        # Update progress bar
        tq = tqdm.tqdm(total=args.train_num_iter)
        total_loss_meter = utils.AverageMeter()

        if not args.val_only:
            for curr_iter, input_dict in enumerate(train_loader):
                if curr_iter >= args.train_num_iter:
                    break

                optimizer.zero_grad()
                lr_scheduler.batch_step(batch_iteration=step)
                tq.set_description('Epoch {}, lr {}'.format(cur_epoch, lr_scheduler.get_lr()))

                try:
                    sinput0 = ME.SparseTensor(
                        input_dict['sinput0_F'], coordinates=input_dict['sinput0_C'], device=args.device)
                    output0 = model(sinput0)
                    # (Minkowski BUG) Decomposed features are not the same as the original one
                    lengths_0 = [temp.shape[0] for temp in output0.decomposed_features]
                    sinput1 = ME.SparseTensor(
                        input_dict['sinput1_F'], coordinates=input_dict['sinput1_C'], device=args.device)
                    output1 = model(sinput1)
                    lengths_1 = [temp.shape[0] for temp in output1.decomposed_features]
                    pos_pairs = input_dict['correspondences']
                    offset_0 = 0
                    offset_1 = 0
                    loss = torch.tensor(0).to(args.device)
                    for batch_idx in range(args.train_batch_size):
                        length_0 = lengths_0[batch_idx]
                        length_1 = lengths_1[batch_idx]
                        batch_pos_pairs = pos_pairs[batch_idx].cuda()
                        temp_0 = loss_func(input0=output0.F[offset_0:offset_0 + length_0],
                                           input1=output1.F[offset_1:offset_1 + length_1],
                                           pos_pairs=batch_pos_pairs)
                        temp_1 = loss_func(input0=output1.F[offset_1:offset_1 + length_1],
                                           input1=output0.F[offset_0:offset_0 + length_0],
                                           pos_pairs=torch.cat([batch_pos_pairs[:, 1:2],
                                                                batch_pos_pairs[:, 0:1]],
                                                               dim=1))
                        temp = args.loss_weight * (0.5 * temp_0 + 0.5 * temp_1)
                        offset_0 += length_0
                        offset_1 += length_1
                        if batch_idx == 0:
                            loss = temp
                        else:
                            loss += temp
                    loss /= args.train_batch_size
                    optimizer.zero_grad()
                    loss.backward()

                except (ValueError, RuntimeError, IndexError) as err:
                    logging.error(err)
                    try:
                        optimizer.zero_grad()
                        optimizer.step()
                    except RuntimeError as err:
                        logging.error(err)
                        continue
                    torch.cuda.empty_cache()
                    continue

                if np.isnan(loss.item()):
                    logging.info(f"loss nan at {curr_iter}")
                    optimizer.zero_grad()
                    optimizer.step()
                    continue

                mean_att_list = list()
                std_att_list = list()
                for name, param in model.named_parameters():
                    if "mean_att" in name:
                        mean_att_list.append(param.detach())
                    if "std_att" in name:
                        std_att_list.append(param.detach())

                step += 1
                total_loss_meter.update(loss.item())
                tq.update(1)

                optimizer.step()

                if len(mean_att_list) >= 1:
                    mean_att = torch.mean(torch.cat(mean_att_list, dim=0), dim=0)
                    std_att = torch.mean(torch.cat(std_att_list, dim=0), dim=0)
                    tq.set_postfix(loss='avg: {:.3f}, cur: {:.3f}'.format(total_loss_meter.avg, loss.item()),
                                   scale_ratio='{:.3f}'.format(input_dict['scale_ratio'][0].item()),
                                   mean_att='{:.3f}'.format(mean_att[0].item()),
                                   std_att='{:.3f}'.format(std_att[0].item()),
                                   )
                    writer.add_scalar('Train/loss', loss.item(), step)
                    writer.add_scalars('Train', {'mean_att_0': mean_att[0].item(),
                                                 'std_att_0': std_att[0].item()}, step)
                else:
                    tq.set_postfix(loss='avg: {:.3f}, cur: {:.3f}'.format(total_loss_meter.avg, loss.item()),
                                   scale_ratio='{:.3f}'.format(input_dict['scale_ratio'][0].item()))
                    writer.add_scalar('Train/loss', loss.item(), step)

                torch.cuda.empty_cache()

            tq.close()
            gc.collect()
            torch.cuda.empty_cache()
            model_path_epoch = log_root / f'checkpoint_model.pt'
            utils.save_checkpoint(epoch=cur_epoch, step=step, model=model,
                                  config=args, path=model_path_epoch)

        if (cur_epoch + 1) % args.val_freq == 0:
            tq.close()
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
            val_loader.dataset.randg.seed(1)
            model.eval()

            with torch.no_grad():
                log = validation.validation_nasal(model=model, loader=val_loader, epoch=cur_epoch,
                                                  loss_func=loss_func, writer=writer,
                                                  val_sampling_size=args.val_sampling_size,
                                                  config_dict=config_dict, num_iter=args.val_num_iter,
                                                  device=args.device,
                                                  vis_mesh=args.vis_mesh,
                                                  vis_mesh_freq=args.vis_mesh_freq)

            gc.collect()
            torch.cuda.empty_cache()
            model_path_epoch = \
                log_root / \
                'checkpoint_model_epoch_{:d}_loss_{:.3f}.pt'.format(
                    cur_epoch, log['loss']
                )
            utils.save_checkpoint(epoch=cur_epoch, step=step, model=model,
                                  config=args, path=model_path_epoch)
            if args.val_only:
                print("Validation finished, program exiting")
                exit()


if __name__ == "__main__":
    main()
