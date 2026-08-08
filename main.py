import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset
import glob

warnings.filterwarnings("ignore")


def find_dataset_dir():
    """Find the dataset directory automatically"""
    possible_paths = [
        'food-ingredients-5',
        './food-ingredients-5',
        '../food-ingredients-5',
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(f'{path}/train/images'):
            return path

    return 'food-ingredients-5'


def train(args, params, data_dir):
    # Model
    model = nn.yolo_v11_n(len(params['names']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = glob.glob(f'{data_dir}/train/images/*.jpg')

    if not filenames:
        print(f"Error: No training images found in {data_dir}/train/images/")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Dataset directory exists: {os.path.exists(data_dir)}")
        if os.path.exists(data_dir):
            print(f"Train images directory exists: {os.path.exists(f'{data_dir}/train/images')}")
        raise FileNotFoundError(f"No training images found in {data_dir}/train/images/")

    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                                                     'metrics/precision', 'metrics/recall', 'metrics/mAP@50', 'metrics/mAP',
                                                     'val/box_loss', 'val/cls_loss', 'val/dfl_loss'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'train_box', 'train_cls', 'train_dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.to(device).float() / 255

                # Forward
                with torch.amp.autocast(device.type):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size
                loss_cls *= args.batch_size
                loss_dfl *= args.batch_size
                loss_box *= args.world_size
                loss_cls *= args.world_size
                loss_dfl *= args.world_size

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    # amp_scale.unscale_(optimizer)
                    # util.clip_gradients(model)
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP and validation losses
                last = test(args, params, data_dir, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'train/box_loss': str(f'{avg_box_loss.avg:.3f}'),
                                 'train/cls_loss': str(f'{avg_cls_loss.avg:.3f}'),
                                 'train/dfl_loss': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'metrics/mAP': str(f'{last[0]:.3f}'),
                                 'metrics/mAP@50': str(f'{last[1]:.3f}'),
                                 'metrics/recall': str(f'{last[2]:.3f}'),
                                 'metrics/precision': str(f'{last[3]:.3f}'),
                                 'val/box_loss': str(f'{last[4]:.3f}'),
                                 'val/cls_loss': str(f'{last[5]:.3f}'),
                                 'val/dfl_loss': str(f'{last[6]:.3f}')})
                log.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[0]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers


@torch.no_grad()
def test(args, params, data_dir, model=None):
    filenames = glob.glob(f'{data_dir}/valid/images/*.jpg')

    if not filenames:
        print(f"Error: No validation images found in {data_dir}/valid/images/")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Dataset directory exists: {os.path.exists(data_dir)}")
        if os.path.exists(data_dir):
            print(f"Valid images directory exists: {os.path.exists(f'{data_dir}/valid/images')}")
        raise FileNotFoundError(f"No validation images found in {data_dir}/valid/images/")

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot = False
    if model is None:
        plot = True
        model = torch.load(f='./weights/best.pt', map_location=device)
        model = model['model'].float().fuse()

    criterion = util.ComputeLoss(model, params)

    model.eval()
    model.half()  # Convert to FP16 once before the loop
    avg_val_box_loss = util.AverageMeter()
    avg_val_cls_loss = util.AverageMeter()
    avg_val_dfl_loss = util.AverageMeter()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).to(device)  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 8) % ('', 'precision', 'recall', 'mAP50', 'mAP', 'val_box', 'val_cls', 'val_dfl'))
    for samples, targets in p_bar:
        samples = samples.to(device)
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).to(device)

        # Compute validation losses
        model.train()
        with torch.amp.autocast(device.type):
            model_outputs = model(samples.float())
            loss_box, loss_cls, loss_dfl = criterion(model_outputs, targets)

            avg_val_box_loss.update(loss_box.item(), samples.size(0))
            avg_val_cls_loss.update(loss_cls.item(), samples.size(0))
            avg_val_dfl_loss.update(loss_dfl.item(), samples.size(0))

        # Switch back to eval mode for inference
        model.eval()
        outputs = model(samples.half())
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = (targets['idx'] == i).squeeze()
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.to(device)
            box = box.to(device)

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).to(device), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=params["names"])

    print(('%10s' + '%10.3g' * 7) % ('', m_pre, m_rec, map50, mean_ap, avg_val_box_loss.avg, avg_val_cls_loss.avg, avg_val_dfl_loss.avg))

    model.float()
    return mean_ap, map50, m_rec, m_pre, avg_val_box_loss.avg, avg_val_cls_loss.avg, avg_val_dfl_loss.avg


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    data_dir = find_dataset_dir()

    # Debug information
    if args.local_rank == 0:
        print(f"Dataset directory: {data_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Dataset exists: {os.path.exists(data_dir)}")
        if os.path.exists(data_dir):
            print(f"Train images dir exists: {os.path.exists(f'{data_dir}/train/images')}")
            print(f"Valid images dir exists: {os.path.exists(f'{data_dir}/valid/images')}")
            train_count = len(glob.glob(f'{data_dir}/train/images/*.jpg'))
            valid_count = len(glob.glob(f'{data_dir}/valid/images/*.jpg'))
            print(f"Number of training images: {train_count}")
            print(f"Number of validation images: {valid_count}")

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params, data_dir)
    if args.test:
        test(args, params, data_dir)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
