import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["WANDB_API_KEY"] = "xxx"
os.environ["WANDB_MODE"] = "offline"

import argparse
from glob import glob
import numpy as np
from model import RetinexNet
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from datasets import CustomDataset
from torch.utils.data import DataLoader


def init_seeds(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='')

parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of total epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=96, help='patch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--data_dir', dest='data_dir', default='/data4/wangyh/RetinexNet/data/LOLdataset', help='directory storing the training data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpts/', help='directory for checkpoints')
parser.add_argument('--local_rank', default=0, help='if use distributed mode, must use variable local_rank')

parser.add_argument('--phase', default='train')

args = parser.parse_args()

def train(model, rank, wandb):

    lr = args.lr * np.ones([args.epochs])
    lr[20:] = lr[0] / 10.0

    '''  datasets  '''
    train_dataset = CustomDataset(args.data_dir, args.phase, args.patch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=args.batch_size, drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=False)

    args.phase = "val"
    val_dataset = CustomDataset(args.data_dir, args.phase, args.patch_size)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, batch_size=1, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler, num_workers=8, pin_memory=False)

    print('Number of training data: %d' % len(train_dataset))
    print('Number of valing   data: %d' % len(val_dataset))

    model.module.train_one(train_loader=train_loader,
                            val_loader=val_loader,
                            epoch=args.epochs,
                            lr=lr,
                            ckpt_dir=args.ckpt_dir,
                            eval_every_epoch=10,
                            train_phase="Decom",
                            rank=rank, wandb=wandb)

    model.module.train_one(train_loader=train_loader,
                            val_loader=val_loader,
                            epoch=args.epochs,
                            lr=lr,
                            ckpt_dir=args.ckpt_dir,
                            eval_every_epoch=10,
                            train_phase="Relight",
                            rank=rank, wandb=wandb)

if __name__ == '__main__':
    init_seeds(1234)
    # Create directories for saving the checkpoints and visuals

    '''  initial distributed mode  '''
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        gpu = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print("os.environ[\"WORLD_SIZE\"]: ", os.environ["WORLD_SIZE"])
        print("os.environ[\"RANK\"]: ", os.environ["RANK"])
        print("os.environ[\"LOCAL_RANK\"]: ", os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")

    torch.cuda.set_device(rank)
    dist_url = 'env://'
    dis_backend = 'nccl'  # communication: nvidia GPU recommened nccl
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    dist.init_process_group(backend=dis_backend, init_method=dist_url, world_size=world_size, rank=rank)
    dist.barrier()


    '''  wandb logger  '''
    if rank==0:
        wandb.init(project="RetinexNet", entity="harrytea", config=args)
    args.vis_dir = args.ckpt_dir + '/visuals/'
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)


    '''  model  '''
    model = RetinexNet().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    if rank==0:
        print("RetinexNet parameters: ", sum(param.numel() for param in model.parameters())/1e6)
    # Train the model
    train(model, rank, wandb)

