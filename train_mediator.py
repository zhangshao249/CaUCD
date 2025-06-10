import sys
from pathlib import Path
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse

from tqdm import tqdm
from utils.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from modules.segment_module import compute_modularity_based_codebook
from utils.dataloader import dataloader
from torch.cuda.amp import autocast, GradScaler
from utils.netloader import dino_loader, network_loader

cudnn.benchmark = True
scaler = GradScaler()

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_clean():
    dist.destroy_process_group()

@Wrapper.EpochPrint
def train(epoch, rank, args, net, cluster, train_loader, optimizer):
    prog_bar = enumerate(train_loader)
    if rank in {0, -1}:
        prog_bar = tqdm(prog_bar, total=len(train_loader))
    for idx, batch in prog_bar:
        # image and label and self supervised feature
        img1 = batch["img1"].cuda()
        img2 = batch["img2"].cuda()
        img = torch.cat([img1, img2], dim=0)
        # intermediate feature
        feat = net.forward_features(img)['x_norm_patchtokens']

        # computing modularity based codebook
        loss_mod = compute_modularity_based_codebook(cluster.codebook, feat, grid=args.grid)

        # optimization
        optimizer.zero_grad()
        scaler.scale(loss_mod).backward()
        scaler.step(optimizer)
        scaler.update()

        # real-time print
        desc = f'[Train]: {loss_mod.detach().cpu().item():.2f}'
        if rank in {0, -1}:
            prog_bar.set_description(desc, refresh=True)

        # Interrupt for sync GPU Process
        if args.distributed: 
            dist.barrier()

def main(rank, args, ngpus_per_node):
    # setup ddp process and setting gpu id of this process
    if args.distributed: 
        ddp_setup(args, rank, ngpus_per_node)
        torch.cuda.set_device(rank)
    else:
        torch.cuda.set_device(int(args.gpu))

    # print argparse
    print_argparse(rank, args)

    # dataset loader
    train_loader, _, sampler = dataloader(rank, args)

    # network loader
    net = dino_loader(rank, args)
    _, cluster, _ = network_loader(rank, args)

    # distributed parsing
    if args.distributed: 
        net = net.module
        cluster = cluster.module

    # optimizer and scheduler
    optimizer = torch.optim.Adam(cluster.parameters(), lr=1e-3 * ngpus_per_node)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    ###################################################################################
    # train only modularity?
    path, is_exist = pickle_path_and_exist(rank, args)
        

    # early save for time
    if not is_exist:
        rprint("No File Exists!!", rank)
        # train
        for epoch in range(args.epoch):

            # for shuffle
            if args.distributed:
                sampler.set_epoch(epoch)

            # train
            train(
                epoch,  # for decorator
                rank,  # for decorator
                args,
                net,
                cluster,
                train_loader,
                optimizer)

            # scheduler step
            scheduler.step()

        # save
        if rank in {0, -1}:
            np.save(path, cluster.codebook.detach().cpu().numpy()
            if args.distributed else cluster.codebook.detach().cpu().numpy())

        # Interrupt for sync GPU Process
        if args.distributed:
            dist.barrier()

    else:
        rprint("Already Exists!!", rank)
    ###################################################################################


    # clean ddp process
    if args.distributed: 
        ddp_clean()


if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()

    # fixed parameter
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--load-weight', default=None, type=str2bool)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=int(os.cpu_count() / 8), type=int)
    parser.add_argument('--cache', default=False, type=str2bool)
    parser.add_argument('--patch-size', default=14, type=int)  
    
    # dataset and baseline
    parser.add_argument('--root', default=ROOT / 'data', type=str)
    parser.add_argument('--dataset', default='SECOND', type=str)
    parser.add_argument('--crop-type', default=None, type=str)
    
    parser.add_argument('--ckpt', default=ROOT / 'dinov2/dinov2_vitb14_reg4_pretrain.pth', type=str)
    parser.add_argument('--cluster', default=True, type=str2bool)
    parser.add_argument('--change', default=False, type=str2bool)
    parser.add_argument('--segment', default=False, type=str2bool)
    
    # DDP
    parser.add_argument('--gpu', default='4,5', type=str)
    parser.add_argument('--port', default='12368', type=str)

    # parameter
    parser.add_argument('--grid', default=True, type=str2bool)
    parser.add_argument('--num-codebook', default=2048, type=int)
    
    # model parameter
    parser.add_argument('--reduced-dim', default=64, type=int)
    parser.add_argument('--projection-dim', default=2048, type=int)

    args = parser.parse_args()

    args.train_resolution = 322
    args.test_resolution = 322
    args.dim = 768
    args.num_queries = args.train_resolution**2 // int(args.patch_size)**2
    
    if "SECOND" in args.dataset:
        args.n_classes = 6
    elif "SYSU" in args.dataset:
        args.n_classes = 6 # plant, building, ground

    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)
    
    args.distributed = ngpus_per_node > 1

    if args.distributed:
        # cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # multiprocess spawn
        mp.spawn(main, args=(args, ngpus_per_node), nprocs=ngpus_per_node, join=True)
    else:
        # first gpu index is activated once there are several gpu in args.gpu
        main(rank=0, args=args, ngpus_per_node=1)