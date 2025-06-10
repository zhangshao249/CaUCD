import sys
from pathlib import Path
import os
import copy
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


from tqdm import tqdm
from utils.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from modules.segment_module import stochastic_sampling, ema_init, ema_update
from modules.segment_module import transform
from utils.dataloader import dataloader
from torch.cuda.amp import autocast, GradScaler
from utils.netloader import dino_loader, network_loader
# from torch.utils.tensorboard import SummaryWriter
from utils.metrics import ChangeEvaluator, SegmentEvaluator

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
def train(epoch, rank, args, net, segment, cluster, change, train_loader,
          optimizer_segment, optimizer_cluster, optimizer_change):
    segment.train()
    cluster.train()
    change.train()
    
    total_acc_linear = 0
    
    total_loss = 0
    total_loss_linear = 0
    total_loss_ctr = 0
    total_loss_cbk = 0
    total_loss_cd = 0
    
    eval_cd4cluster = ChangeEvaluator(args)
    eval_cd4pred = ChangeEvaluator(args)
    
    prog_bar = enumerate(train_loader)
    if rank in {0, -1}:
        prog_bar = tqdm(prog_bar, total=len(train_loader), mininterval=args.mininterval)
        
    for idx, batch in prog_bar:

        # optimizer
        with autocast():
            # image, label and self-supervised feature
            img1 = batch["img1"].cuda()
            img2 = batch["img2"].cuda()
            img = torch.cat([img1, img2], dim=0)
            
            if "SECOND" in args.dataset:
                label1 = batch["label1"].cuda()
                label2 = batch["label2"].cuda()
                label_cd = (label1 != label2).to(dtype=torch.int64)
                label = torch.cat([label1, label2], dim=0) - 1
                mask = (label >= 0) & (label < args.n_classes)
            elif "SYSU" in args.dataset:
                label_cd = batch["label_cd"].cuda()
                label = None            
            
            # if args.shuffle is True:
            #     bs = img.shape[0]
            #     order = get_random_order(bs)
            #     img = img[order]
            # intermediate features
            feat = net.forward_features(img)["x_norm_patchtokens"]
            ######################################################################
            # student
            seg_feat = segment.head(feat, img, drop=segment.dropout)
            proj_feat = segment.projection_head(seg_feat)
            ######################################################################
            
            ######################################################################
            # teacher
            seg_feat_ema = segment.head_ema(feat, img, drop=segment.dropout)
            proj_feat_ema = segment.projection_head_ema(seg_feat_ema)
            ######################################################################
            
            ######################################################################
            # grid
            if args.grid:
                feat_grid, order = stochastic_sampling(feat)
                proj_feat_grid, _ = stochastic_sampling(proj_feat, order=order)
                proj_feat_ema_grid, _ = stochastic_sampling(proj_feat_ema, order=order)
            ######################################################################
            
            ######################################################################
            # bank compute and contrastive loss
            cluster.bank_compute()
            if args.grid:
                loss_ctr = cluster.contrastive_ema_with_codebook_bank(feat_grid,
                                                                        proj_feat_grid,
                                                                        proj_feat_ema_grid)
            else:
                loss_ctr = cluster.contrastive_ema_with_codebook_bank(feat, proj_feat, proj_feat_ema)
            ######################################################################
            
            ######################################################################
            # cluster loss
            seg_feat_img = transform(seg_feat.detach())
            loss_cbk, cluster_logit = cluster.forward_centroid(seg_feat_img, feat)
            cluster_pred = cluster_logit.argmax(dim=1)
            ######################################################################
            
            ######################################################################
            # linear probe loss
            loss_linear = torch.tensor(0).cuda()
            if label is not None:
                linear_logit = segment.linear(seg_feat_img, feat)
                interp_linear_logit = F.interpolate(linear_logit, img.shape[-2:], mode='bilinear', align_corners=False)
                flat_linear_logit = interp_linear_logit.permute(0, 2, 3, 1).reshape(-1, args.n_classes)
                flat_label = label.reshape(-1)
                flat_mask = mask.reshape(-1)
                loss_linear = F.cross_entropy(flat_linear_logit[flat_mask], flat_label[flat_mask])
            ######################################################################
            
            ######################################################################
            # generate pseudo change detection
            cluster_pred1, cluster_pred2 = torch.chunk(cluster_pred, 2, dim=0)
            
            cluster_cd_mask = cluster_pred1 != cluster_pred2
            cluster_cd = cluster_cd_mask.to(dtype=torch.int64)
            ######################################################################
            
            ######################################################################
            
            # using DINO feature
            feat1, feat2 = torch.chunk(feat, 2, dim=0)
            cd_logit = change(feat1, feat2, img1, img2)
            loss_cd = F.cross_entropy(cd_logit, cluster_cd)
            ######################################################################
            
            
            # change prediction
            interp_cd_logit = F.interpolate(cd_logit, img.shape[-2:], mode='bilinear', align_corners=False)
            interp_pred_cd = interp_cd_logit.argmax(dim=1)
            
            # cluster prediction
            interp_cluster_logit = F.interpolate(cluster_logit, img.shape[-2:], mode='bilinear', align_corners=False)
            interp_cluster_cd = interp_cluster_logit.argmax(dim=1)
            interp_cluster_pred1, interp_cluster_pred2 = torch.chunk(interp_cluster_cd, 2, dim=0)
            interp_mask = interp_cluster_pred1 != interp_cluster_pred2
            interp_cluster_cd = interp_mask.to(dtype=torch.int64)
            
            # accuracy check
            pseudo_acc = (interp_cluster_cd == label_cd).sum() / label_cd.numel()
            pred_acc = (interp_pred_cd == label_cd).sum() / label_cd.numel()
            eval_cd4cluster.add_batch(interp_cluster_cd.cpu(), label_cd.cpu())
            eval_cd4pred.add_batch(interp_pred_cd.cpu(), label_cd.cpu())
            
            # loss
            loss = args.ctr*loss_ctr + loss_cbk + loss_linear + loss_cd 
            
        # optimizer
        optimizer_segment.zero_grad()
        optimizer_cluster.zero_grad()
        optimizer_change.zero_grad()
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer_segment) 
        torch.nn.utils.clip_grad_norm_(segment.parameters(), 1)
        
        scaler.step(optimizer_segment)
        scaler.step(optimizer_cluster)
        
        scaler.step(optimizer_change)
        scaler.update()
        
        # ema update
        ema_update(segment.head, segment.head_ema)
        ema_update(segment.projection_head, segment.projection_head_ema)
        
        # bank update
        cluster.bank_update(feat, proj_feat_ema)
        
        # linear probe acc check
        if label is not None:
            flat_linear_pred = flat_linear_logit.argmax(dim=1)
            acc_linear = (flat_linear_pred[flat_mask] == flat_label[flat_mask]).sum() / flat_label[flat_mask].numel()
            total_acc_linear += acc_linear.item()
            
        # loss check
        total_loss += loss.item()
        total_loss_ctr += loss_ctr.item()
        total_loss_cbk += loss_cbk.item()
        total_loss_linear += loss_linear.item()
        total_loss_cd += loss_cd.item()
        
        # real-time print
        desc = f'[Train] Loss=(ctr)+(cbk)+(linear)+(cd): {total_loss / (idx + 1):.2f}=' + \
        f'({total_loss_ctr / (idx + 1):.2f})' + \
        f'+({total_loss_cbk / (idx + 1):.2f})' + \
        f'+({total_loss_linear / (idx + 1):.2f})' + \
        f'+({total_loss_cd / (idx + 1):.2f})' 
        
        if label is not None:
            desc += f' ACC(linear): {100. * total_acc_linear / (idx + 1):.2f}%'
        desc += f' ACC(pse, pred): {100. * pseudo_acc:.2f}%, {100. * pred_acc:.2f}%'
        
        cluster_percent = torch.sum(interp_cluster_cd).item()/interp_cluster_cd.numel()
        pred_percent = torch.sum(interp_pred_cd).item()/interp_pred_cd.numel()
        label_percent = torch.sum(label_cd).item()/label_cd.numel()
        
        desc += f' PERCENT(cluster, pred, label): {100.*cluster_percent:.2f}%, {100.*pred_percent:.2f}%, {100.*label_percent:.2f}%'
        if rank in {0, -1}:
            prog_bar.set_description(desc, refresh=True)
            
        # Interrupt for sync GPU Process
        if args.distributed: 
            dist.barrier()
            
    cd_metric_cluster = eval_cd4cluster()
    cd_metric_pred = eval_cd4pred()
    
    if rank in {0, -1}:
        print('\ncluster cd: ')
        for k, v in cd_metric_cluster.items():
             print(f' {k}: {v:.2f}', end=',')
        print('\npred cd:')
        for k, v in cd_metric_pred.items():
             print(f' {k}: {v:.2f}', end=',')
        print()


@Wrapper.ValPrint
def val(rank, args, net, segment, cluster, change, val_loader, refine=False):
    eval_cd = ChangeEvaluator(args)    
    eval_seg4linear = SegmentEvaluator(args)
    eval_seg4cluster = SegmentEvaluator(args)
    if refine is True:
        eval_cd_refine = ChangeEvaluator(args)
        eval_cluster_cd = ChangeEvaluator(args)
    
    segment.eval()
    cluster.eval()
    change.eval()

    total_acc_linear = 0
    total_acc_cluster = 0
    total_acc_change = 0
    
    prog_bar = enumerate(val_loader)
    if rank in {0, -1}:
        prog_bar = tqdm(prog_bar, total=len(val_loader), mininterval=args.mininterval)
    for idx, batch in prog_bar:
        # image, label and self-supervised feature
        img1 = batch["img1"].cuda()
        img2 = batch["img2"].cuda()
        img = torch.cat([img1, img2], dim=0)
        if "SECOND" in args.dataset:
            label1 = batch["label1"].cuda()
            label2 = batch["label2"].cuda()
            label_cd = (label1 != label2).to(dtype=torch.int64)
            label = torch.cat([label1, label2], dim=0) - 1
            mask = (label >= 0) & (label < args.n_classes)
        elif "SYSU" in args.dataset:
            label_cd = batch["label_cd"].cuda()
            label = None

        # intermediate feature
        with torch.no_grad():
            feat = net.forward_features(img)["x_norm_patchtokens"]
            # segment
            seg_feat = segment.head(feat, img, drop=segment.dropout)
            if label is not None:
                # linear predict
                seg_feat_img = transform(seg_feat)
                linear_logit = segment.linear(seg_feat_img, feat)
                interp_linear_logit = F.interpolate(linear_logit, img.shape[-2:], mode='bilinear', align_corners=False)
                interp_linear_pred = interp_linear_logit.argmax(dim=1)
                eval_seg4linear.add_batch(interp_linear_pred[mask].cpu(), label[mask].cpu())
                
                # cluster predict
                cluster_logit = cluster.forward_centroid(seg_feat_img, feat, inference=True)
                interp_cluster_logit = F.interpolate(cluster_logit, img.shape[-2:], mode='bilinear', align_corners=False)
                interp_cluster_pred = interp_cluster_logit.argmax(dim=1)
                eval_seg4cluster.add_batch(interp_cluster_pred[mask].cpu(), label[mask].cpu())
                
                # linear probe acc check (segment)
                acc_linear = (label[mask] == interp_linear_pred[mask]).sum() / label[mask].numel()
                total_acc_linear += acc_linear.item()
                
                # cluster probe acc check (segment)
                acc_cluster = (label[mask] == interp_cluster_pred[mask]).sum() / label[mask].numel()
                total_acc_cluster += acc_cluster.item()
            else: # only for change detection
                # cluster predict
                seg_feat_img = transform(seg_feat)
                cluster_logit = cluster.forward_centroid(seg_feat_img, feat, inference=True)
                interp_cluster_logit = F.interpolate(cluster_logit, img.shape[-2:], mode='bilinear', align_corners=False)
                interp_cluster_pred = interp_cluster_logit.argmax(dim=1)
                
            # using DINO feature
            feat1, feat2 = torch.chunk(feat, 2, dim=0)
            
            # change 
            # change acc check
            cd_logit = change(feat1, feat2, img1, img2)
            interp_cd_logit = F.interpolate(cd_logit, img.shape[-2:], mode='bilinear', align_corners=False)
            interp_cd_pred = interp_cd_logit.argmax(dim=1)
            eval_cd.add_batch(interp_cd_pred.cpu(), label_cd.cpu())
            if refine is True:
                # cluster cd prediction
                cluster_pred1, cluster_pred2 = torch.chunk(interp_cluster_pred, 2, dim=0)
                cluster_cd_mask = cluster_pred1 != cluster_pred2
                cluster_cd_mask = cluster_cd_mask.to(dtype=torch.int64)
                eval_cluster_cd.add_batch(cluster_cd_mask.cpu(), label_cd.cpu())
                
                # refine cd prediction
                cd_img = torch.abs(img1 - img2)
                interp_cd_pred = refine_map(cd_img, interp_cd_logit)
                eval_cd_refine.add_batch(interp_cd_pred.cpu(), label_cd.cpu())
            
        acc_cd = (label_cd == interp_cd_pred).sum() / label_cd.numel()
        total_acc_change += acc_cd.item()
        
        # real-time print
        desc = f'[Val] Acc (Change): {100. * total_acc_change / (idx + 1):.3f}%'
        if label is not None:
            desc += f' Acc (Linear, Cluster): {100. * total_acc_linear / (idx + 1):.3f}%, {100. * total_acc_cluster / (idx + 1):.3f}%'
        
        if rank in {0, -1}:
            prog_bar.set_description(desc, refresh=True)
        
    cd_metric = eval_cd()
    if refine is True:
        cd_refine_metric = eval_cd_refine()
        cluster_cd_metric = eval_cluster_cd()
        
    if label is not None:
        seg_metric_cluster = eval_seg4cluster()
        seg_metric_linear = eval_seg4linear()
        if refine is True:
            seg_metric_cluster_class = eval_seg4cluster(by_class=True)
            
    if rank in {0, -1}:
        print("\npred cd:")
        for k, v in cd_metric.items():
             print(f' {k}: {v:.2f}', end=',')
        if refine is True:
            print("\nrefine pred cd:")
            for k, v in cd_refine_metric.items():
                print(f" {k}: {v:.2f}", end=',')
            print("\ncluster cd:")
            for k, v in cluster_cd_metric.items():
                print(f" {k}: {v:.2f}", end=',')
        if label is not None:
            print('\ncluster seg: ')
            for k, v in seg_metric_cluster.items():
                print(f' {k}: {v:.2f}', end=',')
            print('\nlinear seg:')
            for k, v in seg_metric_linear.items():
                print(f' {k}: {v:.2f}', end=',')
                
            if refine is True:
                print("\n cluster seg class:")
                for k, v in seg_metric_cluster_class.items():
                    print(f"{k}: {v}", end=', ')
        print()
    
    if args.distributed:
        dist.barrier()
        
    if label is not None:
        return cd_metric, seg_metric_cluster
    return cd_metric


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
    train_loader, val_loader, sampler = dataloader(rank, args)
    
    # network loader 
    net = dino_loader(rank, args)
    change, cluster, segment = network_loader(rank, args)
    
    # distributed parsing
    if args.distributed:
        net = net.module
        segment = segment.module
        cluster = cluster.module
        change = change.module

    # Bank and EMA
    cluster.bank_init()
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)
    
    ###################################################################################
    # First, run train_mediator.py
    path, is_exist = pickle_path_and_exist(rank, args)

    # early save for time
    if is_exist:
        # load
        codebook = np.load(path)
        cluster.codebook.data = torch.from_numpy(codebook).cuda()
        cluster.codebook.requires_grad = False
        segment.head.codebook = torch.from_numpy(codebook).cuda()
        segment.head_ema.codebook = torch.from_numpy(codebook).cuda()

        # print successful loading modularity
        rprint(f'Modularity {path} loaded', rank)

        # Interrupt for sync GPU Process
        if args.distributed: 
            dist.barrier()

    else:
        rprint('Train Modularity-based Codebook First', rank)
        return
    ###################################################################################

    # optimizer
    optimizer_segment = torch.optim.Adam(segment.parameters(), lr=args.lr * ngpus_per_node, weight_decay=1e-4)
    optimizer_cluster = torch.optim.Adam(cluster.parameters(), lr=args.lr * ngpus_per_node)
    optimizer_change = torch.optim.Adam(change.parameters(), lr=args.lr * ngpus_per_node)
    
    # train
    best_score = None
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
            segment,
            cluster,
            change,
            train_loader,
            optimizer_segment,
            optimizer_cluster,
            optimizer_change,
            )

        metrics = val(
            epoch, # for decorator
            rank, # for decorator
            args,
            net,
            segment,
            cluster,
            change,
            val_loader
            )
        
        # filepath hierarchy
        save_root = get_root_path(args)
        
        # save path
        best_path = save_root + '/best_model.pth'
        if rank in {0, -1}:
            check_dir(save_root)
            # save every epoch for resume
            weight_seg = segment.state_dict()
            weight_change = change.state_dict()
            weight_cluster = cluster.state_dict()
            save_content = {
                    "segment": weight_seg,
                    "change": weight_change,
                    "cluster": weight_cluster,
                    "epoch": epoch
            }
            
            save_path = save_root + f"/epoch_{epoch}.pth"
            
            save_content_cur = copy.deepcopy(save_content)
            save_content_cur["optimizer_segment"] = optimizer_segment.state_dict()
            save_content_cur["optimizer_cluster"] = optimizer_cluster.state_dict()
            save_content_cur["optimizer_change"] = optimizer_change.state_dict()
            
            # save_weight
            torch.save(save_content_cur, save_path)
            
            if len(metrics) == 2:
                cd_metric, seg_metric = metrics
                
                # cur_score = cd_metric["f1"]*0.5 + seg_metric["miou"]*0.5
                cur_score = cd_metric["f1"]
                # cur_score = cd_metric["miou"]
            else:
                cd_metric = metrics
                
                cur_score = cd_metric["f1"]
            
            # save best model
            if best_score is None or cur_score > best_score:
                best_score = cur_score
                
                save_content["f1"] = cd_metric["f1"]
                save_content["oa"] = cd_metric["oa"]
                save_content["rec"] = cd_metric["rec"]
                save_content["pre"] = cd_metric["pre"]
                
                # save_weight
                torch.save(save_content, best_path)
                
                print(f'-----------------VAL Epoch {epoch}: SAVING CHECKPOINT-----------------')
                
        # Interrupt for sync GPU Process
        if args.distributed: 
            dist.barrier()
        
    # val best model
    if os.path.exists(best_path):
        best_pth = torch.load(best_path, map_location=f'cuda:{rank}')
        change.load_state_dict(best_pth["change"], strict=False)
        segment.load_state_dict(best_pth["segment"], strict=False)
        cluster.load_state_dict(best_pth["cluster"], strict=False)
        best_epoch = best_pth["epoch"]
        val(
            f"best epoch_{best_epoch} val", # for decorator
            rank, # for decorator
            args,
            net,
            segment,
            cluster,
            change,
            val_loader,
            refine=True,
            )
    
    # Closing DDP
    if args.distributed: 
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    # fetch args
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--NAME-TAG', default='CaUCD', type=str)
    parser.add_argument('--root', default=ROOT / 'data', type=str)
    parser.add_argument('--dataset', default='SECOND', type=str)
    parser.add_argument('--crop-type', default=None, type=str)
    parser.add_argument('--ckpt', default=ROOT / 'dinov2/dinov2_vitb14_reg4_pretrain.pth', type=str)
    parser.add_argument('--patch-size', default=14, type=int)
    
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--cache', default=False, type=str2bool)
    parser.add_argument('--mininterval', default=10, type=int)
    
    parser.add_argument('--load-weight', default=None, type=str) # weight
    parser.add_argument('--cluster', default=True, type=str2bool)
    parser.add_argument('--change', default=True, type=str2bool)
    parser.add_argument('--segment', default=True, type=str2bool)
    
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=int(os.cpu_count() / 8), type=int)
    
    # DDP
    parser.add_argument('--gpu', default='4', type=str)
    parser.add_argument('--port', default='12355', type=str)
    
    # codebook parameter
    parser.add_argument('--grid', default=True, type=str2bool)
    parser.add_argument('--num-codebook', default=2048, type=int)
    
    # model parameter
    parser.add_argument('--reduced-dim', default=64, type=int) # segment dim
    parser.add_argument('--projection-dim', default=2048, type=int) # proj dim
    
    args = parser.parse_args()
    
    args.train_resolution = 322
    args.test_resolution = 322
    args.dim = 768
    args.num_queries = args.train_resolution**2 // int(args.patch_size)**2
    
    if "SECOND" in args.dataset:
        args.n_classes = 6 # cluster and linear 
        args.extra_classes = 0 # extra classes for cluster
        args.lr = 1e-3 # learning rate
        args.noise_type = "gaussian" # counterfactual noise type
        args.ctr = 0.1 # contrastive loss weight
    elif "SYSU" in args.dataset:
        args.n_classes = 6 # cluster
        args.extra_classes = 0 # extra classes for cluster
        args.lr = 1e-3 # learning rate
        args.noise_type = "gaussian" # counterfactual noise type
        args.ctr = 0.1 # contrastive loss weight
    
    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)
    
    args.world_size = ngpus_per_node
    args.distributed = ngpus_per_node > 1
    
    if args.distributed:
        # cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # multiprocess spawn
        mp.spawn(main, args=(args, ngpus_per_node), nprocs=ngpus_per_node, join=True)
    else:
        # first gpu index is activated once there are several gpu in args.gpu
        main(rank=0, args=args, ngpus_per_node=1)
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # main(rank=0, args=args, ngpus_per_node=1)