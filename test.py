import sys
from pathlib import Path
import os
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
from modules.segment_module import transform, untransform
from utils.dataloader import SYSU, SECOND
from utils.netloader import dino_loader, network_loader
from utils.metrics import ChangeEvaluator, SegmentEvaluator
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

cudnn.benchmark = True

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_clean():
    dist.destroy_process_group()

def save_img(path, img):
    img = Image.fromarray(img)
    img.save(path)

@Wrapper.TestPrint
def test(rank, args, net, segment, cluster, change,
         test_loader, img_list=None, refine=False, vis=None):    
    
    eval_seg_linear = SegmentEvaluator(args)
    eval_seg_cluster = SegmentEvaluator(args)
    
    eval_cd = ChangeEvaluator(args)
    eval_cd_cluster = ChangeEvaluator(args)
    
    if refine is True:
        eval_cd_refine = ChangeEvaluator(args)
        eval_cluster_refine = SegmentEvaluator(args)
    
    if segment is not None:
        segment.eval()
    if cluster is not None:
        cluster.eval()
    if change is not None:
        change.eval()
        
    if args.vis is not None or args.vis_seg is True:
        cmap = get_cmap()
    total_acc_linear = 0
    total_acc_cluster = 0
    total_acc_change = 0
    prog_bar = enumerate(test_loader)
    if rank in {0, -1}:
        prog_bar = tqdm(prog_bar, total=len(test_loader), mininterval=args.mininterval)
    
    for idx, batch in prog_bar:
        # image and label
        if "SECOND" in args.dataset:
            if len(batch) == 4:
                img1, img2, label1, label2 = batch
            elif len(batch) == 5:
                img1, img2, label1, label2, index = batch
            img1 = img1.cuda()
            img2 = img2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            label_cd = (label1 != label2).to(dtype=torch.int64)
            img = torch.cat([img1, img2], dim=0)
            label = torch.cat([label1, label2], dim=0) - 1
            mask = (label >= 0 ) & (label < args.n_classes)
            
        elif "SYSU" in args.dataset:
            if len(batch) == 3:
                img1, img2, label_cd = batch
            elif len(batch) == 4:
                img1, img2, label_cd, index = batch
            img1 = img1.cuda()
            img2 = img2.cuda()
            img = torch.cat([img1, img2], dim=0)
            label_cd = label_cd.cuda()
            label = None
        batch_size = img1.size(0)
        # intermediate feature
        with torch.no_grad():
            feat = net.forward_features(img)["x_norm_patchtokens"]
            # segment
            if label is not None or args.vis_seg is True:
                seg_feat = segment.head(feat, img, drop=segment.dropout)
            
            seg_feat_img = transform(seg_feat)
            # cluster predict
            cluster_logit = cluster.forward_centroid(seg_feat_img, crf=True)
            interp_cluster_logit = F.interpolate(cluster_logit, img.shape[-2:], mode='bilinear', align_corners=False)
            # not refine
            cluster_pred = interp_cluster_logit.argmax(dim=1)
            if label is not None:
                eval_seg_cluster.add_batch(cluster_pred.cpu(), label.cpu())
            # refine
            cluster_pred = refine_map(img, interp_cluster_logit)
            if label is not None:
                eval_cluster_refine.add_batch(cluster_pred[mask].cpu(), label[mask].cpu())
            
            if label is not None:
                # linear predict
                linear_logit = segment.linear(seg_feat_img)
                interp_linear_logit = F.interpolate(linear_logit, img.shape[-2:], mode='bilinear', align_corners=False)
                
                # linear probe acc check (segment)
                linear_pred = interp_linear_logit.argmax(dim=1)
                eval_seg_linear.add_batch(linear_pred[mask].cpu(), label[mask].cpu())
                
                acc_linear = (label[mask] == linear_pred[mask]).sum() / label[mask].numel()
                total_acc_linear += acc_linear.item()
                
                # cluster probe acc check (segment)
                acc_cluster = (label[mask] == cluster_pred[mask]).sum() / label[mask].numel()
                total_acc_cluster += acc_cluster.item()
                
            

        # change
        # cluster cd acc check
        cluster_pred1, cluster_pred2 = torch.chunk(cluster_pred, 2, dim=0)
            
        mask = cluster_pred1 != cluster_pred2
        cluster_cd = mask.to(dtype=torch.int64)
        
        # using DINO feature
        feat1, feat2 = torch.chunk(feat, 2, dim=0)
            
        cd_logit = change(feat1, feat2, img1, img2)
        interp_cd_logit = F.interpolate(cd_logit, img.shape[-2:], mode='bilinear', align_corners=False)
        cd_pred = interp_cd_logit.argmax(dim=1)
        eval_cd.add_batch(cd_pred.cpu(), label_cd.cpu())
        if refine is True:
            cd_img = torch.abs(img1 - img2)
            cd_pred = refine_map(cd_img, cd_logit)
            eval_cd_refine.add_batch(cd_pred.cpu(), label_cd.cpu())
            
        acc_cd = (label_cd == cd_pred).sum() / label_cd.numel()
        total_acc_change += acc_cd.item()
        
        
        eval_cd_cluster.add_batch(cluster_cd.cpu(), label_cd.cpu())
        # real-time print
        desc = f'[Test] Acc (Change): {100. * total_acc_change / (idx + 1):.3f}%'
        
        if rank in {0, -1}:
            prog_bar.set_description(desc, refresh=True)
            
        if vis is not None:
            vis_path = os.path.join(vis, args.dataset, args.image_set)
            if args.adjustment is True:
                vis_path += "/adjustment"
            else:
                vis_path += "/no_adjustment"
            pred_path = os.path.join(vis_path, "pred")
            gt_path = os.path.join(vis_path, "gt")
            check_dir(pred_path)
            check_dir(gt_path)
            
            if args.vis_seg is True:
                lb1_path = os.path.join(vis_path, "lb1")
                lb2_path = os.path.join(vis_path, "lb2")
                check_dir(lb1_path)
                check_dir(lb2_path)
                
            if args.vis_clusterCD is True:
                clusterCD_path = os.path.join(vis_path, "clusterCD")
                check_dir(clusterCD_path)
            
            ori_lb1_path = os.path.join(vis_path, "ori_lb1")
            ori_lb2_path = os.path.join(vis_path, "ori_lb2")
            check_dir(ori_lb1_path)
            check_dir(ori_lb2_path)
            
            for batch_index in range(batch_size):
                pred = cd_pred[batch_index].cpu().numpy().astype(np.uint8) * 255
                gt = label_cd[batch_index].cpu().numpy().astype(np.uint8) * 255
                clusterCD_pred = cluster_cd[batch_index].cpu().numpy().astype(np.uint8) * 255
                img_index = index[batch_index].item()
                
                if label is not None:
                    hungarian_pred = eval_seg_cluster.do_hungarian(cluster_pred.cpu())
                    hungarian_pred1, hungarian_pred2 = hungarian_pred.chunk(2, dim=0)
                    lb1 = hungarian_pred1[batch_index].cpu().numpy()
                    lb2 = hungarian_pred2[batch_index].cpu().numpy()
                    
                else:
                    lb1, lb2 = cluster_pred.chunk(2, dim=0)
                    lb1 = lb1[batch_index].cpu().numpy()
                    lb2 = lb2[batch_index].cpu().numpy()
                    
                color_lb1 = cmap[lb1]
                color_lb2 = cmap[lb2]
                
                save_img(os.path.join(ori_lb1_path, f"{img_list[img_index]}"), lb1.astype(np.uint8))
                save_img(os.path.join(ori_lb2_path, f"{img_list[img_index]}"), lb2.astype(np.uint8))
            
                save_img(os.path.join(lb1_path, f"{img_list[index[batch_index].item()]}"), color_lb1)
                save_img(os.path.join(lb2_path, f"{img_list[index[batch_index].item()]}"), color_lb2)
                
                save_img(os.path.join(pred_path, f"{img_list[img_index]}"), pred)
                save_img(os.path.join(gt_path, f"{img_list[img_index]}"), gt)
                
                if args.vis_clusterCD is True:
                    save_img(os.path.join(clusterCD_path, f"{img_list[img_index]}"), clusterCD_pred)
        
    cd_metric = eval_cd()
    cd_refine_metric = eval_cd_refine()
    cd_metric_cluster = eval_cd_cluster()
    if label is not None:
        seg_metric_cluster = eval_seg_cluster()
        seg_metric_linear = eval_seg_linear()
        seg_metric_cluster_refine = eval_cluster_refine()
        
        seg_metric_cluster_class = eval_seg_cluster(by_class=True)
        seg_metric_linear_class = eval_seg_linear(by_class=True)
        seg_metric_cluster_refine_class = eval_cluster_refine(by_class=True)
    if rank in {0, -1}:
        print("\npred cd:")
        for k, v in cd_metric.items():
             print(f' {k}: {v:.4f}', end=',')
        if refine is True:
            print("\npred cd refine:")
            for k, v in cd_refine_metric.items():
                print(f' {k}: {v:.4f}', end=',')
        print("\ncluster cd:")
        for k, v in cd_metric_cluster.items():
            print(f' {k}: {v:.4f}', end=',')
        if label is not None:
            print('\ncluster seg: ')
            for k, v in seg_metric_cluster.items():
                print(f' {k}: {v:.4f}', end=',')
            print('\nlinear seg:')
            for k, v in seg_metric_linear.items():
                print(f' {k}: {v:.4f}', end=',')
                
            if refine is True:
                print("\ncluster seg refine:")
                for k, v in seg_metric_cluster_refine.items():
                    print(f' {k}: {v:.4f}', end=',')
                
                print('\ncluster seg class:')
                for k, v in seg_metric_cluster_class.items():
                    print(f' {k}: {v}', end=',')
                    
                print("\ncluster seg refine class:")
                for k, v in seg_metric_cluster_refine_class.items():
                    print(f' {k}: {v}', end=',')
                    
                print('\nlinear seg class:')
                for k, v in seg_metric_linear_class.items():
                    print(f' {k}: {v}', end=',')
                    
        print()
        
    if args.distributed:
        dist.barrier()
        
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
    root = os.path.join(args.root, args.dataset)
    if "SECOND" in args.dataset:
        data_trans = get_second_transform(args.test_resolution, False)
        target_trans = get_second_transform(args.test_resolution, True)
        data = SECOND(root, args.image_set, data_trans, target_trans, normal=True, filename=False if args.vis is None else True)
    elif "SYSU" in args.dataset:
        data_trans = get_sysu_transform(args.test_resolution, False)
        target_trans = get_sysu_transform(args.test_resolution, True)
        data = SYSU(root, args.image_set, data_trans, target_trans, normal=True, filename=False if args.vis is None else True)
    if args.distributed:
        test_sampler = DistributedSampler(data, shuffle=False)
    test_loader = DataLoader(data, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, sampler=test_sampler if args.distributed else None)

    
    # network loader 
    net = dino_loader(rank, args)
    change, cluster, segment = network_loader(rank, args)
    
    # distributed parsing
    if args.distributed:
        net = net.module
        if segment is not None:
            segment = segment.module
        if cluster is not None:
            cluster = cluster.module
        if change is not None:
            change = change.module
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
    ###################################################################################
    test(
        "Test Process",
        rank,
        args,
        net,
        segment,
        cluster,
        change,
        test_loader,
        img_list = data.imgs,
        refine=args.refine,
        vis=args.vis
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
    parser.add_argument('--image-set', default='test', type=str)
    parser.add_argument('--ckpt', default=ROOT / 'dinov2/dinov2_vitb14_reg4_pretrain.pth', type=str)
    parser.add_argument('--patch-size', default=14, type=int)
   
    parser.add_argument('--vis', default=ROOT / "vis", type=str)
    parser.add_argument('--vis-seg', default=True, type=str2bool)
    parser.add_argument('--vis-clusterCD', default=True, type=str2bool)
    
    parser.add_argument('--refine', default=True, type=str2bool)
    parser.add_argument('--mininterval', default=10, type=int)
    
    
    parser.add_argument('--load-weight', default="best_model.pth", type=str) # weight
    parser.add_argument('--cluster', default=True, type=str2bool)
    parser.add_argument('--change', default=True, type=str2bool)
    parser.add_argument('--segment', default=True, type=str2bool)
    
    parser.add_argument('--batch-size', default=8, type=int)
    
    parser.add_argument('--num-workers', default=int(os.cpu_count() / 8), type=int)

    # DDP
    parser.add_argument('--gpu', default='0,1,3,4,5', type=str)
    parser.add_argument('--port', default='12355', type=str)
    
    # codebook parameter
    parser.add_argument('--num-codebook', default=2048, type=int)
    
    # model parameter
    parser.add_argument('--reduced-dim', default=64, type=int) # segment dim
    parser.add_argument('--projection-dim', default=2048, type=int) # proj dim

    # model parameter
    args = parser.parse_args()
    
    args.train_resolution = 322
    args.test_resolution = 322
    args.dim = 768
    args.num_queries = args.train_resolution**2 // int(args.patch_size)**2
    
    if "SECOND" in args.dataset:
        args.n_classes = 6
        args.extra_classes = 0
        args.noise_type = "gaussian" # counterfactual noise type
    elif "SYSU" in args.dataset:
        args.n_classes = 6 # plant, building, ground
        args.extra_classes = 0
        args.noise_type = "gaussian" # counterfactual noise type
    
    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)
    
    args.distributed = ngpus_per_node > 1
    args.world_size = ngpus_per_node
    
    if args.distributed:
        # cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # multiprocess spawn
        mp.spawn(main, args=(args, ngpus_per_node), nprocs=ngpus_per_node, join=True)
    else:
        # first gpu index is activated once there are several gpu in args.gpu
        main(rank=0, args=args, ngpus_per_node=1)