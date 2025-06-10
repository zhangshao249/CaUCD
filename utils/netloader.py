from utils.utils import *
from modules.segment import Segment_TR, HeadChange
from modules.segment_module import Cluster
from dinov2.utils import build_dino_model
from torch.nn.parallel import DistributedDataParallel

def dino_loader(rank, args):
    # load network
    net = load_model(rank, args.ckpt).cuda()
    if args.distributed:
        net = DistributedDataParallel(net, device_ids=[rank])
    freeze(net)
    return net

def network_loader(rank, args):
    segment = None
    cluster = None
    change = None
        
    pth_dir = get_root_path(args)
    
    if args.load_weight is not None:
        pth_path = os.path.join(pth_dir, args.load_weight)
        pth = torch.load(pth_path, map_location=f'cuda:{rank}')
    
    if args.segment is True:
        segment = Segment_TR(args).cuda()
        if args.load_weight is not None:
            segment.load_state_dict(pth['segment'], strict=False)
        if args.distributed:
            segment = DistributedDataParallel(segment, device_ids=[rank])
    if args.cluster is True:
        cluster = Cluster(args).cuda()
        if args.load_weight is not None:
            cluster.load_state_dict(pth['cluster'], strict=False)
        if args.distributed:
            cluster = DistributedDataParallel(cluster, device_ids=[rank])
    if args.change is True:
        change = HeadChange(args).cuda()
        if args.load_weight is not None:
            change.load_state_dict(pth['change'], strict=False)
        if args.distributed:
            change = DistributedDataParallel(change, device_ids=[rank])
    return change, cluster, segment

def checkpoint_module(checkpoint, net):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    msg = net.load_state_dict(new_state_dict, strict=False)
    return msg

def load_model(rank, ckpt):
    net, msg = build_dino_model(ckpt)

    # check incompatible layer or variables
    rprint(msg, rank)

    return net