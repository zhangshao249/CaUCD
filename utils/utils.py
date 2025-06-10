from torchvision import  transforms
import os
import torch
import random
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
from torchvision.transforms import InterpolationMode

invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.]),
                               ])
Trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

is_sym = lambda x: (x.transpose(1, 0) == x).all().item()



class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64)

def get_photometric_transform():
    return transforms.Compose([
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            transforms.RandomGrayscale(.2),
            transforms.RandomApply([transforms.GaussianBlur((5, 5))])
        ])

def get_sysu_transform(res, is_label):
    if is_label:
        return transforms.Compose([
            transforms.Resize((res, res), interpolation=InterpolationMode.NEAREST),
            ToTargetTensor()
            ])
    else:
        return transforms.Compose([
            transforms.Resize((res, res), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            ])
        
def get_second_transform(res, is_label):
    if is_label:
        return transforms.Compose([
            transforms.Resize((res, res), interpolation=InterpolationMode.NEAREST),
            ToTargetTensor()
            ])
    else:
        return transforms.Compose([
            transforms.Resize((res, res), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            ])

def get_root_path(args):
    root_path = f'weights/{args.dataset}'
    if args.crop_type is not None:
        root_path += f"/crop_{args.crop_type}"
    return root_path

def get_random_order(n):
    order = list(range(n))
    random.shuffle(order)
    return order

def get_cmap(n_classes=6):
    if n_classes == 6:
        cmap = np.array([[0, 128, 0],
                         [128, 128, 128],
                         [0, 255, 0],
                         [0, 0, 255],
                         [128, 0, 0],
                         [255, 0, 0]], dtype=np.uint8)
    else:
        cmap = np.array([[0, 0, 0],
                         [255, 0, 0],
                         [0, 255, 0],
                         [0, 0, 255],
                         [255, 255, 0],
                         [255, 0, 255],
                         [0, 255, 255],
                         [128, 255, 128],
                         [255, 128, 128],
                         [128, 128, 255],
                         [255, 128, 255],
                         [50, 50, 255],
                         [50, 255, 50],
                         [128, 50, 50],
                         [128, 50, 255],
                         [50, 255, 255],
                         [128, 255, 50],
                         [128, 0, 50],
                         [0, 255, 50],
                         [0, 128, 50],
                         [0, 50, 50]], dtype=np.uint8)
    return cmap

def refine_map(img, logit):
    return batched_crf(img, logit).argmax(1).cuda()

# utils
@torch.no_grad()
def reduced_all_gather(tensor, world_size):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = 0
    for tensor in tensors_gather:
        output += tensor
    return output

# from CAUSE    
import datetime
class Wrapper(object):
    @staticmethod
    def InitializePrint(func):
        def wrapper(rank, *args, **kwards):
            rprint(f'-------------Initialize VQ-VAE-------------', rank)
            func(*args, **kwards)
        return wrapper
    @staticmethod
    def TestPrint(func):
        def wrapper(epoch, rank, *args, **kwards):
            rprint(f'-------------TEST EPOCH: {epoch}-------------', rank)
            return func(rank, *args, **kwards)
        return wrapper
    @staticmethod
    def ValPrint(func):
        def wrapper(epoch, rank, *args, **kwards):
            rprint(f'-------------VAL EPOCH: {epoch}-------------', rank)
            return func(rank, *args, **kwards)
        return wrapper
    @staticmethod
    def EpochPrint(func):
        def wrapper(epoch, rank, *args, **kwards):
            rprint(f'-------------TRAIN EPOCH: {epoch}-------------', rank)
            func(epoch, rank, *args, **kwards)
        return wrapper
    @staticmethod
    def KmeansPrint(func):
        def wrapper(rank, *args, **kwards):
            rprint(f'-------------K-Means Clustering-------------', rank)
            func(*args, **kwards)
        return wrapper
    @staticmethod
    def TimePrint(func):
        def wrapper(*args, **kwargs):
            start = datetime.datetime.now()
            out = func(*args, **kwargs)
            end = datetime.datetime.now()
            print(f'[{func.__name__}] Time: {(end - start).total_seconds():.2f}sec')
            return out
        return wrapper

def pickle_path_and_exist(rank, args):
    save_dir = f"weight/{args.dataset}"
    if args.crop_type is not None:
        save_dir += f"/crop_{args.crop_type}"
    if rank in {0, -1}:
        check_dir(save_dir)
    filepath = save_dir + '/modular.npy'
    return filepath, os.path.exists(filepath)

def freeze(net):
    # net eval and freeze
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

def no_freeze(net):
    # net eval and freeze
    net.eval()
    for param in net.parameters():
        param.requires_grad = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_all(args, ind, img, label, cluster_preds, crf_preds, hungarian_preds, cmap, is_tr=False):
    baseline = args.ckpt.split('/')[-1].split('.')[0]
    y = f'{args.num_codebook}'
    check_dir(f'results')
    if is_tr:
        root = os.path.join('results', args.dataset, 'TR', baseline, y)
    else:
        root = os.path.join('results', args.dataset, 'MLP', baseline, y)

    check_dir(f'{root}/imgs')
    check_dir(f'{root}/labels')
    check_dir(f'{root}/kmeans')
    check_dir(f'{root}/crfs')
    check_dir(f'{root}/hungarians')
    # save image
    for id, i in [(id, x.item()) for id, x in enumerate(list(ind))]:
        torchvision.utils.save_image(invTrans(img)[id].cpu(), f'{root}/imgs/imgs_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[label[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/labels/labels_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[cluster_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/kmeans/kmeans_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[crf_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/crfs/crfs_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[hungarian_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/hungarians/hungarians_{i}.png')

def rprint(msg, rank=0):
    if rank==0: print(msg)

def num_param(f):
    out = 0
    for param in f.head.parameters():
        out += param.numel()
    return out

def imshow(img):
    a = 255 * invTrans(img).permute(1, 2, 0).cpu().numpy()
    plt.imshow(a.astype(np.int64))
    plt.show()

def plot(x):
    plt.plot(x.cpu().numpy())
    plt.show()



def getCMap(n_classes=27, cmapName='jet'):

    # Get jet color map from Matlab
    labelCount = n_classes
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    return cmap

def ckpt_to_name(ckpt):
    name = ckpt.split('/')[-1].split('_')[0]
    return name

def ckpt_to_arch(ckpt):
    arch = ckpt.split('/')[-1].split('.')[0]
    return arch

def print_argparse(rank, args):
    dict = vars(args)
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys(): print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')
        
# from STEGO
from scipy.optimize import linear_sum_assignment
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor, max_iter: int=10):
    MAX_ITER = max_iter
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 4
    Bi_XY_STD = 67
    Bi_RGB_STD = 3

    image = np.array(VF.to_pil_image(invTrans(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def batched_crf(img_tensor, prob_tensor):
    batch_size = list(img_tensor.size())[0]
    img_tensor_cpu = img_tensor.detach().cpu()
    prob_tensor_cpu = prob_tensor.detach().cpu()
    out = []
    for i in range(batch_size):
        out_ = dense_crf(img_tensor_cpu[i], prob_tensor_cpu[i])
        out.append(out_)
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in out], dim=0)

