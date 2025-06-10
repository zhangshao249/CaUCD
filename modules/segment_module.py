import math
import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import randperm as perm

"""
Below are Classes for CAUSE
"""
class Merge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(in_dim, 2*out_dim),
            nn.LeakyReLU(),
            nn.Linear(2*out_dim, out_dim)
        )
    def forward(self, *x):
        return self.pipe(torch.cat(x, dim=-1))

class ProjectionFeat(nn.Module):
    def __init__(self, args):
        """
        Projection DINO feature to Segmenation Feature.
        """
        super().__init__()
        self.proj = nn.Linear(args.dim, args.reduced_dim)
        
    def forward(self, feat):
        if len(feat.shape) == 4:
            feat = untransform(feat)
        feat = self.proj(feat)
        return feat
        
class ProjectionHead(nn.Module):
    def __init__(self, args):
        """
        Projection Head for Contrastive Learning
        """
        super().__init__()
        self.proj = nn.Linear(args.reduced_dim, args.projection_dim, bias=False)
        
    def forward(self, seg_feat):
        if len(seg_feat.shape) == 4:
            seg_feat = untransform(seg_feat)
        out = self.proj(seg_feat)
        return out
    
class LinearHead(nn.Module):
    def __init__(self, args):
        """
        Segmentation Head
        """
        super().__init__()
        self.linear = nn.Linear(args.reduced_dim, args.n_classes, bias=False)
        
    def forward(self, seg_feat, feat=None):
        if len(seg_feat.shape) == 4:
            seg_feat = untransform(seg_feat)
        out = self.linear(seg_feat)
        if len(out.shape) == 3:
            out = transform(out)
        return out

class Attention(nn.Module):
    def __init__(self, args, nhead=1, dropout=0.1, hidden_dim=2048):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(args.dim, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(args.dim, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(args.dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, args.dim)

        self.norm1 = nn.LayerNorm(args.dim)
        self.norm2 = nn.LayerNorm(args.dim)
        self.norm3 = nn.LayerNorm(args.dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.norm4 = nn.LayerNorm(args.dim)
        self.dropout4 = nn.Dropout(dropout)
        
        self.pos = nn.Parameter(torch.randn(1, args.num_queries, args.dim)*0.02)
    
    def forward(self, tgt, mem):
        # q = k = tgt + self.pos
        # tgt2 = self.self_attn(q, k, value=tgt)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(query=tgt + self.pos, key=mem, value=mem)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = mem + self.norm3(tgt)
        return tgt
    
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.f1 = nn.Linear(in_dim, out_dim)
        self.f2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.f1(x) + self.f2(x)
    

class Decoder(nn.Module):
    def __init__(self, args, codebook=None):
        super().__init__()
        self.codebook = codebook
        self.decoder_feat = Projector(args.dim, args.reduced_dim)
        
        self.att = Attention(args)
        
        self.noise_type = args.noise_type
        # self.counterfactual = nn.Parameter(torch.randn(1, 1, args.dim))
      
    def forward(self, feat, img=None, drop=nn.Identity()):
        feat_real = self.att(feat, feat)
        out_real = self.decoder_feat(drop(feat_real))
        
        bs, n, d = feat.shape
        # should be same shape with feat_att
        if self.noise_type == 'gaussian':
            std, mean = torch.std_mean(feat.view(bs, n*d), dim=-1)
            noise = torch.randn_like(feat)
            noise = noise * std[:, None, None] + mean[:, None, None]
        elif self.noise_type == 'uniform':
            min_val = torch.min(feat, keepdim=True, dim=-1)[0]
            max_val = torch.max(feat, keepdim=True, dim=-1)[0]
            noise = torch.rand_like(feat) 
            noise = noise * (max_val - min_val) + min_val
        elif self.noise_type == 'zero':
            noise = torch.zeros_like(feat)
        
        
        feat_imag = self.att(noise, feat)
        out_imag = self.decoder_feat(drop(feat_imag))
        if self.training is True: # train
            out = out_real - out_imag
        else: # val
            out = out_real - out_imag
        return out
          
    
    
        
class Cluster(nn.Module):
    def __init__(self, args):
        super().__init__()
        # projection dim
        self.dim = args.dim
        self.reduced_dim = args.reduced_dim
        self.projection_dim = args.projection_dim 

        # num codebook
        self.num_codebook = args.num_codebook

        # Codebook
        self.codebook = nn.Parameter(torch.empty(self.num_codebook, self.dim))
        reset(self.codebook, args.num_codebook)
        
        # cluster centroid
        self.cluster_probe = nn.Parameter(torch.randn(args.n_classes + args.extra_classes, self.reduced_dim))
        reset(self.cluster_probe, args.n_classes)
        self.proj_feat = ProjectionFeat(args)
        
    def bank_init(self):
        self.prime_bank = {}
        start_of_tensor = torch.empty([0, self.projection_dim]).cuda()
        for i in range(self.num_codebook):
            self.prime_bank[i] = start_of_tensor

    def bank_update(self, feat, proj_feat_ema, max_num=100):
        # load all and bank collection
        quant_ind = quantize_index(feat, self.codebook)
        for i in quant_ind.unique():
            # key bank
            key = proj_feat_ema[torch.where(quant_ind == i)]

            # 50% random cutting
            key = key[perm(len(key))][:int(len(key)*0.5)]

            # merging
            self.prime_bank[i.item()] = cat([self.prime_bank[i.item()], key], dim=0)

            # bank length
            length = len(self.prime_bank[i.item()])

            # if maximum number is over, slice by the order of the older
            if length >= max_num:
                self.prime_bank[i.item()] = self.prime_bank[i.item()][length-max_num:]

    def bank_compute(self):
        bank_vq_feat = torch.empty([0, self.dim]).cuda()
        bank_proj_feat_ema = torch.empty([0, self.projection_dim]).cuda()
        for key in self.prime_bank.keys():
            num = self.prime_bank[key].shape[0]
            if num == 0:
                continue
            bank_vq_feat = cat([bank_vq_feat, self.codebook[key].unsqueeze(0).repeat(num, 1)], dim=0)
            bank_proj_feat_ema = cat([bank_proj_feat_ema, self.prime_bank[key]], dim=0)

        # normalized feature and flat its feature for computing correspondence
        self.flat_norm_bank_vq_feat = F.normalize(bank_vq_feat, dim=1)
        self.flat_norm_bank_proj_feat_ema = F.normalize(bank_proj_feat_ema, dim=1)


    def contrastive_ema_with_codebook_bank(self, feat, proj_feat, proj_feat_ema, temp=0.07, pos_thresh=0.5, neg_thresh=0.1):
        """
        get all anchors and positive samples with same codebook index
        """

        # quantized feature to positive sample and negative sample
        vq_feat = vqt(feat, self.codebook)
        norm_vq_feat = F.normalize(vq_feat, dim=2)
        flat_norm_vq_feat = flatten(norm_vq_feat)

        # normalized feature and flat its feature for computing correspondence
        norm_proj_feat = F.normalize(proj_feat, dim=2)

        # normalized feature and flat its feature for computing correspondence
        norm_proj_feat_ema = F.normalize(proj_feat_ema, dim=2)
        flat_norm_proj_feat_ema = flatten(norm_proj_feat_ema)

        # selecting anchors by one-batch for all correspondence to all-batches
        # positive/negative
        loss_NCE_list = []
        for batch_ind in range(proj_feat.shape[0]):

            # anchor selection
            anchor_vq_feat = norm_vq_feat[batch_ind]
            anchor_proj_feat = norm_proj_feat[batch_ind]

            # cosine similarity of student-teacher
            cs_st = anchor_proj_feat @ flat_norm_proj_feat_ema.T

            # Codebook distance
            codebook_distance = anchor_vq_feat @ flat_norm_vq_feat.T
            bank_codebook_distance = anchor_vq_feat @ self.flat_norm_bank_vq_feat.T

            # [1] student-teacher (in-batch, local)
            pos_mask = (codebook_distance > pos_thresh)
            neg_mask = (codebook_distance < neg_thresh)

            auto_mask = torch.ones_like(pos_mask)
            auto_mask[:, batch_ind * pos_mask.shape[0]:(batch_ind + 1) * pos_mask.shape[0]].fill_diagonal_(0)
            pos_mask *= auto_mask

            cs_teacher = cs_st / temp
            shifted_cs_teacher = cs_teacher - cs_teacher.max(dim=1, keepdim=True)[0].detach()
            shifted_cs_teacher_with_only_neg = shifted_cs_teacher.exp() * (pos_mask + neg_mask)
            pos_neg_loss_matrix_teacher = -shifted_cs_teacher + torch.log(shifted_cs_teacher_with_only_neg.sum(dim=1, keepdim=True))
            loss_NCE_list.append(pos_neg_loss_matrix_teacher[torch.where(pos_mask!=0)].mean())

            # [2] student-teacher bank (out-batch, global)
            if self.flat_norm_bank_proj_feat_ema.shape[0] != 0:

                # cosine similarity of student-teacher bank
                cs_st_bank = anchor_proj_feat @ self.flat_norm_bank_proj_feat_ema.T

                bank_pos_mask = (bank_codebook_distance > pos_thresh)
                bank_neg_mask = (bank_codebook_distance < neg_thresh)

                cs_teacher_bank = cs_st_bank / temp
                shifted_cs_teacher_bank = cs_teacher_bank - cs_teacher_bank.max(dim=1, keepdim=True)[0].detach()
                shifted_cs_teacher_bank_with_only_neg = shifted_cs_teacher_bank.exp() * (bank_pos_mask + bank_neg_mask)
                pos_neg_loss_matrix_teacher_bank = -shifted_cs_teacher_bank + torch.log(shifted_cs_teacher_bank_with_only_neg.sum(dim=1, keepdim=True))

                # loss append
                loss_NCE_list.append(pos_neg_loss_matrix_teacher_bank[torch.where(bank_pos_mask!=0)].mean())

        # front
        loss_cbk = sum(loss_NCE_list) / float(len(loss_NCE_list))
        return loss_cbk

    def forward_centroid(self, x, feat=None, inference=False, alpha=2, crf=False):
        if len(x.shape) == 4:
            x = untransform(x)
            
        normed_clusters = F.normalize(self.cluster_probe, dim=1)
        x = F.normalize(x, dim=-1)
        
        if len(x.shape) == 3:
            x = transform(x)
            
        logits = torch.einsum("bchw,nc->bnhw", x, normed_clusters)
        
        if inference:
            return logits

        if crf:
            return torch.log_softmax(logits*alpha, dim=1)

        cluster_probs = F.one_hot(torch.argmax(logits, dim=1), self.cluster_probe.shape[0]) \
            .permute(0, 3, 1, 2).to(torch.float32)

        cluster_loss = -(cluster_probs * logits).sum(1).mean()
        return cluster_loss, logits
    

"""
Below are functions
"""


def transform(x):
    """
    B, P, D => B, D, root(P), root(P)

    Ex) 128, 400, 768 => 128, 768, 20, 20
    """
    B, P, D = x.shape
    return x.permute(0, 2, 1).view(B, D, int(math.sqrt(P)), int(math.sqrt(P)))

def untransform(x):
    """
    B, D, P, P => B, P*P, D,

    Ex) 128, 768, 20, 20 => 128, 400, 768
    """
    B, D, P, P = x.shape
    return x.view(B, D, -1).permute(0, 2, 1)

def flatten(x):
    """
    B, P, D => B*P, D

    Ex) 16, 400, 768 => 6400, 768
    """
    B, P, D = x.shape
    return x.contiguous().view(B*P, D)

def unflatten(x, batch_size):
    """
    B*P, D => B, P, D

    Ex) 6400, 768 => 16, 400, 768
    """
    P, D = x.shape
    return x.contiguous().view(batch_size, P//batch_size, D)

def stochastic_sampling(x, order=None, k=4):
    """
    pooling
    """
    if len(x.shape) == 3:
        x = transform(x)
    x_patch = x.unfold(2, k, k).unfold(3, k, k)
    x_patch = x_patch.permute(0, 2, 3, 4, 5, 1)
    x_patch = x_patch.reshape(-1, x_patch.shape[3:5].numel(), x_patch.shape[5])

    if order==None: 
        order = torch.randint(k ** 2, size=(x_patch.shape[0],))

    x_patch = x_patch[range(x_patch.shape[0]), order].reshape(x.shape[0], x.shape[2]//k, x.shape[3]//k, -1)
    x_patch = x_patch.permute(0, 3, 1, 2)
    x = untransform(x_patch)
    return x, order


def quantize_index(z, c, mode='cos'):
    if mode == 'cos':
        # computing distance
        dist = cos_distance_matrix(z, c)
    elif mode == 'l2':
        dist = l2_distance_matrix(z, c)

    # quantize
    return dist.argmax(dim=2)

def cos_distance_matrix(z, c):
    # flatten z
    z_flattened = z.contiguous().view(-1, z.shape[-1])
    norm_z = F.normalize(z_flattened, dim=1)
    norm_embed = F.normalize(c, dim=1)
    return torch.einsum("ab,cb->ac", norm_z, norm_embed).view(*z.shape[:-1], -1)

def l2_distance_matrix(z, c):
    # flatten z
    z_flattened = z.contiguous().view(-1, z.shape[-1])
    dist = (z_flattened.square().sum(dim=1, keepdims=True) + c.square().sum(dim=1).unsqueeze(0)
    -2 * z_flattened @ c.transpose(0, 1)) / c.shape[1]
    return torch.exp(-dist/z.shape[2]/2).view(*z.shape[:-1], -1)

def codebook_index(z, c):
    # computing distance
    dist = cos_distance_matrix(z, c)

    # codebook index
    return dist.argmax(dim=2)

def vqt(z, c):
    """
    Return Vector-Quantized Tensor
    """
    codebook_ind = codebook_index(z, c)
    return c[codebook_ind].view(*z.shape[:-1], c.shape[1])


def auto_cs(x):
    a = F.normalize(x, dim=1)
    return a @ a.T

def reset(x, n_c):
    x.data.uniform_(-1.0 / n_c, 1.0 / n_c)

def ema_init(x, x_ema):
    for param, param_ema in zip(x.parameters(), x_ema.parameters()):
        param_ema.data = param.data
        param_ema.requires_grad = False

def ema_update(x, x_ema, lamb=0.99):
    for student_params, teacher_params in zip(x.parameters(), x_ema.parameters()):
        teacher_params.data = lamb * teacher_params.data + (1-lamb) * student_params.data


def img_to_patch_for_affinity(img, patch_size):
    img_patch = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    img_patch = img_patch.permute(0, 2, 3, 1, 4, 5)
    img_patch = img_patch.reshape(*img_patch.shape[:3], -1)
    img_patch = img_patch.reshape(img_patch.shape[0], -1, img_patch.shape[3])
    return img_patch

def get_modularity_matrix_and_edge(x, mode='cos'):
    """
        getting W=(A-ddT/2m) and getting all edges (e)
    """
    if mode=='cos':
        norm = F.normalize(x, dim=2)
        A = (norm @ norm.transpose(2, 1)).clamp(0)
    elif mode=='l2':
        A = compute_self_distance_batch(x)

    A = A - A * torch.eye(A.shape[1]).cuda()
    d = A.sum(dim=2, keepdims=True)
    e = A.sum(dim=(1, 2), keepdims=True)
    W = A - (d / e) @ (d.transpose(2, 1) / e) * e
    return W, e

def cluster_assignment_matrix(z, c):
    norm_z = F.normalize(z, dim=2)
    norm_c = F.normalize(c, dim=1)
    return (norm_z @ norm_c.unsqueeze(0).transpose(2, 1)).clamp(0)

def compute_modularity_based_codebook(c, x, temp=0.1, grid=False):
    # detach
    x = x.detach()
    
    # pooling for reducing GPU memory allocation
    if grid:
        x, _ = stochastic_sampling(x)
        
    # modularity matrix and its edge matrix
    W, e = get_modularity_matrix_and_edge(x)
    
    # cluster assignment matrix
    C = cluster_assignment_matrix(x, c)
    
    # tanh with temperature
    D = C.transpose(2, 1)
    E = torch.tanh(D.unsqueeze(3) @ D.unsqueeze(2) / temp)
    delta, _ = E.max(dim=1)
    Q = (W / e) @ delta
    
    # trace
    diag = Q.diagonal(offset=0, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    
    return -trace.mean()

def compute_self_distance_batch(x):
    dist = x.square().sum(dim=2, keepdims=True) + x.square().sum(dim=2).unsqueeze(1) -2 * (x @ x.transpose(2, 1))
    return torch.exp(-dist/x.shape[2])

def img_to_patch(img, patch_size=16):
    img_patch = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    img_patch = img_patch.permute(0, 2, 3, 1, 4, 5)
    return img_patch.reshape(-1, *img_patch.shape[3:])

def patch_to_img(patch, batch_size=16, patch_size=16, img_size=320):
    patch_ = patch.reshape(batch_size, img_size//patch_size, img_size//patch_size, 3, patch_size, patch_size)
    patch_ = patch_.permute(0, 3, 1, 4, 2, 5)
    return patch_.reshape(batch_size, 3, img_size, img_size)