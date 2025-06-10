import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.segment_module import Decoder, ProjectionHead, LinearHead
from modules.segment_module import Attention, Merge
from modules.segment_module import transform, untransform
    
class HeadChange(nn.Module):
    """
    Lightweight change head.
    Only one conv layer.
    """
    def __init__(self, args):
        super().__init__()
        
        # 2 for binary change detection (change or not change)
        n_classes = 2
        dim = args.dim
        
        self.decoder_feat = nn.Linear(dim, n_classes, bias=True)
        self.decoder_img = nn.Linear(dim, n_classes, bias=True)
        
        self.merge = Merge(3*n_classes, n_classes)
        
        self.noise_type = args.noise_type
        
    def forward(self, feat1, feat2, img1=None, img2=None, drop=nn.Identity()):
        if len(feat1.shape) == 4:
            feat1 = untransform(feat1)
        if len(feat2.shape) == 4:
            feat2 = untransform(feat2)
        
        feat_diff = torch.abs(feat1 - feat2)
        
        img1 = feat1
        img2 = feat2
            
        img1_logit = self.decoder_img(drop(img1))
        img2_logit = self.decoder_img(drop(img2))
        feat_logit = self.decoder_feat(drop(feat_diff))
        out_real = self.merge(feat_logit, img1_logit, img2_logit)
        
        bs, n, d = feat_diff.shape
        # should be same shape with feat_att
        if self.noise_type == 'gaussian':
            std, mean = torch.std_mean(feat_diff.view(bs, n*d), dim=-1)
            noise = torch.randn_like(feat_diff)
            noise = noise * std[:, None, None] + mean[:, None, None]
        elif self.noise_type == 'uniform':
            min_val = torch.min(feat_diff, keepdim=True, dim=-1)[0]
            max_val = torch.max(feat_diff, keepdim=True, dim=-1)[0]
            noise = torch.rand_like(feat_diff) 
            noise = noise * (max_val - min_val) + min_val
        elif self.noise_type == 'zero':
            noise = torch.zeros_like(feat_diff)
        
        noise_logit = self.decoder_feat(drop(noise))
        out_imag = self.merge(noise_logit, img1_logit, img2_logit)
        if self.training is True: # train
            out = out_real - out_imag
        else: # val 
            out = out_real- out_imag

        if len(out.shape) == 3:
            out = transform(out)
        return out

class Segment_TR(nn.Module):
    def __init__(self, args):
        super().__init__()

        ##################################################################################
        # dropout
        self.dropout = nn.Dropout(p=0.1)
        ##################################################################################
        
        ##################################################################################
        # TR Decoder Head 
        self.head = Decoder(args)
        self.projection_head = ProjectionHead(args)
        self.linear = LinearHead(args)
        ##################################################################################

        ##################################################################################
        # TR Decoder EMA Head
        self.head_ema = Decoder(args)
        self.projection_head_ema = ProjectionHead(args)
        ##################################################################################
        