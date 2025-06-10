import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import reduced_all_gather

class ChangeEvaluator(object):
    def __init__(self, args=None):
        self.num_class = 2
        self.confusion_matrix = torch.zeros((self.num_class,) * 2)
        
        if args is not None:
            self.world_size = args.world_size

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mAcc = torch.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return torch.nan_to_num(Pre)

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return torch.nan_to_num(Rec)

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return torch.nan_to_num(F1)

    def Mean_Intersection_over_Union(self):
        MIoU = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        MIoU = torch.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].to(dtype=torch.int64) + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, pred_image, gt_image):
        assert gt_image.shape == pred_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pred_image)

    def __call__(self):
        """
        get final change metrics
        """
        if self.world_size > 1:
            cfs_mat = self.confusion_matrix.clone().cuda()
            cfs_mat = reduced_all_gather(cfs_mat, self.world_size)
            self.confusion_matrix = cfs_mat.cpu()
        Acc = self.Pixel_Accuracy()
        Pre = self.Pixel_Precision_Rate()
        Rec = self.Pixel_Recall_Rate()
        F1 = self.Pixel_F1_score()
        MIoU = self.Mean_Intersection_over_Union()
        return {
            "oa": Acc.item(),
            "pre": Pre.item(),
            "rec": Rec.item(),
            "f1": F1.item(),
            "miou": MIoU.item()
        }

class SegmentEvaluator(object):
    def __init__(self, args=None):
        self.n_classes = args.n_classes
        self.histogram = torch.zeros((self.n_classes, self.n_classes))
        
        if args is not None:
            self.world_size = args.world_size

    def scores(self, label_trues, label_preds):
        mask = (label_trues >= 0) & (label_trues < self.n_classes) & (label_preds >= 0) & (label_preds < self.n_classes)  # Exclude unlabelled data.
        hist = torch.bincount(self.n_classes * label_trues[mask] + label_preds[mask], \
                              minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes).t()
        return hist

    def add_batch(self, pred, label):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        self.histogram += self.scores(label, pred)
        
    def get_hist(self):
        self.assignments = linear_sum_assignment(self.histogram, maximize=True)
        hist = self.histogram[np.argsort(self.assignments[1]), :]
        tp = torch.diag(hist)
        fp = torch.sum(hist, dim=0) - tp
        fn = torch.sum(hist, dim=1) - tp
        return hist, tp, fp, fn
    
    def Mean_Intersection_over_Union(self, by_class=False):
        _, tp, fp, fn = self.get_hist()
        iou = tp / (tp + fp + fn)
        if by_class:
            return iou
        return iou[~torch.isnan(iou)].mean().item()
    
    def Mean_Average_precision(self, by_class=False):
        _, tp, _, fn = self.get_hist()
        prc = tp / (tp + fn)
        if by_class:
            return prc
        return prc[~torch.isnan(prc)].mean().item()
    
    def Pixel_Accuracy(self, by_class=False):
        hist, tp, *_ = self.get_hist()
        opc = torch.sum(tp) / torch.sum(hist)
        if by_class:
            return torch.diag(hist) / hist.sum(dim=1)
        return opc.item()

    def do_hungarian(self, clusters):
        self.assignments = linear_sum_assignment(self.histogram, maximize=True)
        return torch.tensor(self.assignments[1])[clusters]
    
    def __call__(self, by_class=False):
        if self.world_size > 1:
            hist = self.histogram.clone().cuda()
            hist = reduced_all_gather(hist, self.world_size)
            self.histogram = hist.cpu()
            
            
        miou = self.Mean_Intersection_over_Union(by_class)
        mAP = self.Mean_Average_precision(by_class)
        acc = self.Pixel_Accuracy(by_class)
        return {
            "miou": miou,
            "map": mAP,
            "acc": acc
        }