import random
import os
from os.path import join

from tqdm import tqdm

import torch.multiprocessing
from PIL import Image
from torch.utils.data import DataLoader, Dataset


from utils.utils import *

from torch.utils.data.distributed import DistributedSampler

def get_random():
    return random.randint(0, 2147483647)

class SECOND(Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, cache=False, normal=False, filename=False):
        self.img_dir = os.path.join(root, image_set)
        self.transform = transform
        self.target_transform = target_transform
        self.cache = cache
        self.filename = filename
        
        self.normalize = None
        if normal is True:
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        
        self.imgs = os.listdir(os.path.join(self.img_dir, "im1"))
    
    def __getitem__(self, index):
        """
        return img1, img2, label1, label2, 
        """
        img1_path  = os.path.join(self.img_dir, "im1", self.imgs[index])
        img2_path = os.path.join(self.img_dir, "im2", self.imgs[index])
        label1_path = os.path.join(self.img_dir, "label1", self.imgs[index])
        label2_path = os.path.join(self.img_dir, "label2", self.imgs[index])
        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img2_path).convert('RGB')
        label1 = Image.open(label1_path)
        label2 = Image.open(label2_path)
        
        if self.transform is not None:
            seed = get_random()
            self._set_seed(seed)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            label1 = self.target_transform(label1)
            label2 = self.target_transform(label2)
        
        if self.normalize is not None:
            image1 = self.normalize(image1)
            image2 = self.normalize(image2)
            
        if self.filename is True:
            index = torch.tensor(index, dtype=torch.int64)
            return image1, image2, label1, label2, index
        return image1, image2, label1, label2
    
    def __len__(self):
        return len(self.imgs)
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        
class SYSU(Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, normal=False, filename=False):
        self.img_dir = os.path.join(root, image_set)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        
        self.normalize = None
        if normal is True:
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.imgs = os.listdir(os.path.join(self.img_dir, "A"))
    def __getitem__(self, index):
        img1_path  = os.path.join(self.img_dir, "A", self.imgs[index])
        img2_path = os.path.join(self.img_dir, "B", self.imgs[index])
        label_path = os.path.join(self.img_dir, "label", self.imgs[index])
        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img2_path).convert('RGB')
        label = Image.open(label_path)
        
        if self.transform is not None:
            seed = get_random()
            self._set_seed(seed)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            label = self.target_transform(label)
            
            # to check if the label is 0-1
            if label.max() > 1:
                label = label/255
            
        if self.normalize is not None:
            image1 = self.normalize(image1)
            image2 = self.normalize(image2)
            
        if self.filename is True:
            index = torch.tensor(index, dtype=torch.int64)
            return image1, image2, label, index
        return image1, image2, label
    
    def __len__(self):
        return len(self.imgs)
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)


def dataloader(rank, args, no_ddp_train_shuffle=True):

    if "SYSU" in args.dataset:
        get_transform = get_sysu_transform
    elif "SECOND" in args.dataset:
        get_transform = get_second_transform
    
    extra_transform = None

    # train dataset
    train_dataset = ContrastiveSegDataset(
        root=args.root,
        dataset_name=args.dataset,
        crop_type=args.crop_type,
        image_set="train",
        transform=get_transform(args.train_resolution, False),
        target_transform=get_transform(args.train_resolution, True),
        extra_transform=extra_transform,
        cache=args.cache,
        rank=rank
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)

    # train loader
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=False if args.distributed else no_ddp_train_shuffle, 
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, sampler=train_sampler if args.distributed else None)

    test_dataset = ContrastiveSegDataset(
        root=args.root,
        dataset_name=args.dataset,
        crop_type=None,
        image_set="val",
        transform=get_transform(args.test_resolution, False),
        target_transform=get_transform(args.test_resolution, True),
        cache=args.cache,
        rank=rank
    )

    if args.distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # test dataloader
    test_loader = DataLoader(test_dataset, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, sampler=test_sampler if args.distributed else None)

    sampler = train_sampler if args.distributed else None

    return train_loader, test_loader, sampler




class CroppedDataset(Dataset):
    def __init__(self, root, dataset_name, crop_type, crop_ratio, image_set, transform, target_transform):
        super(CroppedDataset, self).__init__()
        
        self.dataset_name = dataset_name
        self.image_set = image_set
        
        self.root = join(root, dataset_name, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.transform = transform
        self.target_transform = target_transform
        
        self.img_dir = join(self.root, image_set, "A")
        self.imgs = os.path.join(self.img_dir)
        
        self.num_images = len(os.listdir(self.imgs))

    def __getitem__(self, index):
        img1_path = os.path.join(self.root, self.image_set, "A", "{}.png".format(index))
        img2_path = os.path.join(self.root, self.image_set, "B", "{}.png".format(index))
        target_path = os.path.join(self.root, self.image_set, "label", "{}.png".format(index))
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        target = Image.open(target_path)
        
        seed = get_random()
        
        self._set_seed(seed)
        img1 = self.transform(img1)
        
        self._set_seed(seed)
        img2 = self.transform(img2)
        
        self._set_seed(seed)
        target = self.target_transform(target)

        target = target
        return img1, img2, target

    def __len__(self):
        return self.num_images
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)



class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 root,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 extra_transform=None,
                 cache=False,
                 rank=0
                 ):
        super().__init__()
        self.image_set = image_set
        self.dataset_name = dataset_name
        
        self.transform = transform
        self.target_transform = target_transform
        self.extra_transform = extra_transform
        
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.cache = cache
        
        # cityscapes, cocostuff27 
        if  "SYSU" in dataset_name and crop_type is None:
            dataset_class = SYSU
            root = os.path.join(root, dataset_name)
            extra_args = dict()
        elif "SECOND" in dataset_name and crop_type is None:
            dataset_class = SECOND
            root = os.path.join(root, dataset_name)
            extra_args = dict()

        # cityscapes, cocostuff27 [Crop]
        elif "SYSU" in dataset_name and crop_type is not None:
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name=dataset_name, crop_type=crop_type, crop_ratio=0.5)
        elif "SECOND" in dataset_name and crop_type is not None:
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name=dataset_name, crop_type="five", crop_ratio=0.5)

        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.dataset = dataset_class(
            root=root,
            image_set=self.image_set,
            **extra_args)
        
        if self.cache is True:
            if "SYSU" in dataset_name:
                self.cache_data = {"img1": [], "img2": [], "label_cd": []}
            elif "SECOND" in dataset_name:
                self.cache_data = {"img1": [], "img2": [], "label1": [], "label2": []}
                
            prog_bar = range(len(self.dataset))
            if rank in {0, -1}:
                prog_bar = tqdm(prog_bar)
            for i in prog_bar:
                pack = self.dataset[i]
                if len(pack) == 3:
                    img1, img2, label_cd = pack
                    self.cache_data["img1"].append(img1)
                    self.cache_data["img2"].append(img2)
                    self.cache_data["label_cd"].append(label_cd)
                elif len(pack) == 4:
                    img1, img2, label1, label2 = pack
                    self.cache_data["img1"].append(img1)
                    self.cache_data["img2"].append(img2)
                    self.cache_data["label1"].append(label1)
                    self.cache_data["label2"].append(label2)
                desc = f"Cache:"
                if rank in {0, -1}:
                    prog_bar.set_description(desc, refresh=True)
            

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, index):
        ret = {}
        if self.cache is True:
            if len(self.cache_data.keys()) == 3:
                img1 = self.cache_data["img1"][index]
                img2 = self.cache_data["img2"][index]
                label_cd = self.cache_data["label_cd"][index]
                
                
            elif len(self.cache_data.keys()) == 4:
                img1 = self.cache_data["img1"][index]
                img2 = self.cache_data["img2"][index]
                label1 = self.cache_data["label1"][index]
                label2 = self.cache_data["label2"][index]
                
        else:
            pack = self.dataset[index]
            if len(pack) == 3: # SYSU dataset
                img1, img2, label_cd = pack
                
                # transforms
                seed = get_random()
                self._set_seed(seed)
                
                # augmentation image
                if self.extra_transform is not None:
                    img1_aug = self.extra_transform(img1)
                    img2_aug = self.extra_transform(img2)
                    
                    img1_aug = self.transform(img1_aug)
                    img2_aug = self.transform(img2_aug)
                    
                    img1_aug = self.normalize(img1_aug)
                    img2_aug = self.normalize(img2_aug)
                    
                    ret["img1_aug"] = img1_aug
                    ret["img2_aug"] = img2_aug                
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                
                img1 = self.normalize(img1)
                img2 = self.normalize(img2)
                
                label_cd = self.target_transform(label_cd)
                
                # to check if the label is 0-1
                if label_cd.max() > 1:
                    label_cd = label_cd / 255.0
                    
                ret["img1"] = img1
                ret["img2"] = img2
                ret["label_cd"] = label_cd
                
            elif len(pack) == 4: # SECOND dataset
                img1, img2, label1, label2 = pack
                
                # transforms
                seed = get_random()
                self._set_seed(seed)
                
                # augmentation image
                if self.extra_transform is not None:
                    img1_aug = self.extra_transform(img1)
                    img2_aug = self.extra_transform(img2)
                    
                    img1_aug = self.transform(img1_aug)
                    img2_aug = self.transform(img2_aug)
                    
                    img1_aug = self.normalize(img1_aug)
                    img2_aug = self.normalize(img2_aug)
                    
                    ret["img1_aug"] = img1_aug
                    ret["img2_aug"] = img2_aug 
                
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                
                img1 = self.normalize(img1)
                img2 = self.normalize(img2)
                
                label1 = self.target_transform(label1)
                label2 = self.target_transform(label2)
                
                ret["img1"] = img1
                ret["img2"] = img2
                ret["label1"] = label1
                ret["label2"] = label2
        
        index = torch.tensor(index, dtype=torch.int64)
        ret["index"] = index
        
        return ret