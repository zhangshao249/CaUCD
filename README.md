# CaUCD

## 1 download datasets
Download the dataset from the following link: https://pan.baidu.com/s/1PVqiOA0cenyyXUJHr9v0hw?pwd=twfb. After the download is complete, place the two downloaded compressed packages into **data** directory and extract them into **data** directory.

## 2 download pre-train weights
Download the weight files from the following link: https://pan.baidu.com/s/1-KbVLS7ZnLbWdrPkKgsK3A?pwd=dqgi. After the download is complete, place the **SECOND** folder and the **SYSU-CD** folder into the **weights** directory, and put **vitb14_reg4_pretrain.pth** into the **dinov2** folder.
from: 

## 3 train
python train_net.py --dataset SECOND (or SYSU-CD) --gpu="0,1,2,3" (or "0") --batch-size=8 --epoch=5

## 4 val
python test.py --dataset SECOND (or SYSU-CD) --gpu="0,1,2,3" (or "0") --batch-size=8

## (optional) 5 train mediator
To train on other change detection datasets, first use **train_mediator.py** to generate the **modular.npy** file, then train the change detection model using the **train_net.py** script.
python train_mediator.py --dataset "YOUR DATASET" --gpu "0"