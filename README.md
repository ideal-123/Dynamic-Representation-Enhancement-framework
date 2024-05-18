# Dynamic-Representation-Enhancement-framework
The source code for the paper "learning general features to bridge the cross-domain gaps in few-shot learning."


This code is referenced [ATA](https://github.com/Haoqing-Wang/CDFSL-ATA) and [FWT](https://github.com/hytseng0509/CrossDomainFewShot)

### Prerequisites
* Python >= 3.5
* Pytorch >= 1.2.0
* You can use the requirements.txt file we provide to setup the environment via Anaconda.
 ``
 conda create --name py38 python=3.8
 conda install pytorch torchvision -c pytorch
 pip3 install -r requirements.txt
 ``
### Datasets
Refers to CDFSL-ATA [(https://github.com/Haoqing-Wang/CDFSL-ATA)](https://github.com/Haoqing-Wang/CDFSL-ATA)

### Train
**1. Train the baseline.**
``
python train_ml.py --model ResNet10 --method GNN --n_shot 5 --name baseline
``
**2. Train the model with the proposed DRE framework.**
``
python train.py --model ResNet10_mask --method GNN --rsc true --mask_rate 0.98 --mask_epoch 250 --name GNN_ml_1s_1 --n_shot 1 --lifted_struct_loss True >output/ckpt2/231226_GNN_ml_1s_1.log &;
``
