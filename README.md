# Dynamic-Representation-Enhancement-framework
The source code for the paper "learning general features to bridge the cross-domain gaps in few-shot learning."

### Note
* This code is referenced [ATA](https://github.com/Haoqing-Wang/CDFSL-ATA) and [FWT](https://github.com/hytseng0509/CrossDomainFewShot)
* The dataset, model, and code are for non-commercial research purposes only.

### Prerequisites
* Python >= 3.5
* Pytorch >= 1.2.0
* You can use the requirements.txt file we provide to setup the environment via Anaconda.
    conda create --name py38 python=3.8
  conda install pytorch torchvision -c pytorch
  pip3 install -r requirements.txt        
### Datasets
Refers to CDFSL-ATA [(https://github.com/Haoqing-Wang/CDFSL-ATA)](https://github.com/Haoqing-Wang/CDFSL-ATA)

### Train
**1. Train the baseline.**

    python train.py --model ResNet10 --method GNN --n_shot 1 --name baseline_1s
    python train.py --model ResNet10 --method GNN --n_shot 5 --name baseline_5s
    

**2. Train the model with the proposed DRE framework.**

    python train.py --model ResNet10_mask --method GNN --name GNN_ml_1s --n_shot 1 --rsc True --lifted_struct_loss True
    python train.py --model ResNet10_mask --method GNN --name GNN_ml_5s --n_shot 5 --rsc True --lifted_struct_loss True    
### Fine-tuning
    
python finetune_ml_partial.py --dataset cub --name GNN_ml_1s --finetune_epoch 30 --n_shot 1
python finetune_ml_partial.py --dataset cub --name GNN_ml_5s --finetune_epoch 50 --n_shot 5    
