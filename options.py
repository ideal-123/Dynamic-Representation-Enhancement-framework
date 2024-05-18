import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='few-shot script')
    parser.add_argument('--dataset', default='miniImagenet',
                        help='miniImagenet/cub/cars/places/plantae/CropDiseases/EuroSAT/ISIC/chestX')
    parser.add_argument('--testset', default='miniImagenet',
                        help='miniImagenet/cub/cars/places/plantae/CropDiseases/EuroSAT/ISIC/chestX')
    parser.add_argument('--model', default='ResNet10', help='model: ResNet{10|18|34}')
    parser.add_argument('--method', default='GNN', help='MatchingNet/RelationNet/RelationNetLRP/ProtoNet/GNN'
                                                        '/GNNLRP/TPN')
    parser.add_argument('--train_n_way', default=5, type=int, help='class num to classify for training')
    parser.add_argument('--test_n_way', default=5, type=int, help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--train_aug', action='store_true', help='perform data augmentation or not during training ')
    parser.add_argument('--name', default='tmp', type=str, help='ProtoNet/GNN，训练模型后保存的checkpoints文件夹名称')
    parser.add_argument('--save_dir', default='output', type=str, help='')
    parser.add_argument('--data_dir', default='filelists', type=str, help='')
    parser.add_argument('--save_freq', default=10, type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
    parser.add_argument('--stop_epoch', default=500, type=int, help='Stopping epoch')
    parser.add_argument('--resume_epoch', default=0, type=int, help='')
    # Pretrain
    parser.add_argument('--num_classes', default=200, type=int, help='total number of classes in softmax')
    parser.add_argument('--pretrain', default=True, type=bool, help='')
    # Train
    parser.add_argument('--resume_dir', default='Pretrain', type=str,
                        help='continue from previous trained model with largest epoch')
    parser.add_argument('--pretrain_epoch', default=399, type=int, help='')
    parser.add_argument('--max_lr', default=80., type=float, help='max_lr')
    parser.add_argument('--T_max', default=5, type=int, help='')
    parser.add_argument('--lamb', default=1., type=float, help='alpha')
    parser.add_argument('--prob', default=0.5, type=float, help='probability of using original Images')
    parser.add_argument('--val_episode', default=600, type=int, help='')
    # Finetune
    parser.add_argument('--finetune_epoch', default=30, type=int, help='')
    parser.add_argument('--metric_ft', default=True, type=bool, help='')
    # cam
    parser.add_argument('--cam', default=False, type=bool, help='')
    parser.add_argument('--cam_layer', default=3, type=int, help='看该层输出的特征图')
    parser.add_argument('--grad_cam', default=False, type=bool, help='决定是否使用常规的grad_cam计算方式')
    # Test
    # parser.add_argument('--TestModelName', default='GNN_mask_block', help='')
    parser.add_argument('--TestModelName', default='GNN_mask_lifted_600', help='')
    parser.add_argument('--n_episode', default=600, type=int, help='')
    # RSC
    parser.add_argument('--rsc', default=False, type=bool, help='')
    parser.add_argument('--mask_epoch', default=250, type=int, help='')
    parser.add_argument('--mask_rate', default=1.0, type=float, help='')
    # parser.add_argument('--rate', default=0.4, type=float, help='')
    parser.add_argument('--rate', default=1, type=float, help='')
    parser.add_argument('--layer', default=0, type=int, help='block:0|1|2_to_be_spatial_attention')
    # parser.add_argument('--gflow', default=False, type=bool, help='')
    parser.add_argument('--gflow', default=False, type=bool, help='')
    parser.add_argument('--lifted_struct_loss', default=False, type=bool, help='')
    parser.add_argument('--iter_num', default=2000, type=int, help='')
    # tc
    parser.add_argument('--batch_num', default=100, type=int, help='')
    parser.add_argument('--mask_all', default=False, type=bool, help='')
    parser.add_argument('--lifted_th', default=0.5, type=float, help='')
    # parallel calculate
    parser.add_argument('--parallel', default=False, type=bool, help='')
    # visual_grad
    parser.add_argument('--grad_choice', default=0, type=int, help='0/1/2/3')
    parser.add_argument('--s1_rate', default=1.0, type=float, help='表示con损失在1-shot下的比重')
    return parser.parse_args()
