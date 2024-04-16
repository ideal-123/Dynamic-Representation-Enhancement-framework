import numpy as np
import torch
import torch.optim
import os

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet, RelationNetLRP
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet, GnnNetLRP
from methods.tpn import TPN
from options import parse_args
from methods.Spatial_attention import SpatialAttention

# tensorboard记录
from torch.utils.tensorboard import SummaryWriter


# 实例化该类时确定保存数据的位置 log_dir(默认：./run/时间--主机名)
# class SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params, writer):
    # 优化器中包含在MetaTemplate中自定义的全连接层参数
    # if params.layer == 0:
    #     SpatialAtt = SpatialAttention(in_channels=64, kernel_size=9).cuda()
    # elif params.layer == 1:
    #     SpatialAtt = SpatialAttention(in_channels=128, kernel_size=9).cuda()
    # elif params.layer == 2:
    #     SpatialAtt = SpatialAttention(in_channels=256, kernel_size=9).cuda()

    # optim2 = torch.optim.Adam(SpatialAtt.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_acc = 0.
    total_it = 0
    epoch_num = stop_epoch - start_epoch

    for epoch in range(start_epoch, stop_epoch):
        # 启用 batch normalization(模型使用每一批数据的均值和方差) 和 dropout(随机取一部分网络连接更新参数)
        model.train()
        total_it, writer = model.train_loop(epoch, base_loader, optimizer, total_it, params, epoch_num, writer=writer)
        # 不启用 batch normalization 和 dropout
        model.eval()
        finetune_choice = params.finetune_choice
        finetune_epoch = params.finetune_epoch
        # if finetune_choice:
        #     # 对训练中的测试结果做adapt
        #     acc = model.test_loop(val_loader, SpatialAtt, adapt_epoch=finetune_epoch, adapt=True)
        # else:
        with torch.no_grad():
            acc, writer = model.test_loop(val_loader, params=params, writer=writer, epoch=epoch)

        if acc > max_acc:
            print("Best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            # if params.spatial_low:
            # outfile2 = os.path.join(params.checkpoint_dir, 'best_modelSpatial.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            # torch.save({'epoch': epoch, 'state': SpatialAtt.state_dict()}, outfile2)
        else:
            print("GG! Best accuracy {:f}".format(max_acc))

        if ((epoch + 1) % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            # outfile2 = os.path.join(params.checkpoint_dir, '{:d}Spatial.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            # torch.save({'epoch': epoch, 'state': SpatialAtt.state_dict()}, outfile2)

    return model, writer


def load_weight(state, model_params):
    for k, v in state.items():
        if 'alpha' not in k:
            if k in model_params.keys():
                if model_params[k].size() == v.size():
                    model_params[k] = v
                else:
                    print("diff size: ", k, v.size())
            else:
                print("diff key: ", k)
    return model_params


# --- main function ---
if __name__ == '__main__':

    # set numpy random seed
    np.random.seed(10)

    # parser argument
    params = parse_args()
    print('--- Training ---\n')
    print(params)

    # output and tensorboard dir
    params.checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- Prepare dataloader ---')
    print('\ttrain with seen domain {}'.format(params.dataset))
    print('\tval with seen domain {}'.format(params.dataset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file = os.path.join(params.data_dir, params.dataset, 'val.json')

    # model
    image_size = 224
    # n_query = 16
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
    # n_query = max(1, int(5 * params.test_n_way / params.train_n_way))
    base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.train_n_way, n_support=params.n_shot,
                                  n_eposide=params.batch_num)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot,
                                 n_eposide=params.val_episode)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    # params.model默认为resnet18
    if params.method == 'MatchingNet':
        model = MatchingNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'RelationNet':
        model = RelationNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'RelationNetLRP':
        model = RelationNetLRP(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'ProtoNet':
        model = ProtoNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'GNN':
        model = GnnNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'GNNLRP':
        model = GnnNetLRP(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'TPN':
        model = TPN(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    else:
        print("Please specify the method!")
        assert (False)

    rsc = params.rsc
    # 修改backbone中的RSC
    model.feature.RSC = rsc
    # 修改Meta_template中的RSC
    model.rsc = rsc

    if params.lifted_struct_loss2:
        model.loss_fn = model.loss_fn2

    if params.lifted_struct_loss:
        model.loss_fn = model.loss_fn1
        if params.method == 'TPN':
            model.loss_fn.rate = 1e-11

    # load model
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    pretrain = params.pretrain
    if params.resume_epoch > 0:
        resume_file = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(params.resume_epoch))
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch'] + 1
        model.load_state_dict(tmp['state'])
        print('\tResume the training weight at {} epoch.'.format(start_epoch))
    elif pretrain:
        print('pretrained model')
        print(type(pretrain))
        path = '%s/checkpoints/%s/399.tar' % (params.save_dir, params.resume_dir)
        # path = 'output/checkpoints/ProtoNet_block1_2_3_4/99.tar'
        state = torch.load(path)['state']
        model_params = model.state_dict()
        model_params = load_weight(state, model_params)
        model.load_state_dict(model_params)

    # training
    print('\n--- start the training ---')
    # 需要手动修改
    writer = SummaryWriter('tensorboard_log/GNN_ml_1s')
    model, writer = train(base_loader, val_loader, model, start_epoch, stop_epoch, params, writer)
    writer.close()
