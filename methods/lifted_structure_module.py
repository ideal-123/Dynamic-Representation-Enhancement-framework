import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Callable, Optional


# labels:(num, 1) embeddings:(num, n)
def lifted_structure_loss(embeddings, labels, z_proto=None, proto_based_calculate=False, margin=1.0, alpha=10.0):
    # 计算所有向量之间的内积，即相似度矩阵 (num, num)
    num = embeddings.size(0)
    n = embeddings.size(1)
    # (num,n) --- (num,n,num)
    a = embeddings.unsqueeze(2).expand(num, n, num)
    # (n,num) --- (num,n,num)
    b = embeddings.t().unsqueeze(0).expand(num, n, num)
    # (25,25)
    # 晚点关于512维求和 later_w
    later_w = True
    if later_w:
        # 基于proto_Net的提升策略 x 效果没有原Lifted_struct_loss效果好
        if proto_based_calculate:
            # z_proto:(5,512)
            # a: (25,512,5)
            a = embeddings.unsqueeze(2).expand(num, n, z_proto.shape[0])
            b = z_proto.t().unsqueeze(0).expand(num, n, z_proto.shape[0])
            # (25,512,5)
            sim_mat = torch.pow(a - b, 2)
            # (25,512,5)
            pos_mask = labels.unsqueeze(1).repeat(1, z_proto.shape[0]).eq(torch.FloatTensor([0, 1, 2, 3, 4]).expand(num,
                                                                                                                    z_proto.shape[
                                                                                                                        0])).unsqueeze(
                1).expand([labels.shape[0], embeddings.shape[1], z_proto.shape[0]]).cuda()
            neg_mask = labels.unsqueeze(1).repeat(1, z_proto.shape[0]).ne(torch.FloatTensor([0, 1, 2, 3, 4]).expand(num,
                                                                                                                    z_proto.shape[
                                                                                                                        0])).unsqueeze(
                1).expand([labels.shape[0], embeddings.shape[1], z_proto.shape[0]]).cuda()
            # (25,512,1) 直接计算最难的负proto，而不是三元组
            hardest_neg = (torch.exp(alpha - sim_mat) * neg_mask).sum(2).unsqueeze(2).expand(
                [labels.shape[0], embeddings.shape[1], z_proto.shape[0]])
            # (25,512,5)
            hardest = torch.log(hardest_neg) + sim_mat * pos_mask
        else:
            # (25,512,25)
            sim_mat = torch.pow(a - b, 2)
            # 构建同类样本和异类样本的掩码 同类样本编码中对角线为False
            pos_mask = labels.expand(labels.shape[0], labels.shape[0]).eq(
                labels.expand(labels.shape[0], labels.shape[0]).t())
            diag_indices = torch.arange(pos_mask.size(0))
            pos_mask[diag_indices, diag_indices] = False
            pos_mask = pos_mask.unsqueeze(1).expand([labels.shape[0], embeddings.shape[1], labels.shape[0]]).cuda()

            neg_mask = labels.expand(labels.shape[0], labels.shape[0]).ne(
                labels.expand(labels.shape[0], labels.shape[0]).t())
            neg_mask = neg_mask.unsqueeze(1).expand([labels.shape[0], embeddings.shape[1], labels.shape[0]]).cuda()
            # [25,512,1]
            hardest_neg = (torch.exp(alpha - sim_mat) * neg_mask).sum(2).unsqueeze(2)
            # [25,512,25] 除了ij属于正例的位置外，其余位置元素都为0
            hardest = torch.log(hardest_neg + torch.transpose(hardest_neg, 0, 2)) * pos_mask + sim_mat * pos_mask

            # # 计算同类样本之间的相似度平均值 (25,512,4)
            # pos_sim = sim_mat[pos_mask].view(sim_mat.shape[0], sim_mat.shape[1], 4)
            # # 计算异类样本之间的相似度平均值 ([25,512,20])
            # neg_sim = sim_mat[neg_mask].view(sim_mat.shape[0], sim_mat.shape[1], 20)
            # # 到最难负样本的距离([25,512,1])
            # hardest_neg = torch.exp(alpha - neg_sim).sum(2).unsqueeze(2)
            # pos_dis = True  # 控制损失函数中是否加入正例距离
            # if pos_dis:
            #     # (25,512,4)
            #     hardest = torch.log(hardest_neg + torch.transpose(hardest_neg, 0, 2))[pos_mask].view(sim_mat.shape[0], sim_mat.shape[1], 4)+ pos_sim
            # else:
            #     hardest = torch.log(hardest_neg + hardest_neg.t())[pos_mask]
            # 各个维度上的损失

        loss = F.relu(hardest).sum(2).sum(0) / (hardest.shape[0] * 4)
        loss = loss.sum()
    else:
        sim_mat = torch.pow(a - b, 2).sum(1)

        # 构建同类样本和异类样本的掩码 同类样本编码中对角线为False
        pos_mask = labels.expand(labels.shape[0], labels.shape[0]).eq(
            labels.expand(labels.shape[0], labels.shape[0]).t())
        diag_indices = torch.arange(pos_mask.size(0))
        pos_mask[diag_indices, diag_indices] = False

        neg_mask = labels.expand(labels.shape[0], labels.shape[0]).ne(
            labels.expand(labels.shape[0], labels.shape[0]).t())
        # 计算同类样本之间的相似度平均值 (25,4)
        pos_sim = sim_mat[pos_mask]
        # 计算异类样本之间的相似度平均值 ([25,20])
        neg_sim = sim_mat[neg_mask].view(labels.shape[0], -1)
        # 到最难负样本的距离([25,1])
        hardest_neg = torch.exp(alpha - neg_sim).sum(1).unsqueeze(1)
        # ([100])
        pos_dis = True  # 控制损失函数中是否加入正例距离
        if pos_dis:
            hardest = torch.log(hardest_neg + hardest_neg.t())[pos_mask] + pos_sim
        else:
            hardest = torch.log(hardest_neg + hardest_neg.t())[pos_mask]
        loss = F.relu(hardest).sum() / hardest.shape[0]

    return loss


from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from sklearn.preprocessing import MinMaxScaler


def lifted_structure_loss2(scores, labels, rate, s1_rate=1.0):
    # scores(80,5)等价与sim_mat 为query到各类别之间的相似度
    # dui score guiyihua
    # if rate < 1:
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scores_normalized = scaler.fit_transform(scores.cpu().numpy().T).T
    # scores = scores_normalized

    if rate < 1:
        scores_normalized = scores / scores.detach().sum(1).unsqueeze(1).repeat(1, 5)
        scores = scores_normalized

    num = labels.shape[0]
    n_way = scores.shape[1]
    dist = -scores
    # (80,5)向量，其中元素为False或True
    if len(labels.shape) > 1:  # RelationNet
        labels = torch.from_numpy(np.repeat(range(n_way), int(num / n_way))).cuda()
    pos_mask = labels.unsqueeze(1).repeat(1, n_way).eq(torch.FloatTensor([0, 1, 2, 3, 4]).expand(num, n_way).cuda())
    neg_mask = labels.unsqueeze(1).repeat(1, n_way).ne(torch.FloatTensor([0, 1, 2, 3, 4]).expand(num, n_way).cuda())
    # alpha为各类别上所有正例距离的均值 (n_way,1)
    alpha = torch.mean(pos_mask * dist, dim=0)
    a = alpha.unsqueeze(0).repeat(num, 1)
    # 得到(80,1)向量，为样本到各类别的hardest distance
    hardest_neg = (torch.exp(a - dist) * neg_mask).sum(1).unsqueeze(1)
    # 考虑不同样本是否需要施加不一样的权重
    loss = F.relu(hardest_neg).sum() / (num * (n_way - 1))
    loss = loss*s1_rate
    return loss


def InfoNCE(scores, labels, temprature):
    # scores(80,5)等价与sim_mat 为query到各类别之间的相似度
    # dui score guiyihua
    # if rate < 1:
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scores_normalized = scaler.fit_transform(scores.cpu().numpy().T).T
    # scores = scores_normalized
    P = F.softmax(scores / temprature, dim=1)

    if rate < 1:
        scores_normalized = scores / scores.detach().sum(1).unsqueeze(1).repeat(1, 5)
        scores = scores_normalized

    num = labels.shape[0]
    n_way = scores.shape[1]
    dist = -scores
    # (80,5)向量，其中元素为False或True
    if len(labels.shape) > 1:  # RelationNet
        labels = torch.from_numpy(np.repeat(range(n_way), int(num / n_way))).cuda()
    pos_mask = labels.unsqueeze(1).repeat(1, n_way).eq(torch.FloatTensor([0, 1, 2, 3, 4]).expand(num, n_way).cuda())
    neg_mask = labels.unsqueeze(1).repeat(1, n_way).ne(torch.FloatTensor([0, 1, 2, 3, 4]).expand(num, n_way).cuda())
    # alpha为各类别上所有正例距离的均值 (n_way,1)
    alpha = torch.mean(pos_mask * dist, dim=0)
    a = alpha.unsqueeze(0).repeat(num, 1)
    # 得到(80,1)向量，为样本到各类别的hardest distance
    hardest_neg = (torch.exp(a - dist) * neg_mask).sum(1).unsqueeze(1)
    # 考虑不同样本是否需要施加不一样的权重
    loss = F.relu(hardest_neg).sum() / (num * (n_way - 1))
    return loss


class Lifted_Struct_loss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float
    # th = 1
    rate = 1

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, rate=1.0, s1_rate=1.0) -> None:
        super(Lifted_Struct_loss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.rate = rate
        self.s1_rate=s1_rate

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss1 = lifted_structure_loss2(input, target, self.rate, s1_rate=self.s1_rate)
        # loss1 = lifted_structure_loss2(input, target, self.th)
        return self.rate * loss1 + F.cross_entropy(input, target, weight=self.weight,
                                                   ignore_index=self.ignore_index, reduction=self.reduction,
                                                   label_smoothing=self.label_smoothing)
