import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
from torch.autograd import Variable


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.method = ' ProtoNet'

        if self.rsc:
            print('子类可以继承父类的静态变量')

    def reset_modules(self):
        return

    def set_forward(self, x, is_feature=False, supp_query=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        # (5,5,512)
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        # (80,5,512)
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        if supp_query:
            # [n_data,n_dim]
            return z_support, z_query
        else:
            # z_query,z_proto为特征提取器得到的特征向量
            return z_query, z_proto, scores

    # 模型根据z_support和z_query计算对应的scores
    def scores_calculate(self, z_support, z_query):
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def get_distance(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        return euclidean_dist(z_proto, z_proto)[0, :5].cpu().numpy()

    def loss_calculate(self, scores):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        loss = self.loss_fn(scores, y_query)
        return loss

    def set_forward_loss(self, x, score_calculate = False, rsc=False):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        z_query, z_proto, scores = self.set_forward(x)
        if rsc:
            original_loss = self.loss_fn(scores, y_query)
            # print(original_loss)

            # 超参 -- challenge的向量数目
            percent1 = 1 / 20
            percent2 = 1 / 10
            self.eval()
            # 只记录query和proto的特征输出 并建立新变量放入网络
            z_query_new = z_query.clone().detach()
            z_proto_new = z_proto.clone().detach()
            z_proto_new = Variable(z_proto_new.data, requires_grad=True)
            z_query_new = Variable(z_query_new.data, requires_grad=True)
            dists = euclidean_dist(z_query_new, z_proto_new)
            # scores为 n*m
            scores = -dists
            # 模型的输出结果 scores
            class_num = scores.shape[1]
            # z_query 和 z_proto分别为N*D和M*D 对D做mask
            num_dimension = z_query_new.shape[1]
            # 待预测样本个数 n
            num_rois = z_query_new.shape[0]
            # 待分类类别个数m
            # num_channel = z_proto_new.shape[0]

            # one_hot = torch.zeros(1, dtype=torch.float32).cuda()
            # one_hot = Variable(one_hot, requires_grad=False)

            # 构造稀疏矩阵记录数据真实的标签信息--one_hot_sparse
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = y_query
            sp_v = torch.ones([num_rois])
            # torch.sparse.FloatTensor：以sp_i中的行和列作为位置坐标，以sp_v为具体的值，构造稀疏矩阵
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v,
                                                      torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)

            # scores和y的点积 n*m
            one_hot = torch.sum(scores * one_hot_sparse)
            self.zero_grad()

            # 关于该点积梯度反传
            one_hot.backward()
            # n*D
            # 80*512
            grads_val_query = z_query_new.grad.clone().detach()
            # m*D
            # 5*512
            # 关于所有的query和support proto求均值 得1*D向量
            # 由于使用梯度衡量各个维度对最终结果的影响,所以取绝对值后做mask
            grad_mean = torch.mean(grads_val_query, 0)

            # per表示待查找的分位数
            vector_thresh_percent = round(num_dimension * percent1)
            vector_thresh_value = torch.sort(grad_mean, dim=0, descending=True)[0][vector_thresh_percent]
            # (512，1)
            vector = torch.where(grad_mean > vector_thresh_value,
                                 torch.zeros(grad_mean.shape).cuda(),
                                 torch.ones(grad_mean.shape).cuda())
            # feature dimension mask
            # 扩展为80*512
            mask_query = vector.expand(num_rois, num_dimension)
            # 5*512
            # mask_proto = vector.expand(num_channel, num_dimension)
            # mask_proto = Variable(mask_proto, requires_grad=True)

            # --- batch RSC --- 只对当前最容易分类的percent2的query做RSC
            # scores对每一行求和得sum 关于sum求score中各维度占比 得score_probability
            # score_probability 与 one_hot_sparse相乘后对各行求和得n*1向量,此时值最大的对应的样本为易分类样本
            scores_exp = torch.exp(scores)
            scores_sum = torch.sum(scores_exp, 1).expand(scores.shape[1], scores.shape[0])
            scores_sum = scores_sum.transpose(1, 0)
            scores_prob = scores_exp/scores_sum
            # n*1向量b 表示各个样本被正确分类的概率
            correct_prob = torch.sum(scores_prob * one_hot_sparse, 1)
            # 待查找的分位数
            vector_thresh_percent = round(num_rois * percent2)
            vector_thresh_value = torch.sort(correct_prob, dim=0, descending=True)[0][vector_thresh_percent]
            # (n，1) 需要加mask的样本对应vector中的值为0
            vector = torch.where(correct_prob > vector_thresh_value,
                                 torch.zeros(correct_prob.shape).cuda(),
                                 torch.ones(correct_prob.shape).cuda())
            # nonzero()返回包含所有非零元素索引的张量 其中非0的
            ignore_index = vector.nonzero()
            # 对需要忽略的样本所在行置为1
            mask_query_new = mask_query.clone()
            for i in range(len(ignore_index)):
                k = ignore_index[i]
                mask_query_new[k, :] = 1
            # 对mask_query的变动应该在mask_query放入网络之前
            mask_query = mask_query_new
            mask_query = Variable(mask_query, requires_grad=True)
            # 80*5
            dists = euclidean_dist(z_query_new * mask_query, z_proto_new)
            # scores为 n*m
            scores = -dists
            loss = self.loss_fn(scores, y_query)
            self.train()
        elif score_calculate:
            loss = scores * y_query.unsqueeze(1).expand(scores.size(0),scores.size(1))
        else:
            loss = self.loss_fn(scores, y_query)
        return scores, loss


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # 得到的dist为N乘M向量
    return torch.pow(x - y, 2).sum(2)
