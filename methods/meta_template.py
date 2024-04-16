import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from PIL import Image
import cv2
import math
import os
import utils
from torchvision import transforms
from methods.lifted_structure_module import Lifted_Struct_loss
# 打乱测试中testset的顺序，以防学习到标签规律


class MetaTemplate(nn.Module):
    rsc = False
    Spatial_low = False
    low_feature = False
    low_embedding = None

    mask = False
    mask_channel = None

    shuffle = False
    idx = None

    def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1
        self.gradients = []

        # self.feature确定特征提取网络model_func 默认的配置到其它网络中的backbone为Resnet18
        self.feature = model_func(flatten=flatten, leakyrelu=leakyrelu)
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way

        self.loss_fn2 = Lifted_Struct_loss(rate=2)
        self.loss_fn1 = Lifted_Struct_loss()

    @abstractmethod
    def set_forward(self, x, is_feature=False):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    @abstractmethod
    def loss_calculate(self, x):
        pass

    @abstractmethod
    def scores_calculate(self, x):
        pass

    def forward(self, x):
        out = self.feature(x)
        return out

    def parse_feature(self, x, is_feature):
        x = x.cuda()
        if is_feature:
            z_all = x
        elif self.low_feature:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            # # ([105,512]) 与添加的module无关
            z_all = self.feature.forward(x)
            z_all = torch.cat((z_all, self.low_embedding), dim=1)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        elif self.mask:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x) * self.mask_channel.unsqueeze(0).repeat(105, 1).cuda()
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        shuffle = False
        if shuffle:
            idx = self.idx
            z_query_ = z_query.reshape(80, -1)[idx]
            z_query = z_query_.view(self.n_way, self.n_query, -1)
        # self.gradients = self.feature.gradients
        return z_support, z_query

    def correct(self, x):

        scores, loss = self.set_forward_loss(x)
        y_query = np.repeat(range(self.n_way), self.n_query)
        num_query = len(y_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), num_query, loss.item() * num_query

    def train_loop(self, epoch, train_loader, optimizer, total_it, params, epoch_num, writer=None):
        self.rsc = params.rsc
        self.lrptraining = True
        print_freq = len(train_loader) // 10
        avg_loss = 0
        acc_all = []

        # batch维度最大丢弃1/10
        self.feature.train_ = True

        for i, (x, class_l) in enumerate(train_loader):
            # 置空记录梯度的列表
            self.feature.gradients = []
            # x = torch.ones([5, 21, 3, 224, 224])
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)

            # loss关于weight的导数变为0 每个batch都需要进行一次清零操作，防止梯度累积
            optimizer.zero_grad()

            if params.rsc:
                if epoch < params.mask_epoch:
                    per = params.mask_rate/params.mask_epoch * epoch
                else:
                    per = 0.98

                # 每次都计算无mask时对训练数据的梯度作为mask依据
                # 需要hook函数记录梯度时设置为True
                self.feature.RSC = True
                self.feature.train_ = True
                _, original_loss = self.set_forward_loss(x)
                original_loss.backward()
                if i == 0:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i)
                else:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i, grad1, grad2, grad3, grad4)
                self.feature.mask_(epoch)
                self.feature.train_ = False
                optimizer.zero_grad()
                # del self.feature.gradients
                # torch.cuda.empty_cache()

            correct_this, count_this, loss_this = self.correct(x)
            _, loss = self.set_forward_loss(x)
            acc_all.append(correct_this / count_this * 100)
            # 先做adapter 再优化meta-parameter
            loss.backward(retain_graph=False)
            optimizer.step()
            # 输出的为训练损失
            avg_loss = avg_loss + loss.item()

            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader),
                                                                        avg_loss / float(i + 1)))
            if params.rsc:
                self.feature.mask_rec()

            total_it += 1
            torch.cuda.empty_cache()

        if params.rsc:
            self.feature.generate_mask(grad1, grad2, grad3, grad4, per)

        # 记录在当前的mask网络中得到的模型准确率
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        iter_num = len(train_loader)
        print('--- %d Train Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if self.rsc:
            print('mask_num:per1 {:4.2f}'.format(per))

        writer.add_scalar('Loss/Train', avg_loss / float(i + 1), epoch)
        writer.add_scalar('Accuracy/Train', acc_mean / float(i + 1), epoch)

        del loss, avg_loss
        torch.cuda.empty_cache()

        return total_it, writer

    def train_loop_visual_grad(self, epoch, train_loader, optimizer, total_it, params, epoch_num, writer=None):
        self.rsc = params.rsc
        self.lrptraining = True
        print_freq = len(train_loader) // 10
        avg_loss = 0
        acc_all = []
        visual_gradients = []

        # batch维度最大丢弃1/10
        self.feature.train_ = True

        for i, (x, class_l) in enumerate(train_loader):
            # 置空记录梯度的列表
            self.feature.gradients = []
            # x = torch.ones([5, 21, 3, 224, 224])
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)

            # loss关于weight的导数变为0 每个batch都需要进行一次清零操作，防止梯度累积
            optimizer.zero_grad()
            if params.rsc:
                if epoch < params.mask_epoch:
                    per = params.mask_rate/params.mask_epoch * epoch
                else:
                    per = 0.98

                # 每次都计算无mask时对训练数据的梯度作为mask依据
                # 需要hook函数记录梯度时设置为True
                self.feature.RSC = True
                self.feature.train_ = True
                _, original_loss = self.set_forward_loss(x)
                original_loss.backward()
                if i == 0:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i)
                else:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i, grad1, grad2, grad3, grad4)
                self.feature.mask_(epoch)
                self.feature.train_ = False
                optimizer.zero_grad()
                # del self.feature.gradients
                # torch.cuda.empty_cache()

            correct_this, count_this, loss_this = self.correct(x)
            _, loss = self.set_forward_loss(x)
            acc_all.append(correct_this / count_this * 100)
            # 先做adapter 再优化meta-parameter
            loss.backward(retain_graph=False)
            optimizer.step()
            # 输出的为训练损失
            avg_loss = avg_loss + loss.item()

            # 该列表记录每个iteration的各个参数
            gradients_ = []
            for name, param in self.named_parameters():
                if param.grad is not None:
                    tmp = np.abs(param.grad.data.cpu().numpy())
                    gradients = tmp.reshape(tmp.shape[0],-1).mean(1)
                    gradients_.append(gradients)
            # 记录当前epoch的所有iterations
            visual_gradients.append(gradients_[:36])
            del gradients_,tmp

            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader),
                                                                        avg_loss / float(i + 1)))
            if params.rsc:
                self.feature.mask_rec()

            total_it += 1
            torch.cuda.empty_cache()

        if params.rsc:
            self.feature.generate_mask(grad1, grad2, grad3, grad4, per)

        # 记录在当前的mask网络中得到的模型准确率
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        iter_num = len(train_loader)
        print('--- %d Train Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if self.rsc:
            print('mask_num:per1 {:4.2f}'.format(per))

        writer.add_scalar('Loss/Train', avg_loss / float(i + 1), epoch)
        writer.add_scalar('Accuracy/Train', acc_mean / float(i + 1), epoch)

        del loss, avg_loss
        torch.cuda.empty_cache()

        return total_it, writer, visual_gradients

    def train_loop_visual_grad1(self, epoch, train_loader, optimizer, total_it, params, epoch_num, writer=None):
        self.rsc = params.rsc
        self.lrptraining = True
        print_freq = len(train_loader) // 10
        avg_loss = 0
        acc_all = []

        # batch维度最大丢弃1/10
        self.feature.train_ = True

        for i, (x, class_l) in enumerate(train_loader):
            # 置空记录梯度的列表
            self.feature.gradients = []
            # x = torch.ones([5, 21, 3, 224, 224])
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)

            # loss关于weight的导数变为0 每个batch都需要进行一次清零操作，防止梯度累积
            optimizer.zero_grad()
            if params.rsc:
                if epoch < params.mask_epoch:
                    per = params.mask_rate / params.mask_epoch * epoch
                else:
                    per = 0.98

                # 每次都计算无mask时对训练数据的梯度作为mask依据
                # 需要hook函数记录梯度时设置为True
                self.feature.RSC = True
                self.feature.train_ = True
                _, original_loss = self.set_forward_loss(x)
                original_loss.backward()
                if i == 0:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i)
                else:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i, grad1, grad2, grad3, grad4)
                self.feature.mask_(epoch)
                self.feature.train_ = False
                optimizer.zero_grad()
                # del self.feature.gradients
                # torch.cuda.empty_cache()

            correct_this, count_this, loss_this = self.correct(x)
            _, loss = self.set_forward_loss(x)
            acc_all.append(correct_this / count_this * 100)
            # 先做adapter 再优化meta-parameter
            loss.backward(retain_graph=False)
            optimizer.step()
            # 输出的为训练损失
            avg_loss = avg_loss + loss.item()

            # 该列表记录每个iteration的各个参数
            id=0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    tmp = np.abs(param.grad.data.cpu().numpy())
                    tmp2 = param.grad.data.cpu().numpy()
                    if id==0:
                        gradients = tmp.reshape(tmp.shape[0], -1).mean(1)
                        gradients2 = tmp2.reshape(tmp2.shape[0], -1).mean(1)
                        id = id + 1
                    elif id < 36:
                        gradients = np.concatenate((gradients,tmp.reshape(tmp.shape[0], -1).mean(1)),axis=0)
                        gradients2 = np.concatenate((gradients2,tmp2.reshape(tmp2.shape[0], -1).mean(1)),axis=0)
                        id = id + 1
                    else:
                        break
            # 当前iteration的平均梯度
            visual_grad_iter = gradients.mean()
            visual_grad_iter2 = gradients2.mean()
            del gradients, tmp

            writer.add_scalar('Avg_GradNorm/Iteration', visual_grad_iter, int(epoch*len(train_loader)+i))
            writer.add_scalar('Avg_Grad/Iteration', visual_grad_iter2, int(epoch * len(train_loader) + i))

            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader),
                                                                        avg_loss / float(i + 1)))
            if params.rsc:
                self.feature.mask_rec()

            total_it += 1
            torch.cuda.empty_cache()

        if params.rsc:
            self.feature.generate_mask(grad1, grad2, grad3, grad4, per)

        # 记录在当前的mask网络中得到的模型准确率
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        iter_num = len(train_loader)
        print('--- %d Train Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if self.rsc:
            print('mask_num:per1 {:4.2f}'.format(per))

        del loss, avg_loss
        torch.cuda.empty_cache()

        return total_it, writer

    def train_loop_feature_grad(self, epoch, train_loader, optimizer, total_it, params, epoch_num, writer=None):
        self.rsc = params.rsc
        self.lrptraining = True
        print_freq = len(train_loader) // 10
        avg_loss = 0
        acc_all = []
        visual_feature_grad = []
        visual_feature_absgrad = []
        # batch维度最大丢弃1/10
        self.feature.train_ = True

        for i, (x, class_l) in enumerate(train_loader):
            # 置空记录梯度的列表
            self.feature.gradients = []
            # x = torch.ones([5, 21, 3, 224, 224])
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)

            # loss关于weight的导数变为0 每个batch都需要进行一次清零操作，防止梯度累积
            optimizer.zero_grad()

            if params.rsc:
                if epoch < params.mask_epoch:
                    per = params.mask_rate/params.mask_epoch * epoch
                else:
                    per = 0.98

                # 每次都计算无mask时对训练数据的梯度作为mask依据
                # 需要hook函数记录梯度时设置为True
                self.feature.RSC = True
                self.feature.train_ = True

                _, original_loss = self.set_forward_loss(x)
                original_loss.backward()
                if i == 0:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i)
                else:
                    grad1, grad2, grad3, grad4 = self.feature.generate_grad(i, grad1, grad2, grad3, grad4)
                self.feature.mask_(epoch)
                self.feature.train_ = False
                self.feature.train1_ = True
                # del self.feature.gradients
                # torch.cuda.empty_cache()


            correct_this, count_this, loss_this = self.correct(x)
            _, loss = self.set_forward_loss(x)
            acc_all.append(correct_this / count_this * 100)
            # 先做adapter 再优化meta-parameter
            loss.backward(retain_graph=False)
            optimizer.step()
            # 输出的为训练损失
            avg_loss = avg_loss + loss.item()
            # visual_grad
            tmp = self.feature.gradients[0].cpu().numpy().mean(0)
            gradients = tmp.reshape((tmp.shape[0],-1)).mean(1)
            visual_feature_grad.append(gradients)
            tmp1 = np.abs(self.feature.gradients[0].cpu().numpy()).mean(0)
            gradients1 = tmp1.reshape((tmp1.shape[0], -1)).mean(1)
            visual_feature_absgrad.append(gradients1)
            # 不影响original_loss中各层梯度的计算
            self.feature.train1_ = False
            del self.feature.gradients,tmp,tmp1,gradients,gradients1

            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader),
                                                                        avg_loss / float(i + 1)))
            if params.rsc:
                self.feature.mask_rec()

            total_it += 1
            torch.cuda.empty_cache()

        if params.rsc:
            self.feature.generate_mask(grad1, grad2, grad3, grad4, per)

        # 记录在当前的mask网络中得到的模型准确率
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        iter_num = len(train_loader)
        print('--- %d Train Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if self.rsc:
            print('mask_num:per1 {:4.2f}'.format(per))

        del loss, avg_loss
        torch.cuda.empty_cache()

        return total_it, writer, visual_feature_grad, visual_feature_absgrad

    # lifted_adapter控制是否对当前任务进行关于lifted loss的自适应
    def test_loop(self, test_loader, adapt_epoch=1, adapt=False, params=None, writer=None, epoch=None):
        self.lrptraining = False
        loss = 0.
        count = 0
        acc_all = []
        self.feature.train_ = False

        iter_num = len(test_loader)
        for i, (x, label_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)

            if params.cam:
                a = self.cam_test(x, params, adapt_epoch=i)
            # print(label_) #21 + 21 + 21 + 21 + 21

            # task_adapter --- 关于Support Set上的损失函数最小化
            if adapt:
                a = {}
                for name, param in self.feature.trunk[7].named_parameters():
                    a[name] = param.detach().clone()
                # embeddings:(5,5,512)
                optim2 = torch.optim.Adam(self.feature.trunk[7].parameters(), lr=params.spatial_lr)
                for i in range(adapt_epoch):
                    if params.cam:
                        a = self.cam_test(x, params, adapt_epoch=i)
                    support_embedding, _ = self.set_forward(x, supp_query=True)
                    # # embeddings需要用于优化原始网络的参数，所以不再使用detach()
                    embeddings = support_embedding.view(self.n_way * self.n_support, -1)
                    labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
                    scores = self.scores_calculate(support_embedding, embeddings)
                    if params.method == 'RelationNet':
                        if self.loss_type == 'mse':
                            y_oh = utils.one_hot(labels, self.n_way)
                            y_oh = y_oh.cuda()
                            loss_adapt = self.loss_fn(scores, y_oh)
                        else:
                            y = labels.cuda()
                            loss_adapt = self.loss_fn(scores, y)
                    else:
                        y = labels.cuda()
                        loss_adapt = self.loss_fn(scores, y)
                    loss_adapt.backward()
                    optim2.step()
                    optim2.zero_grad()

            # spatial attention low level
            if self.Spatial_low:
                # print('测试spatial_low')
                self.feature.RSC = True  # 为了得到输出值output
                self.feature.train_ = False
                if params.shuffle:
                    self.shuffle = False
                _, original_loss = self.set_forward_loss(x)  # 正常前传，没有加入任何Spatial相关内容
                if params.shuffle:
                    self.shuffle = True
                    low_feat_ = low_feat.view(self.n_way, self.n_support + self.n_query, -1)
                    low_feat_query = low_feat_[:, self.n_support:]
                    idx = torch.randperm(80)
                    self.idx = idx
                    low_query_ = low_feat_query.reshape(80, 64, 56, 56)[idx]
                    low_feat_[:, self.n_support:] = low_query_.view(self.n_way, self.n_query, -1)
                    low_feat = low_feat_.reshape(105, 64, 56, 56)

            correct_this, count_this, loss_this = self.correct(x)

            # 将神经网络的参数还原到task_adapter之前，使下次迭代依然以meta参数为起始
            if adapt:
                for name, param in self.feature.trunk[7].named_parameters():
                    param.data = a[name]
            if self.Spatial_low:
                self.low_feature = False

            acc_all.append(correct_this / count_this * 100)
            loss += loss_this
            count += count_this

            del x, i
            torch.cuda.empty_cache()

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        # writer.add_scalar('Loss/Validation', loss/count, epoch)
        # writer.add_scalar('Accuracy/Validation', acc_mean, epoch)

        return acc_mean, writer

        # lifted_adapter控制是否对当前任务进行关于lifted loss的自适应
    def test_intuition(self, test_loader, params=None):
        self.lrptraining = False
        loss = 0.
        count = 0
        acc_all = []
        self.feature.train_ = False

        mean_loss = []
        mean_acc = []

        iter_num = len(test_loader)
        for k in range(512):
            Channel_loss = []
            Channel_acc = []
            a = torch.ones(512)
            a[k] = 0
            self.mask_channel = a
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)

                correct_this, count_this, loss_this = self.correct(x)
                self.mask = True
                correct_this1, count_this1, loss_this1 = self.correct(x)
                Channel_loss.append(loss_this1-loss_this)
                Channel_acc.append(correct_this1/count_this1*100-correct_this/count_this*100)
                self.mask = False

            mean_loss.append(np.mean(Channel_loss))
            mean_acc.append(np.mean(Channel_acc))

            acc_all.append(correct_this / count_this * 100)
            loss += loss_this
            count += count_this

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean, Channel_acc, Channel_loss

    # 对test_loader对应的数据集和当前model(self)查看特征图和对应的Grad_cam
    def cam_test(self, x, params, adapt_epoch=1):
        self.feature.train_ = True
        self.feature.RSC = True
        # 置空记录梯度的列表
        self.feature.gradients = []
        self.n_query = x.size(1) - self.n_support
        if self.change_way:
            self.n_way = x.size(0)

        _, original_loss = self.set_forward_loss(x)

        out = self.feature.conv_output
        # CAM 测试
        # protoNet
        # z_support, z_query = self.set_forward(x, supp_query=True)
        # # scores:([80,5])
        # scores = self.scores_calculate(z_support, z_query)
        # GNN scores.size():[80,5]
        scores = self.set_forward(x)
        one_hot_output = torch.zeros(scores.size())
        _, topk_labels = scores.data.topk(1, 1, True, True)
        for i in range(topk_labels.size(0)):
            one_hot_output[i, topk_labels[i]] = 1
        # one_hot_output.size(80,5)
        one_hot_output = one_hot_output.to(device='cuda:0')
        scores.backward(gradient=one_hot_output, retain_graph=True)

        # 待提取的特征图是网络中的第几层，实际上是第几个block模块
        layer = params.cam_layer
        k = int(3 - layer)
        grad = self.feature.gradients[k]
        # grad_cam决定得到一般的关于grad的加权特征图(True)还是保存各个通道的特征图(False)
        # grad_cam = params.grad_cam

        # 在output/cam_test下建立对应模型的文件夹
        folder_name = './output/cam_test/' + params.TestModelName
        if not os.path.exists(folder_name):  # os模块判断并创建
            os.mkdir(folder_name)
        folder_name = folder_name + '/' + params.testset
        if not os.path.exists(folder_name):  # os模块判断并创建
            os.mkdir(folder_name)
        grad_cam = True
        if grad_cam:
            dir_name = folder_name + '/grad_cam'
            if not os.path.exists(dir_name):  # os模块判断并创建
                os.mkdir(dir_name)
            for i in range(grad.size(0)):
                cam = generate_cam(out[layer][i], grad[i, :].squeeze(0), grad_weight=True, select=False)
                path = dir_name + '/' + str(i)
                save_cam(x.view([105, 3, 224, 224])[i].squeeze(0), cam, path)
        else:
            dir_name = folder_name + '/cam'
            if not os.path.exists(dir_name):  # os模块判断并创建
                os.mkdir(dir_name)
            for i in range(grad.size(0)):
                cam = generate_cam(out[layer][i], grad[i, :].squeeze(0), grad_weight=False, select=False)
                path = dir_name + '/' + str(i)
                save_cam(x.view([105, 3, 224, 224])[i].squeeze(0), cam, path)
        # elif params.dependent_filter:
        #     # 输出各个filter
        #     for j in range(grad.size(1) + 1):
        #         dir_name = folder_name + '/filter'
        #         if not os.path.exists(dir_name):
        #             os.mkdir(dir_name)
        #         dir_name = dir_name + '/' + str(j)
        #         if j < grad.size(1):
        #             if not os.path.exists(dir_name):  # os模块判断并创建
        #                 os.mkdir(dir_name)
        #             for i in range(grad.size(0)):
        #                 cam = generate_cam(out[3][i], grad[i, :].squeeze(0), select=True, select_filter=j)
        #                 path = dir_name + '/' + str(i)
        #                 save_cam(x.view([105, 3, 224, 224])[i].squeeze(0), cam, path)
        return grad_cam

    # channel_based feature expansion
    def grad_mask6(self, grad, per1):
        channel_importance = grad.mean(3).mean(2).mean(0)
        # 向上取整，避免值过小mask的比例为0
        channel_thresh_percent = math.ceil(channel_importance.size(0) * per1)
        channel_thresh_value = torch.sort(channel_importance, dim=0, descending=True)[0][channel_thresh_percent]
        # 得到512维的0 1 向量，即512个filter中对应提取到重要模式最多的元素
        Index_channel = torch.where(channel_importance > channel_thresh_value,
                                    torch.zeros(channel_importance.shape),
                                    torch.ones(channel_importance.shape))
        mask_channel = Index_channel.unsqueeze(1).unsqueeze(2)
        mask = mask_channel.repeat(grad.shape[0], 1, grad.shape[2], grad.shape[3])
        return mask.cuda()

    def rank_cal(self, grad, i):
        a = grad[i]
        # (512,49)
        _a = a.mean(0).view(a.size(1), a.size(2) * a.size(3))
        # (7,7)
        __a = a.mean(0).mean(0)
        r1 = np.linalg.matrix_rank(_a.detach().cpu())
        r2 = np.linalg.matrix_rank(__a.detach().cpu())
        return r1, r2


# target[i,:,:] 表示第i个通道的特征图； guided_gradients
# generate_cam默认得到所有特征图与对应权重乘积的累加 给定参数 select=True, select_filter=1 时为选择1通道
def generate_cam(target, guided_gradients, grad_weight=True, select=False, select_filter=None):
    guided_gradients = guided_gradients.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    # Get weights from gradients (512, )
    weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

    # Create empty numpy array for cam (7,7)
    cam = np.ones(target.shape[1:], dtype=np.float32)

    # Have a look at issue #11 to check why the above is np.ones and not np.zeros
    # Multiply each weight with its conv output and then, sum
    # 理解： i为类别 将各个类别对应的梯度值与特征图相乘 cam的各个维度即为各个类别中对结果起重要作用的区域
    if grad_weight:
        for i, w in enumerate(weights):
            if select:
                if select_filter > -1:
                    if i == select_filter:
                        cam += w * target[i, :, :]
                        if (cam != 0).sum == 0:
                            print(i, '在当前图片无特征表示')
            else:
                # 各通道特征图累加
                cam += w * target[i, :, :]
                # cam += target[i, :, :]
    else:
        # 否则直接得到某一层的特征图
        for i, w in enumerate(weights):
            cam += target[i, :, :]

    # 7 ✖ 7
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam_resize = Image.fromarray(cam).resize((224, 224), Image.ANTIALIAS)
    cam = np.uint8(cam_resize) / 255

    # 返回target_layer位置的特征图
    return cam


# target[i,:,:] 表示第i个通道的特征图； guided_gradients
# generate_cam默认得到所有特征图与对应权重乘积的累加 给定参数 select=True, select_filter=1 时为选择1通道
def generate_cam2(target, guided_gradients=None, grad_weight=True, select=False, select_filter=None):
    # 105，512，7，7
    cam = torch.ones(target.size())
    if grad_weight:
        weights = torch.mean(guided_gradients, axis=(1, 2))
        for i, w in enumerate(weights):
            if select:
                if select_filter > -1:
                    if i == select_filter:
                        cam += w * target[i, :, :]
                        if (cam != 0).sum == 0:
                            print(i, '在当前图片无特征表示')
            else:
                # 各通道特征图累加
                cam += w * target[i, :, :]
    else:
        # 否则直接得到某一层的特征图 105,7,7
        cam = target.sum(1)

    # 7 ✖ 7 最大最小正则化
    cam[cam < 0] = 0
    cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))  # Normalize between 0-1
    # 返回target_layer位置的torch cam特征图
    return cam


# x表示从105个图片中选择的图片
def save_cam(x, cam, path):
    mask1 = cv2.resize(cam, (224, 224))
    # mask1_中亮区域为显著区域
    mask1_ = (255 * mask1).astype(np.uint8)
    mask1_ = np.repeat(mask1_[:, :, np.newaxis], 3, axis=2)

    # mask1中暗区域为显著区域
    # mask1 = (255 - 255 * mask1).astype(np.uint8)
    # mask1 = np.repeat(mask1[:, :, np.newaxis], 3, axis=2)

    # COLORMAP_JET表示将原黑白图像映射为彩虹颜色添加到原图上
    # heatmap1 = cv2.applyColorMap(255 - mask1, cv2.COLORMAP_JET)
    # 线性插值是原图的宽和高为224，224 并且使各元素为整数

    img1 = Tensor2cv2(x)
    # 对原图像关于Grad_Cam加遮罩
    heatmap1 = cv2.applyColorMap(mask1_, cv2.COLORMAP_JET)
    img = cv2.addWeighted(src1=img1, alpha=0.5, src2=heatmap1, beta=0.5, gamma=0)


    name1 = path + '_cam.jpg'
    cv2.imwrite(name1, mask1_)
    # img1 原图片，只需修改path1即可替换
    name2 = path + '_src.jpg'
    cv2.imwrite(name2, img1)
    # img 合成后的图片
    name3 = path + '_feature_map.jpg'
    cv2.imwrite(name3, img)
    # waitkey保持窗口固定不动
    # k = cv2.waitKey()
    # cv2.destroyWindow(k)


# 将tensor矩阵转化为cv2可以Imshow的形式
def Tensor2cv2(x):
    # 这种情况使像素点服从对应均值和方差的正态分布
    # x = x.detach().cpu()
    # C, W, H = x.shape
    # x_ = x.numpy().transpose(1, 2, 0).reshape(W*H,-1)
    # mean = x_.mean(0)
    # std = x_.std(0)
    # normalize = transforms.Normalize(mean=mean,std=std)
    # img = normalize(x)
    # return img

    # 使矩阵的像素值分布在[0,1]之间 然后*255转化为整数型
    x_ = x.detach().cpu().numpy().transpose(1, 2, 0)
    W, H, C = x_.shape
    x__ = x_.reshape(W * H, - 1)
    max = x__.max(0)
    min = x__.min(0)
    for i in range(C):
        x_[:, :, i] = (x_[:, :, i] - min[i]) / (max[i] - min[i])
    img = (x_ * 255).astype(np.uint8)
    return img
