import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from methods.resnet12 import ResNet12_mask1
import random


# from methods.resnet12 import ResNet12_

# 其中ResNet-RSC Module是在Resnet基础上self-challenging的模型
# 后面定义的ResNet18模型由Reset-RSC搭建

# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = 10 * cos_dist
        return scores


# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# --- LSTMCell module for matchingnet ---
class LSTMCell(nn.Module):
    FWT = False

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        # 512
        self.input_size = input_size
        # 512
        self.hidden_size = hidden_size
        self.bias = bias

        if self.FWT:
            # 重写Linear区分inner_loop和outer_loop
            # self.hidden_size = 512
            self.x2h = Linear_fw(input_size, 4 * hidden_size, bias=bias)
            self.h2h = Linear_fw(hidden_size, 4 * hidden_size, bias=bias)
        else:
            self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
            self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        # 初始化隐藏层的参数
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        if hidden is None:
            # (1,512)零向量
            hx = torch.zeors_like(x)
            cx = torch.zeros_like(x)
        else:
            hx, cx = hidden

        # x([1,512]) hx([1,512])---作为初始化的support set
        # x2h:([512,2048]) h2h:([512,2048])
        gates = self.x2h(x) + self.h2h(hx)
        # gates: ([1,2048])
        # 每512个元素划分gates为下面的四个门
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # (1,512)
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)


# --- LSTM module for matchingnet ---
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        assert (self.num_layers == 1)

        self.lstm = LSTMCell(input_size, hidden_size, self.bias)

    # G_encoder S: 根据SupportSet编码G函数，其中包含与两个512*2048网络相关的门控单元
    def forward(self, x, hidden=None):
        # swap axis if batch first
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # hidden state
        if hidden is None:
            h0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
            c0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h0, c0 = hidden

        # forward
        outs = []
        hn = h0[0]
        cn = c0[0]
        for seq in range(x.size(0)):
            hn, cn = self.lstm(x[seq], (hn, cn))
            outs.append(hn.unsqueeze(0))
        outs = torch.cat(outs, dim=0)

        # reverse foward
        if self.num_directions == 2:
            outs_reverse = []
            hn = h0[1]
            cn = c0[1]
            for seq in range(x.size(0)):
                seq = x.size(1) - 1 - seq
                hn, cn = self.lstm(x[seq], (hn, cn))
                outs_reverse.append(hn.unsqueeze(0))
            outs_reverse = torch.cat(outs_reverse, dim=0)
            outs = torch.cat([outs, outs_reverse], dim=2)

        # swap axis if batch first
        if self.batch_first:
            outs = outs.permute(1, 0, 2)
        return outs


# --- Linear module ---
class Linear_fw(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
        # fast为自定义的属性，可以为任何名称(若fast非None,依据具体fast的值计算out)
        # 否则,若为None,执行可学习的线性层得到的out
        # 从而区分当前为inner_loop或outer_loop
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            # 若self.weight.fast 和 self
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            # super(Linear_fw,self)表示父类 所以此处执行父类的forward
            out = super(Linear_fw, self).forward(x)
        return out


# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out


# --- softplus module ---
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = False

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * softplus(
                self.beta)).expand_as(out)
            out = gamma * out + beta
        return out


# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            # 外循环更新后的参数作为weight
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


# --- Add Layer replace the + in the residule block----
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


# --- Simple Conv Block ---
class ConvBlock(nn.Module):
    FWT = False

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.FWT:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = FeatureWiseTransformation2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


# --- ConvNet module ---
class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        self.grads = []
        self.fmaps = []
        self.get_feature_map=False
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for first 4 layers
            trunk.append(B)

        self.final_feat_dim = 1600
        if flatten:
            trunk.append(nn.AvgPool2d(5))
            trunk.append(Flatten())
            self.final_feat_dim = 64

        self.trunk = nn.Sequential(*trunk)


    def forward(self, x):
        if self.get_feature_map:
            out = self.trunk[0:4](x)
        else:
            out = self.trunk(x)
        return out


# --- ConvNetNopool module ---
class ConvNetNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


# 任务自适应模块
class ScaleConv2d(nn.Conv2d):
    def __init__(self, in_plane, out_plane, **kwargs):
        super(ScaleConv2d, self).__init__(in_plane, out_plane, **kwargs)
        # self.alpha = nn.Parameter(torch.ones(self.out_channels, self.in_channels))
        self.alpha = nn.Parameter(torch.ones(self.out_channels))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.weight * self.alpha.unsqueeze(1).unsqueeze(2).unsqueeze(2), self.bias)


# scale = True

# ResNet的基本模块
# --- Simple ResNet Block ---
class SimpleBlock(nn.Module):
    FWT = False

    # Scale = scale

    def __init__(self, indim, outdim, half_res, leaky=False, Scale=False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.Scale = Scale
        if self.FWT:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = FeatureWiseTransformation2d_fw(
                outdim)  # feature-wise transformation at the end of each residual block
        elif self.Scale:
            self.C1 = ScaleConv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = ScaleConv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.FWT:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
            elif self.Scale:
                self.shortcut = ScaleConv2d(indim, outdim, kernel_size=1, stride=2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)

        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


# --- ResNet module ---
class ResNet(nn.Module):
    FWT = False
    # Scale = scale
    RSC = False
    Dropout = False

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False, Scale=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        self.gradients = []
        # 只在train_grad_visual文件中为False
        self.no_hook=True
        self.fmaps = []
        self.Scale = Scale
        # 仅在计算Hrank时置为True
        self.get_feature_map=False
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.FWT:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = BatchNorm2d_fw(64)
        elif self.Scale:
            conv1 = ScaleConv2d(in_plane=3, out_plane=64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu, Scale=self.Scale)
                trunk.append(B)
                indim = list_of_out_dims[i]

        # flatten决定是否对网络输出的特征表示进行池化和展平操作
        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        # self.trunk = nn.Sequential(*trunk)
        self.trunk = nn.ModuleList(trunk)
        # self.net = nn.Sequential(*trunk)
        self.mask0_ = False
        self.mask0 = None
        self.mask1_ = False
        self.mask1 = None
        self.mask2_ = False
        self.mask2 = None
        self.mask3_ = False
        self.mask3 = None
        self.mask4_ = False
        self.mask4 = None
        # 用来区分训练或者测试过程
        self.train_ = True
        self.conv_output = []

        # dropout
        self.dropout = False
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)

        # channel_adapter
        self.channel_adapter = False
        # 初始化channel_adapter中需要训练的参数，如果该参数不参与前传，神经网络对其梯度始终为0
        # requires_grad = False说明默认该参数不需要初始化，不需要梯度传播
        self.alpha = nn.Parameter(torch.ones(self.final_feat_dim), requires_grad=False)

        self.channel_adapter2 = False
        self.alpha1 = nn.Parameter(torch.ones(list_of_out_dims[0]), requires_grad=False)
        self.alpha2 = nn.Parameter(torch.ones(list_of_out_dims[1]), requires_grad=False)
        self.alpha3 = nn.Parameter(torch.ones(list_of_out_dims[2]), requires_grad=False)
        self.alpha4 = nn.Parameter(torch.ones(list_of_out_dims[3]), requires_grad=False)

    def forward(self, x):
        self.conv_output = []
        if self.RSC:
            a = self.trunk
            # block 前面的层
            layer_0 = nn.Sequential(*a[0:4]).cuda()
            x = layer_0(x)  # x_0
            train_ = self.train_

            # 对原始数据，即block0做mask
            if train_:
                x.register_hook(self.save_gradient)
            if self.mask0_:
                x = x * self.mask0

            # block1 前传过程不受不同网络不同层数的影响
            layer_1 = a[4].cuda()
            x = layer_1(x)  # (105,64,56,56)
            self.conv_output.append(x)
            # grad[3]
            if train_:
                x.register_hook(self.save_gradient)
            if self.mask1_:
                x = x * self.mask1

            # block2
            layer_2 = a[5].cuda()
            x = layer_2(x)  # (105,128,28,28)
            self.conv_output.append(x)
            # grad[2]
            if train_:
                x.register_hook(self.save_gradient)
            if self.mask2_:
                x = x * self.mask2

            # block3
            layer_3 = a[6].cuda()
            x = layer_3(x)  # ([105, 256, 14, 14])
            self.conv_output.append(x)
            # grad[1]
            if train_:
                x.register_hook(self.save_gradient)
            if self.mask3_:
                x = x * self.mask3

            # block4
            layer_4 = a[7].cuda()
            x = layer_4(x)  # ([105, 512, 7, 7])
            # conv_output[4]
            self.conv_output.append(x)
            # grad[0]
            if train_:
                x.register_hook(self.save_gradient)
            if self.mask4_:
                x = x * self.mask4

            # Flatten+Avgpool
            layer_final = nn.Sequential(*a[8:10])
            out = layer_final(x)
        elif self.dropout:
            a = self.trunk
            # block 前面 的层
            layer_0 = nn.Sequential(*a[0:4]).cuda()
            x = layer_0(x)  # x_0
            # block1 前传过程不受不同网络不同层数的影响
            layer_1 = a[4].cuda()
            x = layer_1(x)
            x = self.dropout1(x)
            # block2
            layer_2 = a[5].cuda()
            x = layer_2(x)
            x = self.dropout2(x)
            # block3
            layer_3 = a[6].cuda()
            x = layer_3(x)
            x = self.dropout3(x)
            # block4
            layer_4 = a[7].cuda()
            x = layer_4(x)
            x = self.dropout2(x)

            # Flatten+Avgpool
            layer_final = nn.Sequential(*a[8:10])
            out = layer_final(x)
        elif self.channel_adapter2:
            a = self.trunk
            # block 前面 的层
            layer_0 = nn.Sequential(*a[0:4]).cuda()
            x = layer_0(x)  # x_0
            # block1 前传过程不受不同网络不同层数的影响
            layer_1 = a[4].cuda()
            x = layer_1(x) * self.alpha1.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            # block2
            layer_2 = a[5].cuda()
            x = layer_2(x) * self.alpha2.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            # block3
            layer_3 = a[6].cuda()
            x = layer_3(x) * self.alpha3.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            # block4
            layer_4 = a[7].cuda()
            x = layer_4(x)
            # Flatten+Avgpool
            layer_final = nn.Sequential(*a[8:10])
            out = layer_final(x) * self.alpha4.unsqueeze(0)
        elif self.get_feature_map:
            trunk = nn.Sequential(*self.trunk[0:8]).cuda()
            out = trunk(x)
        else:
            trunk = nn.Sequential(*self.trunk).cuda()
            out = trunk(x)
            if not self.no_hook:
                self.gradients = []
                out.register_hook(self.save_gradient)
        if self.channel_adapter:
            out = out * self.alpha.unsqueeze(0)
        return out

    # hook函数
    def save_gradient(self, grad):
        self.gradients.append(grad)


# --- ResNet module ---
class ResNet_mask(nn.Module):
    FWT = False
    RSC = False
    Dropout = False
    train_ = False
    train1_=False

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet_mask, self).__init__()
        self.gradients = []
        self.fmaps = []
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.FWT:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu)
                trunk.append(B)
                indim = list_of_out_dims[i]

        # flatten决定是否对网络输出的特征表示进行池化和展平操作
        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        # self.trunk = nn.Sequential(*trunk)
        self.trunk = nn.ModuleList(trunk)
        # self.net = nn.Sequential(*trunk)
        self.mask0_ = False
        self.mask0 = None
        self.mask1_ = False
        self.mask1 = None
        self.mask2_ = False
        self.mask2 = None
        self.mask3_ = False
        self.mask3 = None
        self.mask4_ = False
        self.mask4 = None
        # 用来区分训练或者测试过程
        self.train_ = True
        self.conv_output = []

        # dropout
        self.dropout = False
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)

        # finetune阶段可能用到的参数
        # channel_adapter
        self.channel_adapter = False
        # 初始化channel_adapter中需要训练的参数，如果该参数不参与前传，神经网络对其梯度始终为0
        # requires_grad = False说明默认该参数不需要初始化，不需要梯度传播
        self.alpha = nn.Parameter(torch.ones(self.final_feat_dim), requires_grad=False)

        self.channel_adapter2 = False
        self.alpha1 = nn.Parameter(torch.ones(list_of_out_dims[0]), requires_grad=False)
        self.alpha2 = nn.Parameter(torch.ones(list_of_out_dims[1]), requires_grad=False)
        self.alpha3 = nn.Parameter(torch.ones(list_of_out_dims[2]), requires_grad=False)
        self.alpha4 = nn.Parameter(torch.ones(list_of_out_dims[3]), requires_grad=False)

    def forward(self, x):
        # self.conv_output = []
        if self.RSC:
            a = self.trunk
            # block 前面的层
            layer_0 = nn.Sequential(*a[0:4]).cuda()
            x = layer_0(x)  # x_0
            train_ = self.train_
            # 为了记录反传到grad上的梯度值
            train1_ = self.train1_
            # self.gradients = []

            # 对原始数据，即block0做mask
            # if train_:
            #     x.register_hook(self.save_gradient)
            if self.mask0_:
                x = x * self.mask0
            # if train_:
            #     x.register_hook(self.save_gradient)

            # block1 前传过程不受不同网络不同层数的影响
            layer_1 = a[4].cuda()
            x = layer_1(x)
            # self.conv_output.append(x)
            # grad[3]
            # if train_:
            #     x.register_hook(self.save_gradient)
            if self.mask1_:
                x = x * self.mask1
            if train_:
                x.register_hook(self.save_gradient)

            # block2
            layer_2 = a[5].cuda()
            x = layer_2(x)
            # self.conv_output.append(x)
            # grad[2]
            # if train_:
            #     x.register_hook(self.save_gradient)
            if self.mask2_:
                x = x * self.mask2
            if train_:
                x.register_hook(self.save_gradient)

            # block3
            layer_3 = a[6].cuda()
            x = layer_3(x)
            # self.conv_output.append(x)
            # grad[1]
            # if train_:
            #     x.register_hook(self.save_gradient)
            if self.mask3_:
                x = x * self.mask3
            if train_:
                x.register_hook(self.save_gradient)

            # block4
            layer_4 = a[7].cuda()
            x = layer_4(x)
            # conv_output[4]
            # self.conv_output.append(x)
            # grad[0]
            # if train_:
            #     x.register_hook(self.save_gradient)
            if train1_:
                self.gradients = []
                x.register_hook(self.save_gradient)
            if self.mask4_:
                x = x * self.mask4
            if train_:
                x.register_hook(self.save_gradient)

            # Flatten+Avgpool
            layer_final = nn.Sequential(*a[8:10])
            out = layer_final(x)
        elif self.dropout:
            a = self.trunk
            # block 前面 的层
            layer_0 = nn.Sequential(*a[0:4]).cuda()
            x = layer_0(x)  # x_0
            # block1 前传过程不受不同网络不同层数的影响
            layer_1 = a[4].cuda()
            x = layer_1(x)
            x = self.dropout1(x)
            # block2
            layer_2 = a[5].cuda()
            x = layer_2(x)
            x = self.dropout2(x)
            # block3
            layer_3 = a[6].cuda()
            x = layer_3(x)
            x = self.dropout3(x)
            # block4
            layer_4 = a[7].cuda()
            x = layer_4(x)
            x = self.dropout4(x)

            # Flatten+Avgpool
            layer_final = nn.Sequential(*a[8:10])
            out = layer_final(x)
        elif self.channel_adapter2:
            a = self.trunk
            # block 前面 的层
            layer_0 = nn.Sequential(*a[0:4]).cuda()
            x = layer_0(x)  # x_0
            # block1 前传过程不受不同网络不同层数的影响
            layer_1 = a[4].cuda()
            x = layer_1(x) * self.alpha1.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            # block2
            layer_2 = a[5].cuda()
            x = layer_2(x) * self.alpha2.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            # block3
            layer_3 = a[6].cuda()
            x = layer_3(x) * self.alpha3.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            # block4
            layer_4 = a[7].cuda()
            x = layer_4(x)

            # Flatten+Avgpool
            layer_final = nn.Sequential(*a[8:10])
            out = layer_final(x) * self.alpha4.unsqueeze(0)
        else:
            trunk = nn.Sequential(*self.trunk).cuda()
            out = trunk(x)

        if self.channel_adapter:
            out = out * self.alpha.unsqueeze(0)
        return out

    # hook函数
    def save_gradient(self, grad):
        self.gradients.append(grad)

    def generate_grad(self, i, grad1=None, grad2=None, grad3=None, grad4=None):
        grad = self.gradients
        if i == 0:
            grad1 = grad[3].detach()
            grad2 = grad[2].detach()
            grad3 = grad[1].detach()
            grad4 = grad[0].detach()
        else:
            grad1 += grad[3].detach()
            grad2 += grad[2].detach()
            grad3 += grad[1].detach()
            grad4 += grad[0].detach()
        return grad1, grad2, grad3, grad4

    def generate_grad_abs(self, i, grad1=None, grad2=None, grad3=None, grad4=None):
        grad = self.gradients
        if i == 0:
            grad1 = grad[3].detach().abs()
            grad2 = grad[2].detach().abs()
            grad3 = grad[1].detach().abs()
            grad4 = grad[0].detach().abs()
        else:
            grad1 += grad[3].detach().abs()
            grad2 += grad[2].detach().abs()
            grad3 += grad[1].detach().abs()
            grad4 += grad[0].detach().abs()
        return grad1, grad2, grad3, grad4

    def generate_mask(self, grad1, grad2, grad3, grad4, per):
        # 每个epoch更新一次
        self.mask1 = self.grad_mask(grad1, per)
        self.mask2 = self.grad_mask(grad2, per)
        self.mask3 = self.grad_mask(grad3, per)
        self.mask4 = self.grad_mask(grad4, per)

    def mask_(self, epoch):
        if epoch != 0 and epoch % 4 == 0:
            self.mask1_ = True
        elif epoch % 4 == 1:
            self.mask2_ = True
        # elif epoch != 50 and epoch % 4 == 2:
        elif epoch % 4 == 2:
            self.mask3_ = True
        elif epoch % 4 == 3:
            self.mask4_ = True

    def mask_rec(self):
        self.mask1_ = False
        self.mask2_ = False
        self.mask3_ = False
        self.mask4_ = False
        self.RSC = False

    def grad_mask(self, grad, per):
        channel_importance = grad.cpu().mean(3).mean(2).mean(0)
        # 向上取整，避免值过小mask的比例为0
        channel_thresh_percent = math.floor(channel_importance.size(0) * per)
        channel_thresh_value = torch.sort(channel_importance, dim=0, descending=True)[0][channel_thresh_percent]
        # 得到512维的0 1 向量，即512个filter中对应提取到重要模式最多的元素
        Index_channel = torch.where(channel_importance > channel_thresh_value,
                                    torch.zeros(channel_importance.shape),
                                    torch.ones(channel_importance.shape))
        mask_channel = Index_channel.unsqueeze(1).unsqueeze(2)
        mask = mask_channel.repeat(grad.shape[0], 1, grad.shape[2], grad.shape[3])
        return mask.cuda()


# --- ResNet module ---
class ConvNet_mask(nn.Module):
    RSC = False
    train_ = False
    def __init__(self, depth, flatten=True):
        super(ConvNet_mask, self).__init__()
        self.gradients = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for first 4 layers
            trunk.append(B)

        self.final_feat_dim = 1600
        if flatten:
            trunk.append(nn.AvgPool2d(5))
            trunk.append(Flatten())
            self.final_feat_dim = 64

        self.trunk = nn.Sequential(*trunk)
        # self.trunk = nn.ModuleList(trunk)


        self.mask0_ = False
        self.mask0 = None
        self.mask1_ = False
        self.mask1 = None
        self.mask2_ = False
        self.mask2 = None
        self.mask3_ = False
        self.mask3 = None
        self.mask4_ = False
        self.mask4 = None
        # 用来区分训练或者测试过程
        self.train_ = True
        self.conv_output = []

    def forward(self, x):
        if self.RSC:
            a = self.trunk
            train_ = self.train_

            # block1 前传过程不受不同网络不同层数的影响
            layer_1 = a[0].cuda()
            x = layer_1(x)
            if self.mask1_:
                x = x * self.mask1
            if train_:
                x.register_hook(self.save_gradient)

            # block2
            layer_2 = a[1].cuda()
            x = layer_2(x)
            if self.mask2_:
                x = x * self.mask2
            if train_:
                x.register_hook(self.save_gradient)

            # block3
            layer_3 = a[2].cuda()
            x = layer_3(x)
            if self.mask3_:
                x = x * self.mask3
            if train_:
                x.register_hook(self.save_gradient)

            # block4
            layer_4 = a[3].cuda()
            x = layer_4(x)
            if self.mask4_:
                x = x * self.mask4
            if train_:
                x.register_hook(self.save_gradient)

            # Flatten+Avgpool
            layer_final = a[4:6]
            out = layer_final(x)
        else:
            trunk = nn.Sequential(*self.trunk).cuda()
            out = trunk(x)
        return out


    # hook函数
    def save_gradient(self, grad):
        self.gradients.append(grad)

    def generate_grad(self, i, grad1=None, grad2=None, grad3=None, grad4=None):
        grad = self.gradients
        if i == 0:
            grad1 = grad[3].detach()
            grad2 = grad[2].detach()
            grad3 = grad[1].detach()
            grad4 = grad[0].detach()
        else:
            grad1 += grad[3].detach()
            grad2 += grad[2].detach()
            grad3 += grad[1].detach()
            grad4 += grad[0].detach()
        return grad1, grad2, grad3, grad4

    def generate_grad_abs(self, i, grad1=None, grad2=None, grad3=None, grad4=None):
        grad = self.gradients
        if i == 0:
            grad1 = grad[3].detach().abs()
            grad2 = grad[2].detach().abs()
            grad3 = grad[1].detach().abs()
            grad4 = grad[0].detach().abs()
        else:
            grad1 += grad[3].detach().abs()
            grad2 += grad[2].detach().abs()
            grad3 += grad[1].detach().abs()
            grad4 += grad[0].detach().abs()
        return grad1, grad2, grad3, grad4

    def generate_mask(self, grad1, grad2, grad3, grad4, per):
        # 每个epoch更新一次
        self.mask1 = self.grad_mask(grad1, per)
        self.mask2 = self.grad_mask(grad2, per)
        self.mask3 = self.grad_mask(grad3, per)
        self.mask4 = self.grad_mask(grad4, per)

    def mask_(self, epoch):
        if epoch != 0 and epoch % 4 == 0:
            self.mask1_ = True
        elif epoch % 4 == 1:
            self.mask2_ = True
        elif epoch!=250 and epoch %4 == 2:
        # elif epoch % 4 == 2:
            self.mask3_ = True
        elif epoch % 4 == 3:
            self.mask4_ = True

    def mask_rec(self):
        self.mask1_ = False
        self.mask2_ = False
        self.mask3_ = False
        self.mask4_ = False
        self.RSC = False

    def grad_mask(self, grad, per):
        channel_importance = grad.cpu().mean(3).mean(2).mean(0)
        # 向上取整，避免值过小mask的比例为0
        channel_thresh_percent = math.floor(channel_importance.size(0) * per)
        channel_thresh_value = torch.sort(channel_importance, dim=0, descending=True)[0][channel_thresh_percent]
        # 得到512维的0 1 向量，即512个filter中对应提取到重要模式最多的元素
        Index_channel = torch.where(channel_importance > channel_thresh_value,
                                    torch.zeros(channel_importance.shape),
                                    torch.ones(channel_importance.shape))
        mask_channel = Index_channel.unsqueeze(1).unsqueeze(2)
        mask = mask_channel.repeat(grad.shape[0], 1, grad.shape[2], grad.shape[3])
        return mask.cuda()



# --- Conv networks ---
# def Conv4():
#     return ConvNet(4, flatten=False)

def Conv4(flatten=True, leakyrelu=False):
    return ConvNet(4, flatten=flatten)

def Conv4_mask(flatten=True, leakyrelu=False):
    return ConvNet_mask(4, flatten=flatten)

def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


# --- ResNet networks ---
def ResNet10(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten, leakyrelu)


# --- ResNet networks ---
def ResNet10_mask(flatten=True, leakyrelu=False):
    return ResNet_mask(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten, leakyrelu)

# --- ResNet networks ---
def ResNet12_mask(flatten=True, leakyrelu=False):
    return ResNet12_mask1([64, 160, 320, 640])


def ResNet18(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten, leakyrelu)


def ResNet34(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten, leakyrelu)


# def ResNet12(flatten=True, leakyrelu=False):
# return ResNet12_(flatten=flatten)
# return ResNet12_2([64, 160, 320, 640])

# def ResNet12(flatten=True, leakyrelu=True):
#   return ResNet(SimpleBlock, [1, 1, 2, 2], [64, 160, 320, 640], flatten, leakyrelu)

model_dict = dict(Conv4=Conv4, Conv6=Conv6, ResNet10=ResNet10, ResNet10_mask=ResNet10_mask, ResNet18=ResNet18,
                  ResNet34=ResNet34, Conv4_mask=Conv4_mask, ResNet12_mask=ResNet12_mask)
