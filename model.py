from torch import randn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant, kaiming_normal
from torch.autograd import Variable
from collections import OrderedDict
import math
import torch


def weights_init(m):
    """
    权重初始化，默认使用凯明大神提出的初始化方法。
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        kaiming_normal(m.weight.data)
        if m.bias is not None:
            constant(m.bias.data, 0)


class FFPolicy(nn.Module):
    """
    执行神经网络运算的父类
    """
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic: bool):
        """
        以deterministic的方式，根据网络计算结果采取action，并且评估该状态的价值函数的值。
        :param inputs:
        :param deterministic: True表示取概率最高的action值，False表示按照概率分布取action值。
        :return:
        """
        value, x = self(inputs)
        action = self.cls_linear.sample(x, deterministic=deterministic)
        action_log_probs, cls_entropy = self.cls_linear.logprobs_and_entropy(x, action)
        return value, action, action_log_probs

    def evaluate_actions(self, inputs, actions):
        """
        评估在状态inputs下，采取行为actions的价值。
        :param inputs:
        :param actions:
        :return:
        """
        value, x = self(inputs)
        action_log_probs, cls_entropy = self.cls_linear.logprobs_and_entropy(x, actions)
        return value, action_log_probs, cls_entropy


class FcNet(FFPolicy):
    """
    全连接神经网络
    """

    def __init__(self, layers: list, pi_out: int):
        super(FcNet, self).__init__()

        self.layer_num = len(layers)-1

        self.model = nn.Sequential()
        for i in range(self.layer_num):
            self.model.add_module('layer'+str(i+1),
                                  nn.Linear(in_features=layers[i], out_features=layers[i+1], bias=True))
        self.cls_linear = Categorical(layers[self.layer_num], pi_out)
        self.critic_linear = nn.Linear(layers[self.layer_num], 1, bias=True)

        self.train()
        self.apply(weights_init)

    def forward(self, inputs):
        x = self.model(inputs)
        return self.critic_linear(x), x


class SimplestNet(FFPolicy):
    """
    SimpleNet的速度不给力啊，切一个更小的网络
    """
    def __init__(self, in_channels: int=3, num_classes=1000):
        super(SimplestNet, self).__init__()
        mult = [32, 32]
        print('build simplenet with in_channel {}, and out_channel {}'.format(in_channels, num_classes))
        self.model = nn.Sequential(
            # 2nx112x112 --> 3nx28x28
            nn.Conv2d(in_channels=in_channels, out_channels=mult[0], kernel_size=5, stride=4, padding=1),
            nn.BatchNorm2d(num_features=mult[0]),
            nn.ReLU(inplace=True),
            # 3nx28x28 --> 4nx7x7
            nn.Conv2d(in_channels=mult[0], out_channels=mult[1], kernel_size=5, stride=4, padding=1),
            nn.BatchNorm2d(num_features=mult[1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7)
        )
        self.num_nn = mult[1]
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_nn, out_features=256, bias=True),
            nn.ReLU(inplace=True)
        )

        self.critic_linear = nn.Linear(256, 1)
        self.cls_linear = Categorical(256, num_classes)  # classification 分类器

        self.train()  # 设置成训练模式
        self.apply(weights_init)  # 初始化相关参数

    def forward(self, inputs):
        x = self.model(inputs)
        x = x.view(-1, self.num_nn)
        x = self.fc(x)

        return self.critic_linear(x), x


class ExpandSimpleNet(FFPolicy):
    """
    SImpleNet的扩展网络，扩展方向在广度上
    """

    def __init__(self, in_channels: int=3, num_classes=1000, expand_factor: int=2):
        super(ExpandSimpleNet, self).__init__()
        mult = [32, 32, 64] * expand_factor
        print('build expandsimplenet with in_channel {}, and out_channel {}'.format(in_channels, num_classes))
        self.model = nn.Sequential(
            # x112x112 --> x56x56
            nn.Conv2d(in_channels=in_channels, out_channels=mult[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[0]),
            nn.ReLU(inplace=True),
            # x56x56 --> x28x28
            nn.Conv2d(in_channels=mult[0], out_channels=mult[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[1]),
            nn.ReLU(inplace=True),
            # x28x28 --> x7x7
            nn.Conv2d(in_channels=mult[1], out_channels=mult[2], kernel_size=5, stride=4, padding=1),
            nn.BatchNorm2d(num_features=mult[2]),
            nn.ReLU(inplace=True),
            # pooling --> x1x1
            nn.AvgPool2d(7)
        )
        self.num_nn = mult[2]
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_nn, out_features=512, bias=True),
            nn.ReLU(inplace=True)
        )

        self.critic_linear = nn.Linear(512, 1)
        self.cls_linear = Categorical(512, num_classes)  # classification 分类器

        self.train()  # 设置成训练模式
        self.apply(weights_init)  # 初始化相关参数

    def forward(self, inputs):
        x = self.model(inputs)
        x = x.view(-1, self.num_nn)
        x = self.fc(x)

        return self.critic_linear(x), x



class SimpleNet(FFPolicy):
    """
    非常简单的网络，为了能够在自己的破电脑上也能运行程序，我也是拼了
    224x224已经改成112x112
    """

    def __init__(self, in_channels: int=3, num_classes=1000):
        super(SimpleNet, self).__init__()
        mult = [32, 32, 64]
        print('build simplenet with in_channel {}, and out_channel {}'.format(in_channels, num_classes))
        self.model = nn.Sequential(
            # x112x112 --> x56x56
            nn.Conv2d(in_channels=in_channels, out_channels=mult[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[0]),
            nn.ReLU(inplace=True),
            # x56x56 --> x28x28
            nn.Conv2d(in_channels=mult[0], out_channels=mult[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[1]),
            nn.ReLU(inplace=True),
            # x28x28 --> x7x7
            nn.Conv2d(in_channels=mult[1], out_channels=mult[2], kernel_size=5, stride=4, padding=1),
            nn.BatchNorm2d(num_features=mult[2]),
            nn.ReLU(inplace=True),
            # pooling --> x1x1
            nn.AvgPool2d(7)
        )
        self.num_nn = mult[2]
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_nn, out_features=512, bias=True),
            nn.ReLU(inplace=True)
        )

        self.critic_linear = nn.Linear(512, 1)
        self.cls_linear = Categorical(512, num_classes)  # classification 分类器

        self.train()  # 设置成训练模式
        self.apply(weights_init)  # 初始化相关参数

    def forward(self, inputs):
        x = self.model(inputs)
        x = x.view(-1, self.num_nn)
        x = self.fc(x)

        return self.critic_linear(x), x


class DeeperSimpleNet(FFPolicy):
    """
    SimpleNet的深度扩展
    """
    def __init__(self, in_channels: int=3, num_classes=1000, expand_factor: int=2):
        super(DeeperSimpleNet, self).__init__()
        mult = [32, 32, 64, 128] * expand_factor
        print('build deepersimplenet with in_channel {}, and out_channel {}'.format(in_channels, num_classes))
        self.model = nn.Sequential(
            # x112x112 --> x56x56
            nn.Conv2d(in_channels=in_channels, out_channels=mult[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mult[0], out_channels=mult[0], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=mult[0]),
            nn.ReLU(inplace=True),
            # x56x56 --> x28x28
            nn.Conv2d(in_channels=mult[0], out_channels=mult[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mult[1], out_channels=mult[1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=mult[1]),
            nn.ReLU(inplace=True),
            # x28x28 --> x14x14
            nn.Conv2d(in_channels=mult[1], out_channels=mult[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mult[2], out_channels=mult[2], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=mult[2]),
            nn.ReLU(inplace=True),
            # x14x14 --> x7x7
            nn.Conv2d(in_channels=mult[2], out_channels=mult[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mult[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mult[3], out_channels=mult[3], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=mult[3]),
            nn.ReLU(inplace=True),
            # pooling --> x1x1
            nn.AvgPool2d(4)
        )
        self.num_nn = mult[3]
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_nn, out_features=512, bias=True),
            nn.ReLU(inplace=True)
        )

        self.critic_linear = nn.Linear(512, 1)
        self.cls_linear = Categorical(512, num_classes)  # classification 分类器

        self.train()  # 设置成训练模式
        self.apply(weights_init)  # 初始化相关参数

    def forward(self, inputs):
        x = self.model(inputs)
        x = x.view(-1, self.num_nn)
        x = self.fc(x)

        return self.critic_linear(x), x


class AlexNet(FFPolicy):
    """
    AlexNet的结构变体，可见：https://github.com/BoyuanYan/pytorch-playground/blob/master/imagenet/alexnet.py
    """

    def __init__(self, in_channels: int=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.critic_linear = nn.Linear(4096, 1)
        self.cls_linear = Categorical(4096, num_classes)  # classification 分类器

        self.train()  # 设置成训练模式
        self.apply(weights_init)  # 初始化相关参数

    def forward(self, inputs):
        x = self.features(inputs)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)

        return self.critic_linear(x), x


class MobileNetV2(FFPolicy):
    """
    mobilenet V2。原文可见：https://arxiv.org/abs/1801.04381

    本网络结构是为了A2C程序设计的，因此会在最后输出的部分，区分pi函数和vf函数，即策略函数和价值函数。这两个函数共享基本网络。
    TODO 此外，还需要注意的一点是，输入图像的长宽可能不是224x224，这个要检察一下。
    TODO 还有，输入的channels比较多，一上来就压缩channels，会不会造成信息丢失比较大？


    下述结构即MobileNet v2的结构，其中channels表示扩张倍数，c表示输出channels个数，n表示重复次数，s表示stride

    |     name     |    Input   |  Operator  | t |  c  | n | s |
    | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- |
    |     conv_1   | 224x224 x3 | conv2d 3x3 | - | 32  | 1 | 2 |
    | bottleneck_1 | 112x112x32 | bottleneck | 1 | 16  | 1 | 1 |
    | bottleneck_2 | 112x112x16 | bottleneck | 6 | 24  | 2 | 2 |
    | bottleneck_3 | 56 x56 x24 | bottleneck | 6 | 32  | 3 | 2 |
    | bottleneck_4 | 28 x28 x32 | bottleneck | 6 | 64  | 4 | 2 |
    | bottleneck_5 | 14 x14 x64 | bottleneck | 6 | 96  | 3 | 1 |
    | bottleneck_6 | 14 x14 x96 | bottleneck | 6 | 160 | 3 | 2 |
    | bottleneck_7 | 7  x7 x160 | bottleneck | 6 | 320 | 1 | 1 |
    |    conv_2    | 7  x7 x320 | conv2d 1x1 | - | 1280| 1 | 1 |
    |    avgpool   | 7  x7 x1280| avgpool 7x7| - |  -  | 1 | - |
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    |      fc      | 1  x1 xk   |   Linear   | - |  k  | - |   |
    """

    def __init__(self, in_channels: int=3, num_classes: int=1000, t: int=6):
        super(MobileNetV2, self).__init__()
        #          c1, b1, b2, b3, b4, b5, b6,  b7,  c2
        out_chs = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        strides = [ 2,  1,  2,  2,  2,  1,   2,   1,    1]
        r_times = [ 1,  1,  2,  3,  4,  3,   3,   1,    1]
        factors = [-1,  1,  t,  t,  t,  t,   t,   t,   -1]

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_chs[0], kernel_size=3,
                      stride=strides[0], padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_chs[0]),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential()
        for i in range(7):
            name = 'bottleneck_' + str(i+1)
            bnk = StackBottleneck(in_features=out_chs[i], out_features=out_chs[i+1],
                                  first_stride=strides[i+1], factor_t=factors[i+1],
                                  repeated_times=r_times[i+1], name=name)
            self.bottleneck.add_module(name, bnk)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chs[7], out_channels=out_chs[8], kernel_size=1,
                      stride=strides[8], padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_chs[8]),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, ceil_mode=True)

        self.critic_linear = nn.Linear(1280, 1)  # value function 评价器
        self.cls_linear = Categorical(1280, num_classes)  # classification 分类器

        self.train()  # 设置成训练模式
        self.apply(weights_init)  # 初始化相关参数

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        x = self.conv_1(inputs)
        x = self.bottleneck(x)
        x = self.conv_2(x)
        x = self.avgpool(x)
        x = x.view(-1, 1280)

        return self.critic_linear(x), x


class Bottleneck(nn.Module):
    """
    mobilenet v2中定义的bottleneck。
    """

    def __init__(self, in_features: int, out_features: int, stride: int, factor_t: int):
        """

        :param in_features:
        :param out_features:
        :param stride:
        :param factor_t:
        """
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.use_res = in_features == out_features
        middle_features = in_features * factor_t
        self.stem = nn.Sequential(
            # pointwise conv
            nn.Conv2d(in_channels=in_features, out_channels=middle_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=middle_features),
            nn.ReLU(inplace=True),
            # depthwise conv
            nn.Conv2d(in_channels=middle_features, out_channels=middle_features, kernel_size=3,
                      stride=stride, padding=1, groups=middle_features, bias=False),
            nn.BatchNorm2d(num_features=middle_features),
            nn.ReLU(inplace=True),
            ## pointwise conv
            nn.Conv2d(in_channels=middle_features, out_channels=out_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_features)
            # 最后一层无ReLU。
        )

    def forward(self, x):
        y = self.stem(x)
        if self.stride is 1 and self.use_res:
            y = y + x
        return y


class StackBottleneck(nn.Module):
    """
    mobilenet v2中bottlenecks的叠加
    """

    def __init__(self, in_features: int, out_features: int, first_stride: int,
                 factor_t: int, repeated_times: int, name: str):
        """

        :param in_features:
        :param out_features:
        :param first_stride:
        :param factor_t:
        :param repeated_times:
        """
        super(StackBottleneck, self).__init__()

        self.model = nn.Sequential(
            Bottleneck(in_features=in_features, out_features=out_features, stride=first_stride,
                       factor_t=factor_t)
        )

        for i in range(repeated_times-1):
            module = Bottleneck(in_features=out_features, out_features=out_features, stride=1,
                                factor_t=factor_t)
            self.model.add_module(name=name+'_'+str(i+1), module=module)

    def forward(self, x):
        y = self.model(x)
        return y


class Categorical(nn.Module):
    """
    一层全连接神经网络的实现，作用是实现分类器的功能。
    """

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        """
        根据输入，进行输出抽样，选择一个action
        :param x: 前面FC或者CNN或者LSTM实现得到输出，已经被压缩成一维
        :param deterministic: 确定性选项，如果为True，则表示从所有输出action选项中选择可能性最大的；如果为False，则按照概率进行选择。
        :return: 采取的行为
        """
        x = self(x)
        # 计算各种行为的概率。dim=1表示在行的维度计算softmax。
        probs = F.softmax(x, dim=1)
        if deterministic is False:
            #  按照softmax的输出，以输出中各元素占据的权重为概率输出一个值
            action = probs.multinomial()
        else:
            action = probs.max(1, keepdim=True)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        """
        计算
        :param x: n x num_inputs向量，n表示样本数
        :param actions: 应该是nx1的向量，对应x[i]:actions[i]
        :return: log_softmax计算以后得到的选择actions对应的值，以及熵值
        """
        x = self(x)  # n x num_output

        log_probs = F.log_softmax(x, dim=1)  # n x num_output
        probs = F.softmax(x, dim=1)  # n x num_output
        # n x 1，聚合，从每一行中，选择出log概率最大的那个
        # print('before shape is {}'.format(log_probs))
        action_log_probs = log_probs.gather(1, actions)
        # print('action is {}'.format(actions))
        # print('after shape is {}'.format(action_log_probs))
        # 计算熵，由于没有绝对正确的label标记，因此，该熵值仅仅用于计算action space中所有选项的概率分布是否平均，越平均，说明越接近random，值越高
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        self.group1 = nn.Sequential(
            OrderedDict([
                ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
                ('squeeze_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group2 = nn.Sequential(
            OrderedDict([
                ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
                ('expand1x1_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group3 = nn.Sequential(
            OrderedDict([
                ('expand3x3', nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)),
                ('expand3x3_activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return torch.cat([self.group2(x),self.group3(x)], 1)


class SqueezeNet(FFPolicy):

    def __init__(self, in_channels: int=3, num_classes=1000, version=1.0):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, 1024, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )
        # TODO 在Squeezenet最后增加了linear评价器和分类器
        self.critic_linear = nn.Linear(1024, 1)  # value function 评价器
        self.cls_linear = Categorical(1024, num_classes)  # classification 分类器

        self.train()  # 设置成训练模式
        self.apply(weights_init)  # 初始化相关参数

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                if m is final_conv:
                    m.weight.data.normal_(0, 0.01)
                else:
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    u = math.sqrt(3.0 * gain / fan_in)
                    m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.classifier(x)
        x = x.view(x.size(0), 1024)

        return self.critic_linear(x), x
