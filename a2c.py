import os
import time
import numpy as np
import torch.nn as nn
import torch
from model import MobileNetV2

"""
1. openai的实现中，有部分变量是属于LSTM独有的内容，本工程初步实现不涉及到LSTM，因此全部都删掉了。
2. openai的实现中，optimizer使用的是RMSProp，但是由于tf和pytorch的RMSProp参数对应不上，为了降低风险，使用了SGD。因此相关的超参也做了调整
3. 
"""


class Model(object):

    def __init__(self, policy, nact, nenvs, nsteps, wave_num, k,
                 ent_coef=0.01, vf_coef=0.5, base_lr=7e-4,
                 weight_decay=1e-5, momentum=0.9, total_timesteps=int(80e6), lrschedule='linear'):
        """
        A2C的模型，除了超参的设置以外，主要是model（在openai的A2C中是包括step_model和train_model），openai的实现是基于tensorflow的，
        step_model和train_model 共享基础变量。而在Pytorch中直接以model实现，具体的功能区分，在model中实现。
        :param policy: 神经网络模型的类，用以生成model
        :param nact: number of action space (Discrete)
        :param nenvs: environments number。表示同时进行的游戏进程数
        :param nsteps: 进行一次训练，所需要游戏进行的步数
        :param wave_num: 链路波长数
        :param k: 表示ksp算法中k的取值，关系到action space
        :param ent_coef: entropy coefficient 熵系数，意义不明
        :param vf_coef: value function coefficient 价值函数系数，意义不明
        :param base_lr: 初始学习率
        :param momentum: 动量，SGD优化器的参数
        :param total_timesteps: 一共要进行的步骤
        :param lrschedule: 学习率调整方式
        """
        self.base_lr = base_lr
        self.total_timesteps = total_timesteps

        nbatch = nenvs * nsteps  # 一次训练的batch_size大小
        if policy == MobileNetV2:
            model = policy(in_channels=wave_num, num_classes=k*wave_num+1, t=6)
        else:
            raise NotImplementedError
        model = MobileNetV2()

        optimizer = torch.optim.SGD(params=model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

        torch.optim.lr_scheduler.StepLR()

    def train(self, obs, rewards, dones, actions, values):
        """

        :param obs:
        :param rewards:
        :param dones:
        :param actions:
        :param values:
        :return:
        """





    def adjust_learning_rate(self, optimizer, epoch, mode):
        """
        以不同方式调整学习率
        """
        if mode == 'linear':
            # 线性调整学习率
            lr = self.base_lr * (1 - epoch/self.total_timesteps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            raise NotImplementedError