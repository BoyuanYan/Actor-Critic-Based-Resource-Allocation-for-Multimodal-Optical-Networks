import torch
from torch import zeros, ones
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):

    def __init__(self, num_steps: int, num_processes: int, obs_shape: tuple, action_shape: int):
        """
        存储形式是[steps，workers, Channels, Height, Width]
        :param num_steps: 进行一次训练所需要游戏进行的步骤数
        :param num_processes: 同时运行的游戏进程数
        :param obs_shape: observation space的shape
        :param action_shape: action space的size
        """
        self.observations = zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = zeros(num_steps, num_processes, 1)
        self.value_preds = zeros(num_steps + 1, num_processes, 1)
        self.returns = zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = zeros(num_steps, num_processes, 1)
        self.actions = zeros(num_steps, num_processes, action_shape).long()
        self.masks = ones(num_steps + 1, num_processes, 1)  # 游戏是否结束的标记

    def cuda(self):
        """
        呵呵，感觉这个函数永远不会被用到了。。。
        :return:
        """
        self.observations = self.observations.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_obs, action, action_log_prob, value_pred, reward, mask):
        self.observations[step + 1].copy_(current_obs)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def after_update(self):
        """
        执行梯度更新以后，要刷新所有本对象中缓存的内容，首先observations和masks是下一步的结果，因此将梯度更新以前的最后一个obs和mask，放到梯度
        更新以后的第一个位置。此外，其他变量以及observation和masks的后续更新，都通过insert准确在step位置插入，完成更新。
        :return:
        """
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau=None):
        """
        计算returns。在不考虑use_gae,tau的情况下，递推公式如下：
        $$return_i = return_{i+1} \cdot \gamma \cdot mask_{i+1} + reward_i$$
        :param next_value: 下一个值，use_gae为False的时候，表示下一个return。
        :param use_gae: 相关解释，可见https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/issues/49
        :param gamma: 计算reward的时候，评估当前选择对之后状态变化的影响的折扣因子
        :param tau: gae模式下用到的参数，不懂
        :return: 计算returns值
        """
        if use_gae:
            # 把最后一个值改成next_value
            raise NotImplementedError
            # self.value_preds[-1] = next_value
            # gae = 0
            # for step in reversed(range(self.rewards.size(0))):
            #     delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            #     gae = delta + gamma * tau * self.masks[step + 1] * gae
            #     self.returns[step] = gae + self.value_preds[step]
        else:
            # 把returns第一个维度的末尾值对应的矩阵，全部改成next_value的值
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices)

            if advantages.is_cuda:
                indices = indices.cuda()

            observations_batch = self.observations[:-1].view(-1,
                                        *self.observations.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ
