import os
import torch
import torch.distributed as dist
from torch.nn import Module


class DistModule(Module):

    def __init__(self, module):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)
        dist._clear_group_cache()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        dist._clear_group_cache()
        super(DistModule, self).train(mode)
        self.module.train(mode)

    def act(self, inputs, deterministic: bool):
        """
        以deterministic的方式，根据网络计算结果采取action，并且评估该状态的价值函数的值。
        :param inputs:
        :param deterministic: True表示取概率最高的action值，False表示按照概率分布取action值。
        :return:
        """
        return self.module.act(inputs, deterministic)

    def evaluate_actions(self, inputs, actions):
        """
        评估在状态inputs下，采取行为actions的价值。
        :param inputs:
        :param actions:
        :return:
        """
        return self.module.evaluate_actions(inputs, actions)


def average_gradients(model):
    """ average gradients """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)


def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def dist_init(port):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    print(addr)

    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size
