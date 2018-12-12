from matplotlib import use
use('Agg')
from args import args
import os
from model import MobileNetV2, SimpleNet, AlexNet, SqueezeNet, SimplestNet
from Service import RwaGame, ARRIVAL_OP_OT
import torch
from torch.autograd import Variable

"""
本文件主要用于模型结果对比
"""


def main():
    global actor_critic, directory, weight
    num_cls = args.wave_num * args.k + 1  # 所有的路由和波长选择组合，加上啥都不选

    if args.append_route.startswith("True"):
        channel_num = args.wave_num+args.k
    else:
        channel_num = args.wave_num

    # 解析weight
    if args.weight.startswith('None'):
        weight = None
    else:
        weight = args.weight

    # CNN学习模式下，osb的shape应该是CHW
    assert args.mode.startswith('learning')
    # 模型初始化
    if args.cnn.startswith('mobilenetv2'):
        actor_critic = MobileNetV2(in_channels=channel_num, num_classes=num_cls, t=6)
    elif args.cnn.startswith('simplenet'):
        actor_critic = SimpleNet(in_channels=channel_num, num_classes=num_cls)
    elif args.cnn.startswith('simplestnet'):
        actor_critic = SimplestNet(in_channels=channel_num, num_classes=num_cls)
    elif args.cnn.startswith('alexnet'):
        actor_critic = AlexNet(in_channels=channel_num, num_classes=num_cls)
    elif args.cnn.startswith('squeezenet'):
        actor_critic = SqueezeNet(in_channels=channel_num, num_classes=num_cls, version=1.0)
    else:
        raise NotImplementedError

    times = 1  # 重复次数
    prefix = "trained_models"
    directory = os.path.join(prefix, 'a2c', args.cnn, args.step_over)

    if args.comp.startswith("states"):
        all_states_comp()
    elif args.comp.startswith("random"):
        random_comp(times=times)
    elif args.comp.startswith("None"):
        raise ValueError("Wrong call for this script")
    else:
        raise NotImplementedError


def all_states_comp():
    """
    遍历所有可能的状态，对比ksp算法和指定算法的选择差异
    :return:
    """
    raise NotImplementedError


def random_comp(times: int=1):
    """
    用随机生成的业务序列进行算法对比.
    结果中会显示网络状态与业务请求，以及根据不同策略执行的选择
    :return:
    """
    env = RwaGame(net_config=args.net, wave_num=args.wave_num, rou=args.rou, miu=args.miu,
                  max_iter=args.max_iter, k=args.k, mode=args.mode, img_width=args.img_width,
                  img_height=args.img_height, weight=weight, step_over=args.step_over)

    for model_file in reversed(sorted(os.listdir(directory), key=lambda item: int(item.split('.')[0]))):
        model_file = os.path.join(directory, model_file)
        print("evaluate model {}".format(model_file))
        params = torch.load(model_file)
        actor_critic.load_state_dict(params['state_dict'])
        actor_critic.eval()  # 切换模式很重要

        print("model loading is finished")
        for t in range(times):
            total_reward, total_services, allocated_services = 0, 0, 0
            env.mode = "learning"
            obs, reward, done, info = env.reset()
            while not done:
                inp = Variable(torch.Tensor(obs).unsqueeze(0), volatile=True)  # 禁止梯度更新
                value, action, action_log_prob = actor_critic.act(inputs=inp, deterministic=True)  # 确定性决策
                action = action.data.numpy()[0]
                obs, reward, done, info = env.step(action=action[0])
                total_reward += reward
                if reward == ARRIVAL_OP_OT:
                    allocated_services += 1
                if args.step_over.startswith('one_time'):
                    if info:
                        total_services += 1
                elif args.step_over.startswith('one_service'):
                    total_services += 1
                else:
                    raise NotImplementedError
            bp = (total_services-allocated_services) / total_services
            print("{}: allocated services is {}, total services is {}, bp is {}"
                  .format(model_file, allocated_services,total_services, bp))
            # 开始计算ksp算法的对应结果
            env.mode = "alg"
            total_reward, total_services, allocated_services = 0, 0, 0
            obs, reward, done, info = env.again()
            while not done:
                path_list = env.net.k_shortest_paths(obs[0], obs[1])
                exist, path_index, wave_index = env.net.exist_rw_allocation(path_list)
                action = args.wave_num * args.k  # 主动阻塞
                if obs[0] is not None:
                    total_services += 1
                    # 如果当前时间有业务到达
                    if exist:
                        # 如果有可用分配方案
                        action = path_index*args.wave_num + wave_index
                        allocated_services += 1
                obs, reward, done, info = env.step(action)
            bp = (total_services-allocated_services) / total_services
            print("ksp+First-Fit: allocated services is {}, total services is {}, bp is {}"
                  .format(allocated_services,total_services, bp))


if __name__ == "__main__":
    main()
