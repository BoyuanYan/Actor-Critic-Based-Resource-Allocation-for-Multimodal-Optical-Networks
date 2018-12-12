from RwaNet import RwaNetwork
import numpy as np
from networkx import shortest_simple_paths
import random
from args import args
import utils

modes = ['alg', 'learning']



# 重新开启一轮游戏
INIT = 0
# 该时间点没有业务到达，可能有业务离去（取决于事件排序）
NOARRIVAL_NO    =   1  # 选择No-Action
NOARRIVAL_OT    =  -1  # 选择其他RW选项
# 该时间点有业务到达（可能同时有业务离去），但是没有可达RW选项
ARRIVAL_NOOP_NO =   args.punish  # 选择No-Action
ARRIVAL_NOOP_OT =   args.punish  # 选择其他RW选项
# 该时间点有业务到达（可能同时有业务离去），并且有可达RW选项
ARRIVAL_OP_OT   = args.reward  # 选择可达的RW选项
ARRIVAL_OP_NO   = args.punish  # 选择不可达或者No-Action


class Service(object):
    def __init__(self, index: int, src: str, dst: str,
                 arrival_time: int, leave_time: int):
        super(Service, self).__init__()
        self.index = index
        self.src = src
        self.dst = dst
        self.arrival_time = arrival_time
        self.leave_time = leave_time

    def add_allocation(self, path: list, wave_index: int):

        self.path = path
        self.wave_index = wave_index


def cmp(x, y):
    if x[0] < y[0]:
        return -1
    if x[0] > y[0]:
        return 1
    return 0


class RwaGame(object):
    """
    RWA game, 模仿gym的实现
    """

    def __init__(self, net_config: str, wave_num: int, rou: float, miu: float,
                 max_iter: int, k: int, mode: str, img_width: int, img_height: int,
                 weight, step_over: str='one_time'):
        """

        :param net_config: 网络配置文件
        :param wave_num: 链路波长数，CWDM是40， DWDM是80
        :param rou: 平均隔多少时间单位到达一条业务
        :param miu: 一条业务平均会在网络中存在多少时间单位后会离去
        :param max_iter: 一次episode中最大的轮数，即一次仿真的最大业务数
        :param mode: 模式，分为alg和learning两种，前者表示使用ksp+firstfit分配，后者表示使用rl算法学习
        :param img_width: 游戏界面的宽度
        :param img_height: 游戏界面的高度
        :param step_over: 步进的模式，one_time表示每调用一次step，执行一个时间步骤；one_service表示每调用一次step，执行到下一个service到达的时候。
        """
        super(RwaGame, self).__init__()
        print('创建RWA Game')
        self.net_config = net_config
        self.wave_num = wave_num
        self.img_width = img_width
        self.img_height = img_height
        self.weight = weight
        self.rou = rou
        self.miu = miu
        self.erl = miu / rou
        self.max_iter = max_iter
        self.k = k
        self.NO_ACTION = k*wave_num
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError("wrong mode parameter.")
        # 一旦游戏开始，iter和time都指向当前的event下标和时间点。
        self.event_iter = 0
        self.time = 0
        self.net = RwaNetwork(self.net_config, wave_num=self.wave_num)
        self.services = {}
        self.events = []  # time_point, service_index, is_arrival_event
        self.step_over = step_over

    def gen_src_dst(self):
        nodes = list(self.net.nodes())
        assert len(nodes) > 1
        src_index = random.randint(0, len(nodes)-1)
        dst_index = random.randint(0, len(nodes)-1)
        while src_index == dst_index:
            dst_index = random.randint(0, len(nodes)-1)
        return nodes[src_index], nodes[dst_index]

    def again(self):
        """
        清空所有的状态缓存，将环境重置回开始之前的状态，但是业务序列不变
        :return:
        """
        self.event_iter = 0
        self.time = 0
        ss = {}
        self.net = RwaNetwork(self.net_config, wave_num=self.wave_num)

        for val in self.services.values():
            serv = Service(val.index, val.src, val.dst, val.arrival_time, val.leave_time)
            ss[val.index] = serv
        self.services = ss

        # 返回第一个业务请求的状态
        src, dst = self.services[0].src, self.services[0].dst
        observation = self.net.gen_img(self.img_width, self.img_height, src, dst, self.mode)
        reward = INIT
        done = False
        info = None
        self.time = self.services[0].arrival_time
        return observation, reward, done, info

    def reset(self):
        """
        reset environment
        :return:
        """
        self.event_iter = 0
        self.time = 0
        self.services = {}
        self.events = []
        self.net = RwaNetwork(self.net_config, wave_num=self.wave_num)

        base_time = 0
        rand_val = int(random.random()*1000000000)
        np.random.seed(rand_val)
        for base_index in range(self.max_iter):
            src, dst = self.gen_src_dst()
            arrival = np.random.poisson(lam=self.rou) + base_time + 1
            leave = np.random.poisson(lam=self.miu) + arrival + 1
            self.services[base_index] = Service(base_index, src, dst, arrival, leave)
            self.events.append([arrival, base_index, True])
            self.events.append([leave, base_index, False])

            base_time = arrival
        self.events.sort(key=lambda time: time[0])

        # 返回第一个业务请求的状态
        src, dst = self.services[0].src, self.services[0].dst
        observation = self.net.gen_img(self.img_width, self.img_height, src, dst, self.mode)
        reward = INIT
        done = False
        info = None
        self.time = self.services[0].arrival_time
        return observation, reward, done, info

    def render(self):
        """
        渲染当前环境，返回当前环境的图像
        :return:
        """
        raise NotImplementedError

    def step(self, action) -> [object, float, bool, dict]:
        """
        根据self.step_over的设置，执行不同的step操作
        :param action:
        :return:
        """
        if self.step_over.startswith('one_time'):
            return self.step_one_time(action=action)
        elif self.step_over.startswith('one_service'):
            return self.step_one_service(action=action)

    def step_one_service(self, action) -> [object, float, bool, dict]:
        """
        在当前业务点self.services，执行行为action，获取reward，并且转向下一个到达的service
        :param action: 所采取的行为，默认是int类型。如果取值为-1，表示暂停游戏，游戏状态不发生变化
        :return:
        """

        if action is -1:
            return np.array([None, None]), 0, True, None

        # 先处理当前到达的业务以及采取行为action获取的reward
        # 下一步如果没有到达业务，则这一步的obs为None。相关的处理在下面while里面有判断
        obs, reward, done, info = self.step_one_time(action=action, obs_for_invalid_time=False)
        if done:
            return obs, reward, done, info
        obs_none = 0
        obs_times = 0
        while self.events[self.event_iter][0] > self.time:
            # 如果下一个时间点没有到达的业务，则action选择No-Action。reward保持第一个action的reward，其他值跟随时间推进

            obs, _, done, info = self.step_one_time(action=self.k * self.wave_num, obs_for_invalid_time=False)
            obs_times += 1
            if obs is None:
                obs_none += 1

            if done:
                return obs, reward, done, info

        return obs, reward, done, info

    def step_one_time(self, action, obs_for_invalid_time: bool=True) -> [object, float, bool, dict]:
        """
        在当前时间点self.time,执行行为action，获取reward，并且转向下一个时间点。
        :param action: 所采取的行为，默认是int类型。如果取值为-1，表示暂停游戏，游戏状态不变化。
        :param obs_for_invalid_time: 表示如果下一个状态没有到达业务，即无用时间（invalid time），则无需生成obs图像返回。这样是为了在
        step_one_service模式下，可以加快程序运行速度。不然运行速度会相差args.rou倍。
        :return:
        """
        if action is -1:
            return np.array([None, None]), 0, True, None

        done = False
        info = False  # info表示本次是否处理业务到达事件
        # 首先，判断当前的处境，该时间点是否有业务到达或者离去，如果有，有几个
        # print('event id is: {}, total events is {}'.format(self.event_iter, len(self.events)))
        if self.events[self.event_iter][0] > self.time:
            # 如果该时间点没有到达或者离去的业务，则action选什么都无所谓
            if action == self.k * self.wave_num:
                # 如果主动阻塞
                reward = NOARRIVAL_NO
            else:
                # 如果选择其他行为，虽然没用，但是还是要惩罚
                reward = NOARRIVAL_OT
            self.time += 1  # 时间推进，事件已经指向下一个要处理的下标，暂时不动

        elif self.events[self.event_iter][0] == self.time:
            # 如果该时间点恰巧有业务到达或者离去
            # TODO 处理当前时间点的业务，并且将self.event_iter指向下一个要处理的事件
            if self.events[self.event_iter][2] is False:
                # 如果该时间点第一个事件是业务离去，则说明处理逻辑出了问题，抛错
                raise RuntimeError("执行action遇到业务离去事件，该事件应该在action之前被处理！")
            else:
                # 如果该时间点第一个事件是业务到达，则按照action选择处理
                # print("process arrival event")
                # print("event id is {}".format(self.event_iter))
                info = True  # info中包含了本次action所处理的事件是否是业务到达事件
                ser = self.services[self.events[self.event_iter][1]]
                reward = self.exec_action(action, ser)
                # TODO 此处做一个有争议的决策，如果处理的到达业务是最后一个到达业务的话，则本游戏直接结束。因为后续只能是业务释放
                if self.events[self.event_iter][1] == (self.max_iter-1):
                    observation = self.net.gen_img(self.img_width, self.img_height, None, None, self.mode)
                    done = True
                    return observation, reward, done, info

                self.event_iter += 1
                while self.events[self.event_iter][0] == self.time:
                    # 该时间点处理完业务到达以后，后续还有业务离去事件(不可能同一个时间点有多个业务到达)
                    assert self.events[self.event_iter][2] is False
                    leave_service = self.services[self.events[self.event_iter][1]]
                    # print('process leave event')
                    if hasattr(leave_service, 'path'):  # 如果该业务分配时候成功了
                        self.net.set_wave_state(wave_index=leave_service.wave_index,
                                                nodes=leave_service.path,
                                                state=True,
                                                check=True)
                    else:  # 如果该业务分配时候失败了
                        pass
                    self.event_iter += 1
                self.time += 1  # 时间推进，事件也推进到下一个要处理的下标
        else:
            # 如果该时间点之前还有没处理完的业务
            raise EnvironmentError("时间推进过程中，有漏掉未处理的事件")

        # 其次，判断是否已经走到了头
        if self.event_iter == len(self.events):
            # 如果已经把事件全部处理完，
            done = True
            observation = self.net.gen_img(self.img_width, self.img_height, None, None, self.mode)
            # print('已经走到尽头')
            return observation, reward, done, info

        # 第三，开始进行下一状态的处理。之前的处理中，时间和事件都已经推进到下一个单位了
        if self.events[self.event_iter][0] > self.time:
            # 如果该时间点没有到达或者离去的业务
            if obs_for_invalid_time:
                # 如果要求返回无效时间的拓扑图像
                observation = self.net.gen_img(self.img_width, self.img_height, None, None, self.mode)
            else:
                # 如果不要求返回无效时间的拓扑图像
                observation = None
        elif self.events[self.event_iter][0] == self.time:
            # 如果该时间点恰巧有业务到达或者离去
            # TODO 处理当前时间点排在到达业务之前的离去业务，并将self.event_iter指向下一个要处理的事件
            while self.events[self.event_iter][2] is False and self.events[self.event_iter][0] == self.time:
                leave_service = self.services[self.events[self.event_iter][1]]
                # print('process leave event')
                if hasattr(leave_service, 'path'):  # 如果该业务分配时候成功了
                    self.net.set_wave_state(wave_index=leave_service.wave_index,
                                            nodes=leave_service.path,
                                            state=True,
                                            check=True)
                else:  # 如果该业务分配时候失败了
                    pass
                self.event_iter += 1
                if self.event_iter == len(self.events):
                    # 如果已经把事件全部处理完，
                    # print('已经走到尽头')
                    done = True
                    observation = self.net.gen_img(self.img_width, self.img_height, None, None, self.mode)
                    return observation, reward, done, info

            if self.events[self.event_iter][0] == self.time:
                # 这时候只能是到达业务了，到达业务不可能是最后一个事件。
                assert self.events[self.event_iter][2] is True
                service = self.services[self.events[self.event_iter][1]]
                src, dst = service.src, service.dst
                observation = self.net.gen_img(self.img_width, self.img_height, src, dst, self.mode)
            else:
                # 表示下一个时间点没有到达业务事件
                if obs_for_invalid_time:
                    # 如果需要没有到达业务时间点的图像
                    observation = self.net.gen_img(self.img_width, self.img_height, None, None, self.mode)
                else:
                    # 如果不需要没有到达业务时间点的图像
                    observation = None
        else:
            # 如果该时间点之前还有没处理完的业务
            raise EnvironmentError("时间推进过程中，还有漏掉未处理的事件")

        return observation, reward, done, info

    def exec_action(self, action: int, service: Service) -> float:
        """
        对到达的业务service，执行行为action，并且返回reward。
        如果分配业务成功，则注意给service对象加入分配方案
        :param action:
        :param service:
        :return: reward
        """
        path_list = self.k_shortest_paths(service.src, service.dst)
        is_avai, _, _ = self.net.exist_rw_allocation(path_list)
        if action == self.NO_ACTION:
            if is_avai:
                # 如果存在可分配的方案，但是选择了NO-ACTION
                return ARRIVAL_OP_NO
            else:
                # 如果不存在可分配的方案，选择了NO-ACTION
                return ARRIVAL_NOOP_NO
        else:
            if is_avai:
                route_index = action // (self.k*self.wave_num)
                wave_index = action % (self.k*self.wave_num)
                if self.net.is_allocable(path_list[route_index], wave_index):
                    # 如果存在可分配方案，并且指定的分配方案是可行的
                    self.net.set_wave_state(wave_index=wave_index, nodes=path_list[route_index],
                                            state=False, check=True)
                    service.add_allocation(path_list[route_index], wave_index)
                    return ARRIVAL_OP_OT
                else:
                    # 如果存在可分配方案，并且指定的分配方案是不可行的
                    return ARRIVAL_OP_NO
            else:
                # 如果不存在可分配的方案，但是选择了非NO-ACTION的选项
                return ARRIVAL_NOOP_OT

    def k_shortest_paths(self, source, target):
        """
        如果源宿点是None，则返回len为1的None数组
        :param source:
        :param target:
        :return:
        """
        if source is None:
            return [None]
        generator = shortest_simple_paths(self.net, source, target, weight=self.weight)
        rtn = []
        index = 0
        for i in generator:
            index += 1
            if index > self.k:
                break
            rtn.append(i)
        return rtn
