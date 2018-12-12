from multiprocessing import Process, Pipe
import numpy as np
import os


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            # 如果本轮游戏结束，则开启新一轮游戏
            if done:
                print("游戏结束")
                ob, reward, _, info = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob, reward, done, info = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'net':
            remote.send(env.net)
        elif cmd == 'exist_rw_allocation':
            is_exist, route_index, wave_index = env.net.exist_rw_allocation(data)
            remote.send((is_exist, route_index, wave_index))
        elif cmd == 'k_shortest_paths':
            paths = env.k_shortest_paths(data[0], data[1])
            remote.send(paths)
        else:
            raise NotImplementedError


class SubprocEnv(object):
    """
    RWA游戏的多进程实现，用于A2C，替代replay memory的作用。
    """

    def __init__(self, envs: list):
        """
        多进程启动RWA游戏
        :param envs: list of envs
        """
        self.waiting = False
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.processes = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                          for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, envs)]
        for p in self.processes:
            p.daemon = True  # 如果主进程挂掉，保证子进程不受影响。但这是为啥捏？
            p.start()  # 开启子进程
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        """
        异步在多进程中执行行为，无返回值。返回值在step_wait函数中得到
        :param actions:
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        """
        异步在多进程中或许执行结果，一定要在step_async后调用
        :return:
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)

    def exist_rw_allocation(self, path_list):
        """
        判断在path_list中是否有可行的分配方案
        :param path_list:
        :return:
        """
        for remote, paths in zip(self.remotes, path_list):
            remote.send(('exist_rw_allocation', paths))
        results = [remote.recv() for remote in self.remotes]
        exist, path_index, wave_index = zip(*results)
        return np.stack(exist), np.stack(path_index), np.stack(wave_index)

    def k_shortest_paths(self, src_dst):
        """
        计算ksp路径
        :param src_dst:
        :return:
        """
        for remote, sd in zip(self.remotes, src_dst):
            remote.send(('k_shortest_paths', sd))
        result = [remote.recv() for remote in self.remotes]
        return result

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
        self.closed = True


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
