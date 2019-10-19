"""
vector単位での操作をする環境。
マルチプロセスで動作し、プロセス間通信は共有メモリを通じて行う。
同期はなるべくLockベースの方法で行う。

gymの環境と異なり, doneとreach_limitを区別する。

"""

import logging
import time
import multiprocessing as mp
import numpy as np
import ctypes
import cloudpickle

import gym


_NP_TO_CT = {np.dtype("float32"): ctypes.c_float,
             np.dtype("int64"): ctypes.c_int64,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}

class SectionArea:
    def __init__(self, shared : mp.Array, dtype, shape : tuple):
        """プロセスの担当エリアを表現する
        [idx_start, idx_end]の範囲を担当する
        """
        self.shared = shared
        self.shape = shape
        self.dtype = dtype

    def get_np(self):
        return np.reshape(np.frombuffer(self.shared, dtype=self.dtype), self.shape)


def worker(make_env, obs_areas, idx_start, idx_end,
           barrier_start, barrier_finish, seeds):
    """

    :param mp.Array shared_obs:
    :param mp.Condition cond_start:
    :param barrier_finish:
    """
    obs_area, next_obs_area, action_area, reward_area, done_area, reach_limit_area = obs_areas

    make_env = cloudpickle.loads(make_env)
    list_envs = []
    list_steps = []
    max_episode_steps = 0

    np_next_obs = next_obs_area.get_np()

    for idx in range(idx_start, idx_end):
        env = make_env()

        if seeds[idx-idx_start] is not None:
            env.seed(int(seeds[idx-idx_start]))
        max_episode_steps = env._max_episode_steps
        obs = env.reset()
        np_next_obs[idx] = obs
        list_envs.append(env)
        list_steps.append(0)

    barrier_start.wait()

    np_obs = obs_area.get_np()
    # reset考慮後の次のobs
    np_action = action_area.get_np()
    np_reward = reward_area.get_np()
    np_done = done_area.get_np()
    np_reach_limit = reach_limit_area.get_np()

    while True:
        barrier_start.wait()
        actions = np_action[idx_start:idx_end]

        # false sharingの影響は小さそう
        for idx in range(idx_start, idx_end):
            rel_idx = idx - idx_start
            action = actions[rel_idx]
            next_state, reward, done, _ = list_envs[rel_idx].step(action)
            list_steps[rel_idx] += 1

            if list_steps[rel_idx] == max_episode_steps:
                reach_limit = True
            else:
                reach_limit = False

            np_obs[idx] = next_state
            np_reward[idx] = reward
            np_done[idx] = done
            np_reach_limit[idx] = reach_limit

            # プロセス数を増やすと, doneが遅い奴が足を引っ張る
            if done:
                next_state = list_envs[rel_idx].reset()
                list_steps[rel_idx] = 0

            np_next_obs[idx] = next_state

        barrier_finish.wait()


class ShmemEnv(object):
    """
    共有メモリを用いた通信実験設備
    """

    def __init__(self, make_env, num_processes, envs_per_process, observation_space, action_space, seeds=None):
        """

        :param env_fns:
        :param gym.spaces.Box observation_space:
        :param gym.spaces.Box or gym.spaces.Discrete action_space:

        """
        self.logger = logging.getLogger(__name__)

        # processをspawnで立ち上げる。interpreterが新規に立つので起動は遅いが、tensorflowのメモリ空間を共有するとめんどうそうなので
        ctx = mp.get_context("spawn")

        num_envs = num_processes * envs_per_process
        # shape=(num_envs, observation_space.shape)
        obs_shape = (num_envs,) + observation_space.shape
        obs_size = int(np.prod(obs_shape))
        shared_obs = ctx.Array(_NP_TO_CT[observation_space.dtype], obs_size, lock=False)
        shared_next_obs = ctx.Array(_NP_TO_CT[observation_space.dtype], obs_size, lock=False)
        self.obs_area = SectionArea(shared_obs, dtype=observation_space.dtype, shape=obs_shape)
        # reset考慮後のobservation
        self.next_obs_area = SectionArea(shared_next_obs, dtype=observation_space.dtype, shape=obs_shape)

        action_shape = (num_envs,) + action_space.shape
        action_size = int(np.prod(action_shape))
        shared_action = ctx.Array(_NP_TO_CT[action_space.dtype], action_size, lock=False)
        self.action_area = SectionArea(shared_action, dtype=action_space.dtype, shape=action_shape)

        shared_reward = ctx.Array(ctypes.c_float, num_envs, lock=False)
        self.reward_area = SectionArea(shared_reward, dtype=np.float32, shape=(num_envs,))

        shared_done = ctx.Array(ctypes.c_bool, num_envs, lock=False)
        self.done_area = SectionArea(shared_done, dtype=np.bool, shape=(num_envs,))
        shared_reach = ctx.Array(ctypes.c_bool, num_envs, lock=False)
        self.reach_area = SectionArea(shared_reach, dtype=np.bool, shape=(num_envs,))

        # step開始用のbarrier (これによって指示を読み取る)
        self.barrier_start = ctx.Barrier(parties=num_processes+1)
        self.barrier_finish = ctx.Barrier(parties=num_processes+1)
        self.list_proc = []

        pickled_env = cloudpickle.dumps(make_env)
        for idx_process in range(num_processes):
            if seeds is None:
                cur_seed = [None] * envs_per_process
            else:
                cur_seed = seeds[idx_process]

            obs_areas = self.obs_area, self.next_obs_area, self.action_area, self.reward_area, self.done_area, self.reach_area

            args = (pickled_env,
                    obs_areas,
                    idx_process*envs_per_process, (idx_process+1)*envs_per_process,
                    self.barrier_start, self.barrier_finish, cur_seed)

            proc = ctx.Process(target=worker, args=args, daemon=True)
            proc.start()
            self.list_proc.append(proc)

        # reset結果が格納されるまでの待機
        self.barrier_start.wait()

        self.np_obs = self.obs_area.get_np()
        self.np_obs_next = self.next_obs_area.get_np()
        self.np_action = self.action_area.get_np()
        self.np_reward = self.reward_area.get_np()
        self.np_done = self.done_area.get_np()
        self.np_reach_limit = self.reach_area.get_np()

    def step(self, actions):
        np.copyto(self.np_action, actions)

        self.barrier_start.wait()
        self.barrier_finish.wait()

        # copyすべきかどうか考慮必要
        return self.np_obs.copy(), self.np_reward.copy(), self.np_done.copy(), self.np_reach_limit.copy()

    def observation(self):
        # タイミングによっては考慮不要
        return self.np_obs_next.copy()

    def close(self):
        for proc in self.list_proc:
            proc.terminate()


def main():
    def make_env():
        return gym.make("Pendulum-v0")

    env = make_env()
    vec_env = ShmemEnv(make_env, 4, 2, env.observation_space, env.action_space)
    obs = vec_env.observation()
    print(obs)
    action = env.action_space.sample()
    actions = np.expand_dims(action, axis=0)
    next_obs, reward, done, reach_limit = vec_env.step(actions)
    print(next_obs)


if __name__ == '__main__':
    main()
