import time

import numpy as np

from tf2rl.experiments.utils import frames_to_gif

NP_DTYPE = np.float64


def is_pos_def(x):
    """
    Check if the input matrix is positive definite
    :param x: matrix to be tested
    :return: True if `x` is positive definite
    """
    return np.all(np.linalg.eigvals(x) > 0)


def rollout(make_env, policy="zeros", max_steps=None):
    """Generate a trajectory by using the given policy.

    :param make_env: a function to make an environment
    :param policy: "zeros" uses zero-vector control. "random" generates control from uniform distribution.
    :param max_steps: if None, rollout while done is False.
    :return:
     (X, U, cost)
    """
    env = make_env()
    env.reset()
    dim_action = env.action_space.shape[0]

    X = [env.get_state_vector()]
    U = []
    cost = 0.

    while True:
        if policy == "zeros":
            u = np.zeros(dim_action, dtype=NP_DTYPE)
        elif policy == "ou" or policy == "random":
            random_action = np.random.uniform(low=-env.action_space.high[0],
                                              high=env.action_space.high[0],
                                              size=dim_action)
            if len(U) == 0 or policy == "random":
                u = np.copy(random_action)
            else:
                u += random_action
                u = np.clip(u, env.action_space.low, env.action_space.high)
        else:
            raise ValueError("Unknown policy : {}".format(policy))

        cost_state = env.cost_state()
        cost_control = env.cost_control(u)
        cur_cost = cost_state + cost_control

        _, _, done, _ = env.step(u)
        cost += cur_cost

        U.append(u)
        X.append(env.get_state_vector())

        if max_steps is not None and len(U) == max_steps:
            break

        if done:
            break

    return np.array(X, dtype=NP_DTYPE), np.array(U, dtype=NP_DTYPE), cost


def visualize_rollout(viewer_env, initial_state, U, save_movie, prefix):
    frames = []

    def render():
        if save_movie:
            frames.append(env.render(mode="rgb_array"))
        else:
            env.render()

    env = viewer_env
    env.reset()
    env.set_state_vector(initial_state)

    # Call render once before calling `env.render(mode="rgb_array"))` due to the known bug in openai/gym
    env.render()
    render()

    for u in U:
        env.step(u)
        render()
        time.sleep(1/30.)  # 30 FPS

    if save_movie:
        assert len(frames) > 0
        frames_to_gif(frames, prefix=prefix)
