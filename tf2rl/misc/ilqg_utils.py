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
