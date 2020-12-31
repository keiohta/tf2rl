import cv2
from tf2rl.envs.frame_stack_wrapper import FrameStack


class DMCWrapper(FrameStack):
    """
    Wrapper class to visualize DMC environments.
    """
    def __init__(self, env, k, obs_shape, wait_ms=1000 / 30, **kwargs):
        super().__init__(env, k, obs_shape, **kwargs)
        self._wait_ms = int(wait_ms)

    def render(self):
        rgb_array = self.env.render(mode="rgb_array")
        cv2.imshow("DMC Visualizer", cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB))
        cv2.waitKey(self._wait_ms)


if __name__ == "__main__":
    import dmc2gym

    env = DMCWrapper(
        dmc2gym.make(
            domain_name="finger",
            task_name="spin",
            visualize_reward=False,
            from_pixels=True,
            height=100,
            width=100,
            frame_skip=2,
            channels_first=False),
        obs_shape=(100, 100, 9),
        k=3,
        channel_first=False)

    for _ in range(10):
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
            env.render()
