import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gym.envs.mujoco.reacher import ReacherEnv

from tf2rl.algos.ilqg import ILQG
from tf2rl.envs.numerical_diff_dynamics import ILQGInterfaceEnv
from tf2rl.misc.ilqg_utils import visualize_rollout
from tf2rl.misc.initialize_logger import initialize_logger


class ReacherILQGEnv(ReacherEnv, ILQGInterfaceEnv):
    """
    Modified MuJoCo Reacher env for iLQGcomputation.
    https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py
    """

    def __init__(self):
        ReacherEnv.__init__(self)
        ILQGInterfaceEnv.__init__(self)

    def step(self, a):
        next_obs, rew, done, env_info = ReacherEnv.step(self, a)
        # Recompute xpos for get_body_com("fingertip") from qpos
        # self.sim.forward()

        return next_obs, rew, done, env_info

    def cost_state(self):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        cost_dist = np.linalg.norm(vec)
        return cost_dist

    def cost_control(self, control):
        return np.square(control).sum()

    def set_state_vector(self, state):
        msg = "Shape of the state to be set is {}, expected {}".format(state.shape, (self.model.nq + self.model.nv,))
        assert state.shape == (self.model.nq + self.model.nv,), msg
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:]
        self.set_state(qpos, qvel)

    def get_state_vector(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    @property
    def dim_state(self):
        return self.model.nq + self.model.nv

    @property
    def dim_control(self):
        return 2


class PendulumILQGEnv(PendulumEnv, ILQGInterfaceEnv):
    """
    Modified gym Pendulum env for iLQG computation.
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """

    def __init__(self):
        PendulumEnv.__init__(self)
        ILQGInterfaceEnv.__init__(self)

    def cost_state(self):
        th, thdot = self.state
        return angle_normalize(th) ** 2 + .1 * thdot ** 2

    def cost_control(self, control):
        return .001 * np.square(control).sum()

    def set_state_vector(self, state):
        assert state.shape == (2,)  # (\theta, \dot{\theta})
        self.state = state

    def get_state_vector(self):
        return self.state

    @property
    def dim_state(self):
        return 2

    @property
    def dim_control(self):
        return 1


def main():
    logger = initialize_logger(save_log=False)

    parser = ILQG.get_argument()
    parser.add_argument("--save-movie", action="store_true")
    parser.add_argument("--visualize-interval", type=int, default=5)
    parser.add_argument("--env-name", choices=["Pendulum", "Reacher"], default="Reacher")
    args = parser.parse_args()

    if args.env_name == "Pendulum":
        make_env = PendulumILQGEnv
        args.horizon = 100
    elif args.env_name == "Reacher":
        make_env = ReacherILQGEnv
        args.horizon = 30

    ilqg = ILQG(make_env, horizon=args.horizon)
    ilqg.initialize()

    logger.info("Initial trajectory: T = {} cost = {:.5f}".format(len(ilqg.U), ilqg.cost))
    viewer_env = make_env()
    visualize_rollout(viewer_env=viewer_env, initial_state=ilqg.X[0], U=ilqg.U, save_movie=args.save_movie,
                      prefix="{}_{}".format(args.env_name, 0))

    for i in range(args.max_iter_optimization):
        ilqg.optimize(max_iter=args.max_iter_each_step)
        logger.info("Iter {}: cost = {:.5f}".format(i + 1, ilqg.cost))
        if (i + 1) % args.visualize_interval == 0:
            visualize_rollout(viewer_env=viewer_env, initial_state=ilqg.X[0], U=ilqg.U, save_movie=args.save_movie,
                              prefix="{}_{}".format(args.env_name, i + 1))


if __name__ == '__main__':
    main()
