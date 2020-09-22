import argparse

import numpy as np

from tf2rl.envs.numerical_diff_dynamics import NumericalDiffDynamics
from tf2rl.misc.ilqg_utils import NP_DTYPE, is_pos_def
from tf2rl.misc.initialize_logger import initialize_logger


class ILQG:
    def __init__(
            self,
            make_env,
            policy="ou",
            max_iter=30,
            mu=1.5,
            min_mu=1e-8,
            max_mu=1e16,
            tol_cost=1e-7,
            horizon=100):
        """Generate locally-optimal controls using iterative LQG.
        Variable names follow the symbols in
        "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization".

        :param mu: coefficient of diagonal regularization to make the Hessian positive-definite
        :param min_mu: minimum coefficient for regularization
        :param max_mu: maximum coefficient for regularization
        :param tol_cost: tolerance cost to stop iLQG optimization
        """
        self._env = make_env()
        self._make_env = make_env
        self._policy = policy
        self._max_iter = max_iter
        self._logger = initialize_logger(save_log=False)
        self._tol_cost = tol_cost
        self._mu = mu
        self._min_mu = min_mu
        self._max_mu = max_mu
        self.horizon = horizon
        # Coefficients for improved line search in Eq.12
        self._alphas = np.power(10, np.linspace(0, -3, 21))
        self._X, self._U, self._cost = None, None, None

    def initialize(self, initial_state=None):
        if initial_state is not None:
            self._env.set_state_vector(initial_state)
        self._X, self._U, self._cost = self.rollout(initial_state)

    def rollout(self, initial_state=None):
        """Generate a trajectory by using the given policy.

        :param make_env: a function to make an environment
        :param policy: "zeros" uses zero-vector control. "random" generates control from uniform distribution.
        :param max_steps: if None, rollout while done is False.
        :return:
         (X, U, cost)
        """
        if initial_state is not None:
            self._env.set_state_vector(initial_state)
        else:
            self._env.reset()

        X = [self._env.get_state_vector()]
        U = []
        cost = 0.

        while True:
            if self._policy == "zeros":
                u = np.zeros(self._env.dim_control, dtype=NP_DTYPE)
            elif self._policy == "random":
                u = np.random.uniform(low=self._env.action_space.low, high=self._env.action_space.high,
                                      size=self._env.dim_control)
            elif self._policy == "ou":
                random_u = np.random.uniform(low=-self._env.action_space.high[0], high=self._env.action_space.high[0],
                                             size=self._env.dim_control)
                if len(U) == 0:
                    u = np.copy(random_u)
                else:
                    u += random_u
                    u = np.clip(u, self._env.action_space.low, self._env.action_space.high)
            else:
                raise ValueError("Unknown policy : {}".format(self._policy))

            cur_cost = self._env.cost_state() + self._env.cost_control(u)

            _, _, done, _ = self._env.step(u)
            cost += cur_cost

            U.append(u)
            X.append(self._env.get_state_vector())

            if self.horizon is not None and len(U) == self.horizon:
                break

            if done:
                break

        return np.array(X, dtype=NP_DTYPE), np.array(U, dtype=NP_DTYPE), cost

    def optimize(self, max_iter=None):
        if max_iter is None:
            max_iter = self._max_iter
        dynamics = NumericalDiffDynamics(self._make_env)
        X, U, cost = self._X, self._U, self._cost

        for _ in range(max_iter):
            # Compute open-loop term k and feed-back gain term K
            k, K = self.backward(X, U, dynamics)

            # Compute a new trajectory with improved line search, Eq.8, 12
            min_cost = np.inf
            for cur_alpha in self._alphas:
                cur_X, cur_U, cur_cost = self.forward(self._make_env, X, U, k, K, cur_alpha)

                if cur_cost < min_cost:
                    new_X, new_U, min_cost = cur_X, cur_U, cur_cost

            if min_cost > cost:
                self._logger.info("Cannot decrease cost : {:.4f} > {:.4f}".format(min_cost, cost))
                break

            if (cost - min_cost) < self._tol_cost:
                break

            X, U = new_X, new_U
            cost = min_cost

        self._X, self._U, self._cost = X, U, cost

    def backward(self, X, U, dynamics):
        """
        Perform backward pass using the previous states and controls.
        This step returns a local optimal controller gain k and K

        :param np.ndarray X: initial state vectors [T+1, N]
        :param np.ndarray U: initial control vectors [T, M]
        :param NumericalDiffDynamics dynamics:
        :return list k_list: open-loop term
        :return list K_list: feedback gain term
        """
        # Get planning horizon
        T = U.shape[0]

        mu = 0

        fail = True
        while fail:
            fail = False

            # Derivative of Value
            V_x = dynamics.L_x(X[-1])
            V_xx = dynamics.L_xx(X[-1])

            k_list = []  # Open-loop term
            K_list = []  # Feedback gain term

            # Define the regularization matrix
            eta_eye = mu * np.eye(dynamics.dim_state, dtype=NP_DTYPE)

            # Compute state-action-state function at each time step
            # Ignore the second order dynamics for model
            for t in range(T - 1, -1, -1):
                x = X[t]
                u = U[t]

                # Compute the second-order expansion coefficients of cost
                l_x, l_xx, l_u, l_uu, l_ux = dynamics.compute_cost_deriv(state=x, control=u)

                # Compute the first-order expansion coefficients of dynamics
                f_x, f_u = dynamics.compute_model_deriv(state=x, control=u)

                # Compute the second-order expansion coefficients of pseudo-Hamiltonian Q. Eq.5
                Q_x = l_x + np.dot(f_x.T, V_x)
                Q_u = l_u + np.dot(f_u.T, V_x)
                Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)
                Q_uu = l_uu + np.dot(np.dot(f_u.T, V_xx), f_u)
                Q_ux = l_ux + np.dot(np.dot(f_u.T, V_xx), f_x)

                Q_uu_d = l_uu + np.dot(np.dot(f_u.T, (V_xx + eta_eye)), f_u)
                Q_ux_d = l_ux + np.dot(np.dot(f_u.T, (V_xx + eta_eye)), f_x)

                # Increase regularization term if Q_uu is not positive definite
                if not is_pos_def(Q_uu_d):
                    mu = np.max([mu * self._mu, self._min_mu])

                    if mu > self._max_mu:
                        raise ValueError('Reached max iterations to find a PD Q mat-- Something is wrong! eta: {}'
                                         .format(mu))
                    fail = True
                    break

                # Compute local optimal feedback gain
                Q_uu_inv = np.linalg.inv(Q_uu_d)
                k = -np.dot(Q_uu_inv, Q_u)
                K = -np.dot(Q_uu_inv, Q_ux_d)
                k_list.append(k)
                K_list.append(K)

                # Update value function with improved regularization. Eq.11
                V_x = Q_x + np.dot(np.dot(K.T, Q_uu), k) + np.dot(K.T, Q_u) + np.dot(Q_ux.T, k)
                V_xx = Q_xx + np.dot(np.dot(K.T, Q_uu), K) + np.dot(K.T, Q_ux) + np.dot(Q_ux.T, K)

        k_list.reverse()
        K_list.reverse()

        return k_list, K_list

    def forward(self, make_env, X, U, k, K, alpha):
        """
        Perform forward pass using computed gains in the backward process.
        :param np.ndarray X: the previous state vectors [T+1, N]
        :param np.ndarray U: the previous control vectors [T, M]
        :return: updated states, controls, and cost
        """
        # Get planning horizon
        T = U.shape[0]

        new_X = np.zeros_like(X)
        new_U = np.zeros_like(U)

        x = X[0]
        new_X[0] = x

        # Initialize env
        env = make_env()
        env.set_state_vector(x)

        cost = 0.
        for t in range(T):
            # Compute control Eq.12
            u = U[t] + alpha * k[t] + np.dot(K[t], x - X[t])
            u = np.clip(u, env.action_space.low, env.action_space.high)
            new_U[t] = u

            # Compute cost
            cost += env.cost_state() + env.cost_control(u)

            # Evolve environment Eq.8
            env.step(u)
            x = env.get_state_vector()
            new_X[t + 1] = x

        return new_X, new_U, cost

    @property
    def X(self):
        return self._X

    @property
    def U(self):
        return self._U

    @property
    def cost(self):
        return self._cost

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--max-iter-optimization', type=int, default=30)
        parser.add_argument('--max-iter-each-step', type=int, default=1)
        parser.add_argument('--horizon', type=int, default=50)
        return parser
