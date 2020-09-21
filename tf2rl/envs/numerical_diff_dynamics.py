import numpy as np

from tf2rl.misc.ilqg_utils import NP_DTYPE


class ILQGInterfaceEnv(object):
    """Interface definition of environments for iLQG
    """
    def __init__(self):
        if hasattr(self, 'model'):
            assert self.model.na == 0

    def cost_state(self):
        """Compute cost for the current state.
        """
        raise NotImplementedError()

    def cost_control(self, control):
        """Compute cost for the given control.
        """
        raise NotImplementedError()

    def set_state_vector(self, state):
        raise NotImplementedError()

    def get_state_vector(self):
        raise NotImplementedError()

    @property
    def dim_state(self):
        raise NotImplementedError()


class NumericalDiffDynamics:
    """Compute derivatives of dynamics using numerical differentiation.
    We referred http://www.mujoco.org/book/source/derivative.cpp for computation.
    """

    def __init__(self, make_env):
        """
        :param make_env: a function to make an environment
        """
        # An environment to calculate values.
        self._env = make_env()
        self._env.reset()

        self.observation_space = self._env.observation_space
        self.dim_state = self._env.dim_state

        self.action_space = self._env.action_space
        self.dim_control = self.action_space.shape[0]

    def cost_state(self, state):
        self._env.set_state_vector(state)
        return self._env.cost_state()

    def cost_control(self, control):
        return self._env.cost_control(control)

    def L_x(self, state, eps=1e-6):
        L_x = finite_differences(self.cost_state, state, epsilon=eps)
        assert L_x.shape == (self.dim_state,)
        return L_x

    def L_xx(self, state, eps=1e-6):
        L_xx = finite_differences(self.L_x, state, epsilon=eps)
        assert L_xx.shape == (self.dim_state, self.dim_state)
        return L_xx

    def L_u(self, control, eps=1e-6):
        L_u = finite_differences(self.cost_control, control, epsilon=eps)
        assert L_u.shape == (self.dim_control,)
        return L_u

    def L_uu(self, control, eps=1e-6):
        L_uu = finite_differences(self.L_u, control, epsilon=eps)
        assert L_uu.shape == (self.dim_control, self.dim_control)
        return L_uu

    def L_ux(self, state, control, eps=1e-6):
        # We assume the cost function does not depend both state and control at the same term
        zeros = np.zeros((self.dim_control, self.dim_state), dtype=NP_DTYPE)
        return zeros

    def compute_cost_deriv(self, state, control, eps=1e-6):
        """
        Compute the derivative of costs: lx, lxx, lu, luu, lux.
        We assume the control cost does not depend on state, i.e. stateless, so we don't need dynamics to compute
        cost derivatives.
        """
        l_x = self.L_x(state, eps)
        l_u = self.L_u(control, eps)
        l_xx = self.L_xx(state, eps)
        l_uu = self.L_uu(control, eps)
        l_ux = self.L_ux(state, control, eps)
        return l_x, l_xx, l_u, l_uu, l_ux

    def compute_model_deriv(self, state, control, eps=1e-6):
        """
        Compute the derivative (Jacobian) of model: fx, fu
        """
        fx = np.zeros((self.dim_state, self.dim_state), NP_DTYPE)
        for idx in range(self.dim_state):
            test_state = np.copy(state)
            test_state[idx] += eps
            self._env.set_state_vector(test_state)
            self._env.step(control)
            obj_d1 = self._env.get_state_vector()

            test_state = np.copy(state)
            test_state[idx] -= eps
            self._env.set_state_vector(test_state)
            self._env.step(control)
            obj_d2 = self._env.get_state_vector()

            diff = (obj_d1 - obj_d2) / (2 * eps)
            fx[:, idx] = diff

        fu = np.zeros((self.dim_state, self.dim_control))
        for idx in range(self.dim_control):
            test_control = np.copy(control)
            test_control[idx] += eps
            self._env.set_state_vector(state)
            self._env.step(test_control)
            obj_d1 = self._env.get_state_vector()

            test_control = np.copy(control)
            test_control[idx] -= eps
            self._env.set_state_vector(state)
            self._env.step(test_control)
            obj_d2 = self._env.get_state_vector()

            diff = (obj_d1 - obj_d2) / (2 * eps)
            fu[:, idx] = diff

        return fx, fu


def finite_differences(func, inputs, epsilon=1e-6):
    """
    Computes gradients via finite differences as:
    ```
    derivative = (func(x+epsilon) - func(x-epsilon)) / (2*epsilon)
    ```
    Args:
        func: Function to compute gradient of. Inputs and outputs can be arbitrary dimension.
        inputs: Vector value to compute gradient at. So, enter the argument of the function for which we need the gradient
        epsilon: Difference to use for computing gradient.
    Returns:
        Jacobian of `func`
    """
    gradient = None

    for idx, _ in np.ndenumerate(inputs):
        test_inputs = np.copy(inputs)
        test_inputs[idx] += epsilon
        obj_d1 = func(test_inputs)

        if gradient is None:
            gradient = np.zeros(obj_d1.shape + inputs.shape)

        test_inputs = np.copy(inputs)
        test_inputs[idx] -= epsilon
        obj_d2 = func(test_inputs)

        diff = (obj_d1 - obj_d2) / (2 * epsilon)
        gradient[idx] += diff
    return gradient
