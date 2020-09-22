from tf2rl.algos.ilqg import ILQG


class MPCILQG(ILQG):
    """Model Predictive Control using iLQG.
    """
    def update_state(self, state):
        self._env.set_state_vector(state)

    def get_next_action(self, state):
        self.update_state(state)
