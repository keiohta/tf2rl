import numpy as np

from tf2rl.algos.ilqg import ILQG


class MPCILQG(ILQG):
    """Model Predictive Control using iLQG.
    """
    def get_next_action(self, state):
        # Update state
        self._env.set_state_vector(state)

        # Initialize action with previously computed control sequence
        self._U = np.concatenate((self._U[1:], [self._U[-1]]))

        # Update state sequence
        self._X = [state]
        for u in self._U:
            self._env.step(u)
            state = self._env.get_state_vector()
            self._X.append(state)

        # Optimize control sequence
        self.optimize()
        return self._U[0]
