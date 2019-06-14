import unittest

from tests.algos.common import CommonAlgos


class CommonModel(CommonAlgos):
    def _test_call(self, inputs, expected_action_shapes, expected_log_prob_shapes):
        """Check shape of ouputs
        """
        # Probabilistic sampling
        actions, log_probs = self.policy(inputs, test=False)
        self.assertEqual(actions.shape, expected_action_shapes)
        self.assertEqual(log_probs.shape, expected_log_prob_shapes)
        # Greedy sampling
        actions, log_probs = self.policy(inputs, test=True)
        self.assertEqual(actions.shape, expected_action_shapes)
        self.assertEqual(log_probs.shape, expected_log_prob_shapes)

    def _test_compute_log_probs(self, states, actions, expected_shapes):
        log_probs = self.policy.compute_log_probs(states, actions)
        self.assertEqual(log_probs.shape, expected_shapes)


if __name__ == '__main__':
    unittest.main()
