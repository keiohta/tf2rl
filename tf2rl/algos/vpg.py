import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.contrib.distributions import MultivariateNormalDiag

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.algos.models import CategoricalActor, GaussianActor


class VPG(OnPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            is_discrete,
            max_action=1.,
            actor_units=[256, 256],
            lr=3e-4,
            name="VPG",
            **kwargs):
        super().__init__(self, name=name, **kwargs)
        self._is_discrete = is_discrete
        if is_discrete:
            self.actor = CategoricalActor(
                state_shape, action_dim, actor_units)
        else:
            self.actor = GaussianActor(
                state_shape, action_dim, actor_units)
        self._action_dim = action_dim
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float64)
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()

    # @tf.contrib.eager.defun
    def _get_action_body(self, state, test):
        if not test:
            if self._is_discrete:
                probs = self.actor(state)
                elems = tf.range(self._action_dim)
                samples = tf.multinomial(tf.log(probs), 1)
                return elems[tf.cast(samples[0][0], tf.int32)]
            else:
                return self.actor(state)[0][0]
        else:
            if self._is_discrete:
                return tf.argmax(self.actor(state)) 
            else:
                return self.actor.mean_action(state)

    def train(self, states, actions, next_states, rewards, done):
        loss, entropy = self._train_body(states, actions, next_states, rewards, done)

        tf.contrib.summary.scalar(name="Loss", tensor=loss, family="loss")
        tf.contrib.summary.scalar(name="Entropy", tensor=entropy, family="loss")

        return loss

    @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done):
        with tf.device(self.device):
            # loss = - log_likelihood * advantage + lambda * entropy
            if self._is_discrete:
                raise NotImplementedError
            else:
                raise NotImplementedError
        # return loss


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)
    import gym
    from tf2rl.envs.utils import is_discrete

    discrete_env = gym.make("CartPole-v0")
    agent = VPG(
        discrete_env.observation_space.shape,
        discrete_env.action_space.n,
        is_discrete(discrete_env.action_space))
    obs = discrete_env.reset()
    for _ in range(10):
        print(agent.get_action(obs))

    continuous_env = gym.make('Pendulum-v0')
    agent = VPG(
        continuous_env.observation_space.shape,
        continuous_env.action_space.low.size,
        is_discrete(continuous_env.action_space))
    obs =continuous_env.reset()
    for _ in range(10):
        print(agent.get_action(obs))
