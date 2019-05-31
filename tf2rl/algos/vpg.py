import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.contrib.distributions import MultivariateNormalDiag

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.algos.models import CategoricalActor, GaussianActor


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, units, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        with tf.device('/cpu:0'):
            self(dummy_state)

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)


class VPG(OnPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            is_discrete,
            max_action=1.,
            actor_units=[256, 256],
            critic_units=[256, 256],
            lr_actor=3e-4,
            lr_critic=3e-4,
            name="VPG",
            **kwargs):
        super().__init__(name=name, **kwargs)
        self._is_discrete = is_discrete
        if is_discrete:
            self.actor = CategoricalActor(
                state_shape, action_dim, actor_units)
        else:
            self.actor = GaussianActor(
                state_shape, action_dim, max_action, actor_units)
        self.critic = CriticV(state_shape, critic_units)
        self._action_dim = action_dim
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=lr_actor)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=lr_critic)

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)

        single_input = state.ndim == 1
        if single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, log_pi = self._get_action_body(tf.constant(state), test)

        action = action.numpy()
        log_pi = log_pi.numpy()
        if single_input:
            action = action[0]
            log_pi = log_pi[0] if not test else None
        return action, log_pi if not test else log_pi

    # @tf.contrib.eager.defun
    def _get_action_body(self, state, test):
        if not test:
            if self._is_discrete:
                probs = self.actor(state)
                elems = tf.range(self._action_dim)
                samples = tf.multinomial(tf.log(probs), 1)
                return elems[tf.cast(samples[0][0], tf.int32)]
            else:
                return self.actor(state)
        else:
            if self._is_discrete:
                return tf.argmax(self.actor(state)) 
            else:
                return self.actor.mean_action(state), None

    def train_actor(self, states, actions, next_states, rewards, dones, log_pis):
        loss = self._train_actor_body(states, actions, next_states, rewards, dones, log_pis)
        tf.contrib.summary.scalar(name="Loss", tensor=loss, family="loss")
        return loss

    def train_critic(self, states, actions, next_states, rewards, dones):
        loss = self._train_critic_body(states, actions, next_states, rewards, dones)
        tf.contrib.summary.scalar(name="Loss", tensor=loss, family="loss")
        return loss

    @tf.contrib.eager.defun
    def _train_actor_body(self, states, actions, next_states, rewards, dones, log_pis):
        with tf.device(self.device):
            # Train policy
            with tf.GradientTape() as tape:
                not_dones = tf.constant(1., dtype=tf.float32) - dones
                log_probs = self.actor.compute_log_probs(states, actions)
                log_likelihood_ratio = log_probs - log_pis  # old log_probs
                advantages = rewards + not_dones * self.critic(next_states) - self.critic(states)
                actor_loss = -log_likelihood_ratio * advantages  # + lambda * entropy
                if self._is_discrete:
                    raise NotImplementedError
                else:
                    return actor_loss

    @tf.contrib.eager.defun
    def _train_critic_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                not_dones = tf.constant(1., dtype=tf.float32) - dones
                target_V = self.critic(next_states)
                target_V = rewards + not_dones * target_V
                target_V = tf.stop_gradient(target_V)
                current_V = self.critic(states)
                td_errors = target_V - current_V
                critic_loss = tf.reduce_mean(tf.square(td_errors) * 0.5)
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        return critic_loss


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
