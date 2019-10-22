import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.sac import SAC
from tf2rl.policies.categorical_actor import CategoricalActor
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss


class CriticQ(tf.keras.Model):
    """
    The output of Q-function moves
        from Q: S x A -> R
        to   Q: S -> R^|A|
    compared with continuous version of SAC
    """

    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.l2 = Dense(256, name="L2", activation='relu')
        self.l3 = Dense(action_dim, name="L2", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return values


class SACDiscrete(SAC):
    def __init__(
            self,
            *args,
            **kwargs):
        kwargs["name"] = "SAC_discrete"
        super().__init__(*args, **kwargs)

    def _set_up_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        # The output of actor is categorical distribution
        self.actor = CategoricalActor(
            state_shape, action_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_q(self, state_shape, action_dim, lr):
        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        self.qf1_target = CriticQ(state_shape, action_dim, name="qf1_target")
        self.qf2_target = CriticQ(state_shape, action_dim, name="qf2_target")
        update_target_variables(self.qf1_target.weights,
                                self.qf1.weights, tau=1.)
        update_target_variables(self.qf2_target.weights,
                                self.qf2.weights, tau=1.)
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_v(self, state_shape, lr):
        """
        Do not need state-value function because it can be directly computed from Q-function.
        See Eq.(10) in paper.

        :param state_shape:
        :param lr:
        :return:
        """
        pass

    def train(self, states, actions, next_states, rewards, done, weights=None):
        # TODO: Replace `done` with `dones`
        if weights is None:
            weights = np.ones_like(rewards)

        td_errors, actor_loss, mean_ent, logp_min, logp_max = \
            self._train_body(states, actions, next_states,
                             rewards, done, weights)

        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_loss", data=td_errors)
        tf.summary.scalar(name=self.policy_name + "/mean_ent", data=mean_ent)
        tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights=None):
        with tf.device(self.device):
            batch_size = states.shape[0]
            # rewards = tf.squeeze(rewards, axis=1)
            not_done = 1. - tf.cast(done, dtype=tf.float32)
            actions = tf.cast(actions, dtype=tf.int32)

            indices = tf.concat(
                values=[tf.expand_dims(tf.range(batch_size), axis=1),
                        actions], axis=1)

            with tf.GradientTape(persistent=True) as tape:
                # Update critic
                _, _, next_action_param = self.actor(next_states)
                next_action_prob = next_action_param["prob"]
                next_action_logp = tf.math.log(next_action_prob + 1e-8)
                next_q = tf.minimum(
                    self.qf1_target(next_states), self.qf2_target(next_states))

                target_q = tf.expand_dims(tf.einsum(
                    'ij,ij->i', next_action_prob, next_q - next_action_logp), axis=1)  # Eq.(10)
                target_q = tf.stop_gradient(
                    self.scale_reward * rewards + not_done * self.discount * target_q)

                current_q1 = tf.expand_dims(
                    tf.gather_nd(self.qf1(states), indices), axis=1)  # [batchsize, 1]
                current_q2 = tf.expand_dims(
                    tf.gather_nd(self.qf2(states), indices), axis=1)  # [batchsize, 1]
                current_q = tf.minimum(current_q1, current_q2)
                tf.assert_equal(target_q.shape, current_q1.shape)

                td_loss1 = tf.reduce_mean(huber_loss(
                    target_q - current_q1, delta=self.max_grad))
                td_loss2 = tf.reduce_mean(huber_loss(
                    target_q - current_q2, delta=self.max_grad))  # Eq.(7)

                # Update policy
                _, _, current_action_param = self.actor(states)
                current_action_prob = current_action_param["prob"]
                current_action_logp = tf.math.log(current_action_prob + 1e-8)

                policy_loss = tf.reduce_mean(
                    tf.einsum('ij,ij->i', current_action_prob,
                              current_action_logp - tf.stop_gradient(current_q)))  # Eq.(12)
                mean_entropy = tf.reduce_mean(
                    tf.einsum('ij,ij->i', current_action_prob, current_action_logp)) * (-1)

            q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            update_target_variables(self.qf1_target.weights,
                                    self.qf1.weights, tau=self.tau)
            update_target_variables(self.qf2_target.weights,
                                    self.qf2.weights, tau=self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

        return (td_loss1 + td_loss2) / 2., policy_loss, mean_entropy, \
               tf.reduce_min(current_action_logp), tf.reduce_max(current_action_logp)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_erros_Q1, td_errors_Q2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(
            np.abs(td_erros_Q1.numpy()) +
            np.abs(td_errors_Q2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            batch_size = states.shape[0]
            rewards = tf.squeeze(rewards, axis=1)
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            actions = tf.cast(actions, dtype=tf.int32)

            # Compute TD errors for V-value func
            sample_actions, logp, param = self.actor(states)
            probs = param["prob"]

            indices = tf.concat(
                values=[tf.expand_dims(tf.range(batch_size), axis=1),
                        actions], axis=1)

            # Compute TD errors for Q-value func
            current_Q1 = tf.expand_dims(
                tf.gather_nd(self.qf1(states), indices), axis=1)  # [batchsize, 1]
            current_Q2 = tf.expand_dims(
                tf.gather_nd(self.qf2(states), indices), axis=1)  # [batchsize, 1]
            current_Q = tf.minimum(current_Q1, current_Q2)

            target_Q = rewards + not_dones * self.discount * tf.stop_gradient(
                tf.einsum('ij,ij->i', probs, current_Q - tf.math.log(probs + 1e-8)))  # Eq.(10)

            td_errors_Q1 = target_Q - current_Q1
            td_errors_Q2 = target_Q - current_Q2

        return td_errors_Q1, td_errors_Q2


if __name__ == "__main__":
    import gym

    env = gym.make('LunarLander-v2')
    agent = SACDiscrete(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n)
    action = agent.get_action(env.observation_space.sample())
    print(action)
    batch_size = 4
    state_dim = env.observation_space.high.size
    act_dim = env.action_space.n

    states = np.zeros(shape=(batch_size, state_dim))
    next_states = np.zeros_like(states)
    actions = np.expand_dims(
        np.random.randint(0, act_dim, size=(batch_size), dtype=np.int32), axis=1)
    dones = np.zeros(shape=(batch_size,), dtype=bool)
    rewards = np.zeros(shape=(batch_size, 1), dtype=np.float32)

    agent.compute_td_error(
        states=states, actions=actions, next_states=states, dones=dones, rewards=rewards)

    agent.train(
        states=states, actions=actions, next_states=states, done=dones, rewards=rewards)
