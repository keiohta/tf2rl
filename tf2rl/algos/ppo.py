import numpy as np
import tensorflow as tf

from tf2rl.algos.vpg import VPG


class PPO(VPG):
    def __init__(
            self,
            clip=True,
            clip_ratio=0.2,
            name="PPO",
            **kwargs):
        kwargs["hidden_activation"] = "tanh"
        super().__init__(name=name, **kwargs)
        self.clip = clip
        self.clip_ratio = clip_ratio

    def train(self, states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        if self.actor_critic is not None:
            actor_loss, critic_loss, logp_news, ratio, ent = \
                self._train_actor_critic_body(
                    states, actions, advantages, logp_olds, returns)
        else:
            actor_loss, logp_news, ratio, ent = self._train_actor_body(
                states, actions, advantages, logp_olds)
            critic_loss = self._train_critic_body(states, returns)
        # Visualize results in TensorBoard
        tf.summary.scalar(name=self.policy_name+"/actor_loss",
                          data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_max",
                          data=np.max(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_min",
                          data=np.min(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_mean",
                          data=np.mean(logp_news))
        tf.summary.scalar(name=self.policy_name+"/adv_max",
                          data=np.max(advantages))
        tf.summary.scalar(name=self.policy_name+"/adv_min",
                          data=np.min(advantages))
        tf.summary.scalar(name=self.policy_name+"/kl",
                          data=tf.reduce_mean(logp_olds - logp_news))
        tf.summary.scalar(name=self.policy_name+"/ent",
                          data=ent)
        tf.summary.scalar(name=self.policy_name+"/ratio",
                          data=tf.reduce_mean(ratio))
        tf.summary.scalar(name=self.policy_name+"/critic_loss",
                          data=critic_loss)
        # DEBUG START
        from tf2rl.policies.gaussian_actor import GaussianActor
        if isinstance(self.actor, GaussianActor):
            if self.actor._state_independent_std:
                tf.summary.scalar(name=self.policy_name+"/lop_std_mean",
                                  data=tf.reduce_mean(self.actor.out_log_std))
        tf.summary.scalar(name=self.policy_name+"/actions",
                          data=tf.reduce_mean(actions))
        # DEBUG END
        return actor_loss, critic_loss

    @tf.function
    def _train_actor_critic_body(
            self, states, actions, advantages, logp_olds, returns):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                _, _, current_V = self.actor_critic(states)
                ent = tf.reduce_mean(
                    self.actor_critic.compute_entropy(states))
                # Train actor
                if self.clip:
                    logp_news = self.actor_critic.compute_log_probs(
                        states, actions)
                    ratio = tf.math.exp(
                        logp_news - tf.squeeze(logp_olds))
                    min_adv = tf.clip_by_value(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio) * tf.squeeze(advantages)
                    actor_loss = -tf.reduce_mean(tf.minimum(
                        ratio * tf.squeeze(advantages),
                        min_adv))
                    actor_loss -= self.entropy_coef * ent
                else:
                    raise NotImplementedError
                # Train critic
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(0.5 * tf.square(td_errors))
                total_loss = actor_loss + self.vfunc_coef * critic_loss

            grads = tape.gradient(
                total_loss, self.actor_critic.trainable_variables)
            self.actor_critic_optimizer.apply_gradients(
                zip(grads, self.actor_critic.trainable_variables))

        return actor_loss, critic_loss, logp_news, ratio, ent

    @tf.function
    def _train_actor_body(self, states, actions, advantages, logp_olds):
        with tf.device(self.device):
            # Update actor
            with tf.GradientTape() as tape:
                ent = tf.reduce_mean(
                    self.actor.compute_entropy(states))
                if self.clip:
                    logp_news = self.actor.compute_log_probs(
                        states, actions)
                    ratio = tf.math.exp(logp_news - tf.squeeze(logp_olds))
                    min_adv = tf.clip_by_value(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio) * tf.squeeze(advantages)
                    actor_loss = -tf.reduce_mean(tf.minimum(
                        ratio * tf.squeeze(advantages),
                        min_adv))
                    actor_loss -= self.entropy_coef * ent
                else:
                    raise NotImplementedError
            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

        return actor_loss, logp_news, ratio, ent
