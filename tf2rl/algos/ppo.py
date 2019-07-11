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

    def train_actor(self, states, actions, advantages, logp_olds):
        actor_loss, logp_news, ratio = self._train_actor_body(
            states, actions, advantages, logp_olds)
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
                          data=tf.reduce_mean(-logp_news))
        tf.summary.scalar(name=self.policy_name+"/ratio",
                          data=tf.reduce_mean(ratio))
        return actor_loss

    @tf.function
    def _train_actor_body(self, states, actions, advantages, logp_olds):
        with tf.device(self.device):
            # Update actor
            with tf.GradientTape() as tape:
                if self.clip:
                    logp_news = self.actor.compute_log_probs(states, actions)
                    ratio = tf.math.exp(logp_news - tf.squeeze(logp_olds))
                    min_adv = tf.squeeze(tf.where(
                        advantages > 0,
                        (1. + self.clip_ratio) * advantages,
                        (1. - self.clip_ratio) * advantages))
                    surr_loss = -tf.reduce_mean(tf.minimum(
                        ratio * tf.squeeze(advantages), min_adv))
                else:
                    raise NotImplementedError
                actor_loss = surr_loss  # + lambda * entropy
            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

        return actor_loss, logp_news, ratio
