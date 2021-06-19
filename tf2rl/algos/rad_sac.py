import numpy as np
import tensorflow as tf

from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.algos.curl_sac import CURL
from tf2rl.tools.img_tools import center_crop, random_crop


class RAD(CURL):
    def __init__(self,
                 *args,
                 aug_types="",
                 name="RAD",
                 normalize_input=False,
                 **kwargs):
        # Normalize input image in Encoder
        super().__init__(*args, name=name, normalize_input=normalize_input, **kwargs)

        aug_type_to_func = {
            'crop': random_crop,
            # 'grayscale': random_grayscale,
            # 'cutout': random_cutout,
            # 'cutout_color': random_cutout_color,
            # 'flip': random_flip,
            # 'rotate': random_rotation,
            # 'rand_conv': random_convolution,
            # 'color_jitter': random_color_jitter,
            # 'translate': random_translate,
            # 'no_aug': no_aug,
        }

        self._aug_types = aug_types
        self._aug_funcs = {}

        if self._aug_funcs:
            for aug_type in self._aug_types.split('-'):
                assert aug_type in aug_type_to_func, f'Augmentation type {aug_type} is invalid.'
                self._aug_funcs[aug_type] = aug_type_to_func[aug_type]

    @tf.function
    def _get_action_body(self, state, test):
        # Scale image to [0, 1] because data augmentation is done in this scale.
        state = tf.divide(tf.cast(state, tf.float32),
                          tf.constant(255.))
        encoded_state = self._encoder(state)
        actions, log_pis = self.actor(encoded_state, test)
        return actions

    def _augment_imgs(self, states, next_states):
        # TODO: All data augmentation functions should be written by TF methods and use tf.function to speedup computation
        if self._aug_funcs:
            for aug_type, func in self._aug_funcs.items():
                if 'crop' in aug_type:
                    states = func(states, self._input_img_size)
                    next_states = func(next_states, self._input_img_size)
                elif 'cutout' in aug_type:
                    states = func(states)
                    next_states = func(next_states)
                elif 'translate' in aug_type:
                    cropped_states = center_crop(states)
                    cropped_next_states = center_crop(next_states)
                    states, rnd_indexes = func(cropped_states, self._input_img_size, return_rand_indexes=True)
                    next_states = func(cropped_next_states, self._input_img_size, **rnd_indexes)

        states = states.astype(np.float32) / 255.
        next_states = next_states.astype(np.float32) / 255.

        if self._aug_funcs:
            for aug_type, func in self._aug_funcs.items():
                if 'crop' in aug_type or 'cutout' in aug_type or 'translate' in aug_type:
                    continue
                states = func(states)
                next_states = func(next_states)

        return states, next_states

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        states, next_states = self._augment_imgs(states, next_states)

        # Update critic
        td_errors, qf_loss = self._update_critic(
            states, actions, next_states, rewards, dones, weights)
        tf.summary.scalar(name=self.policy_name + "/critic_loss", data=qf_loss)
        if self._n_update % self._update_critic_target_freq == 0:
            update_target_variables(self.qf1_target.weights, self.qf1.weights, self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, self.tau)

        # Update actor
        if self._n_update % self._update_actor_freq == 0:
            obs_features = self._encoder(states)
            actor_loss, logp_min, logp_max, logp_mean, alpha_loss = self._update_actor(obs_features)
            tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
            tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
            tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
            tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
            if self.auto_alpha:
                tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
                tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
            tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)
            tf.summary.scalar(name=self.policy_name + "/alpha_loss", data=alpha_loss)

        self._n_update += 1

        return td_errors
