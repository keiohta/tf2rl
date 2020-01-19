import numpy as np
import tensorflow as tf


class Policy(tf.keras.Model):
    def __init__(
            self,
            name,
            memory_capacity,
            update_interval=1,
            batch_size=256,
            discount=0.99,
            n_warmup=0,
            max_grad=10.,
            n_epoch=1,
            gpu=0):
        super().__init__()
        self.policy_name = name
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.discount = discount
        self.n_warmup = n_warmup
        self.n_epoch = n_epoch
        self.max_grad = max_grad
        self.memory_capacity = memory_capacity
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    def get_action(self, observation, test=False):
        raise NotImplementedError

    @staticmethod
    def get_argument(parser=None):
        import argparse
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--n-warmup', type=int, default=int(1e4))
        parser.add_argument('--batch-size', type=int, default=32)
        return parser


class OnPolicyAgent(Policy):
    """
    Base class for on-policy agents
    """

    def __init__(
            self,
            horizon=2048,
            lam=0.95,
            enable_gae=True,
            normalize_adv=True,
            entropy_coef=0.01,
            vfunc_coef=1.,
            **kwargs):
        self.horizon = horizon
        self.lam = lam
        self.enable_gae = enable_gae
        self.normalize_adv = normalize_adv
        self.entropy_coef = entropy_coef
        self.vfunc_coef = vfunc_coef
        kwargs["n_warmup"] = 0
        kwargs["memory_capacity"] = self.horizon
        super().__init__(**kwargs)
        assert self.horizon % self.batch_size == 0, \
            "Horizon should be divisible by batch size"

    @staticmethod
    def get_argument(parser=None):
        parser = Policy.get_argument(parser)
        parser.add_argument('--horizon', type=int, default=2048)
        parser.add_argument('--normalize-adv', action='store_true')
        parser.add_argument('--enable-gae', action='store_true')
        return parser


class OffPolicyAgent(Policy):
    """
    Base class for off-policy agents
    """

    def __init__(
            self,
            memory_capacity,
            **kwargs):
        super().__init__(memory_capacity=memory_capacity, **kwargs)

    @staticmethod
    def get_argument(parser=None):
        parser = Policy.get_argument(parser)
        parser.add_argument('--memory-capacity', type=int, default=int(1e6))
        return parser


class IRLPolicy(Policy):
    def __init__(
            self,
            n_training=1,
            memory_capacity=0,
            **kwargs):
        self.n_training = n_training
        super().__init__(memory_capacity=memory_capacity, **kwargs)
