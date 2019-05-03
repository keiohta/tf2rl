import numpy as np
import tensorflow as tf


class Policy(tf.contrib.checkpoint.Checkpointable):
    def __init__(
            self,
            name,
            memory_capacity,
            update_interval=1,
            batch_size=256,
            discount=0.99,
            n_warmup=0,
            gpu=0):
        self.name = name
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.discount = discount
        self.n_warmup = n_warmup
        self.memory_capacity = memory_capacity
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    def get_action(self, observation, test=False):
        raise NotImplementedError


class OnPolicyAgent(Policy):
    """Base class for on-policy agent
    """
    def __init__(
            self,
            memory_capacity=None,
            **kwargs):
        if memory_capacity is None:
            memory_capacity = kwargs["batch_size"]
        super().__init__(memory_capacity=memory_capacity, **kwargs)


class OffPolicyAgent(Policy):
    """Base class for off-policy agent
    """
    def __init__(
            self,
            memory_capacity,
            **kwargs):
        super().__init__(memory_capacity=memory_capacity, **kwargs)
