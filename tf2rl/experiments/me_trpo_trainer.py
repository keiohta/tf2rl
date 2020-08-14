from tf2rl.experiments.mpc_trainer import MPCTrainer


class MeTrpoTrainer(MPCTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["n_dynamics_model"] = 5
        super().__init__(*args, **kwargs)
