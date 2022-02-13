import platform
import unittest
import tensorflow as tf

from tests.algos.common import CommonOffPolImgContinuousAlgos


@unittest.skipIf((platform.system() == 'Windows') and (tf.version.VERSION.rsplit('.', 1)[0] == "2.0"),
                 "tensorflow-addons does not support tf2.0 on Windows")
class TestSAC(CommonOffPolImgContinuousAlgos):
    @classmethod
    def setUpClass(cls):
        from tf2rl.algos.sac_ae import SACAE
        super().setUpClass(img_dim=84)
        cls.agent = SACAE(
            action_dim=cls.continuous_env.action_space.low.size,
            batch_size=cls.batch_size,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
