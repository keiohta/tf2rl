import unittest


class CommonDist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dim = 3
        cls.batch_size = 4
        cls.dist = None

    def _skip_test_in_parent(self):
        if self.dist is None:
            return
        else:
            raise NotImplementedError

    def test_kl(self):
        self._skip_test_in_parent()

    def test_log_likelihood(self):
        self._skip_test_in_parent()

    def test_ent(self):
        self._skip_test_in_parent()

    def test_sample(self):
        self._skip_test_in_parent()
