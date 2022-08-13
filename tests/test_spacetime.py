import spacetime as st
import numpy as np
import unittest

class SpacetimeTestSuite(unittest.TestCase):
    # Test boosting in one spatial dimension
    def test_boost_1D(self):
        def boost_1D(v, x, t):
            L_factor = (1 - v ** 2) ** 0.5
            t_out = (t - v * x) / L_factor
            x_out = (x - v * t) / L_factor
            return x_out, t_out

        # Perform some randomized tests
        for _ in range(100):
            v = np.random.uniform(low=0.1, high=1.0, size=())
            x = np.random.uniform(low=-1000, high=1000, size=())
            t = np.random.uniform(low=-1000, high=1000, size=())

            event, _ = st.boost((v,), (t, x))
            x_expected, t_expected = boost_1D(v, x, t)

            assert np.isclose(t_expected, event[0]).all()
            assert np.isclose(x_expected, event[1]).all()

if __name__ == '__main__':
    unittest.main()
