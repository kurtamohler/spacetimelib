import spacetime as st
import numpy as np
import unittest

def check_boost_1D(v, x, t):
    L_factor = (1 - v ** 2) ** 0.5
    t_out = (t - v * x) / L_factor
    x_out = (x - v * t) / L_factor
    return x_out, t_out

class SpacetimeTestSuite(unittest.TestCase):
    # Test boosting in one spatial dimension with randomized inputs
    def test_boost_1D_random(self):
        v_batch = []
        event_batch = []
        event_out_batch = []

        for _ in range(10):
            v = np.random.uniform(low=0.1, high=1.0, size=())
            x = np.random.uniform(low=-1000, high=1000, size=())
            t = np.random.uniform(low=-1000, high=1000, size=())
            event = (t, x)
            event_out, _ = st.boost(v, event)
            x_expected, t_expected = check_boost_1D(v, x, t)

            assert np.isclose(t_expected, event_out[0]).all()
            assert np.isclose(x_expected, event_out[1]).all()

            v_batch.append(v)
            event_batch.append(event)
            event_out_batch.append(event_out)

        # Test batched mode

        v = np.array(v_batch)

        # TODO: I don't like this expansion. I should be able to remove the
        # need for it, right?
        v = np.expand_dims(v, -1)

        event = np.array(event_batch)
        event_out_expected = np.array(event_out_batch)

        event_out, _ = st.boost(v, event)
        assert np.isclose(event_out, event_out_expected).all()


if __name__ == '__main__':
    unittest.main()
