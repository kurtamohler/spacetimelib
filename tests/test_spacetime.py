import spacetime as st
import numpy as np
import unittest

def check_boost_event_1D(v, event):
    t = event[0]
    x = event[1]
    L_factor = (1 - v ** 2) ** 0.5
    t_out = (t - v * x) / L_factor
    x_out = (x - v * t) / L_factor
    return np.array([t_out, x_out])

def check_boost_velocity_1D(v, u):
    v = np.array(v)
    u = np.array(u)
    return (u - v) / (1 - u * v)

class SpacetimeTestSuite(unittest.TestCase):

    # Test boosting events in one spatial dimension with randomized inputs
    def test_boost_event_1D_random(self):
        v_batch = []
        event_batch = []
        event_out_batch = []

        for _ in range(10):
            v = np.random.uniform(low=0.1, high=1.0, size=()).astype(np.double)
            event = np.random.uniform(low=-1000, high=1000, size=(2,)).astype(np.double)

            event_out = st.boost(v, event)

            event_out_check = check_boost_event_1D(v, event)
            assert np.isclose(event_out, event_out_check).all()

            v_batch.append(v)
            event_batch.append(event)
            event_out_batch.append(event_out)

        # Test batched mode
        v = np.array(v_batch, dtype=np.double)
        event = np.array(event_batch, dtype=np.double)
        event_out_check = np.array(event_out_batch, dtype=np.double)

        event_out = st.boost(v, event)

        assert np.isclose(event_out, event_out_check).all()

    # Test boosting velocities in one spatial dimension with randomized inputs
    def test_boost_velocity_1D_random(self):
        v_batch = []
        event_batch = []
        event_out_batch = []
        u_batch = []
        u_out_batch = []

        for _ in range(10):
            v = np.random.uniform(low=0.1, high=1.0, size=()).astype(np.double)
            u = np.random.uniform(low=0.1, high=1.0, size=()).astype(np.double)
            event = np.random.uniform(low=-1000, high=1000, size=(2,)).astype(np.double)

            u_out_check = check_boost_velocity_1D(v, u)
            event_out_check = check_boost_event_1D(v, event)

            # Check boosting velocity without event
            _, u_out = st.boost(v, None, u)
            assert np.isclose(u_out, u_out_check).all()

            # Check boosting velocity with event
            event_out, u_out = st.boost(v, event, u)

            assert np.isclose(u_out, u_out_check).all()
            assert np.isclose(event_out, event_out_check).all()


            v_batch.append(v)
            event_batch.append(event)
            event_out_batch.append(event_out)
            u_batch.append(u)
            u_out_batch.append(u_out)

        # Test batched mode
        v = np.array(v_batch, dtype=np.double)
        event = np.array(event_batch, dtype=np.double)
        event_out_check = np.array(event_out_batch, dtype=np.double)
        u = np.array(u_batch, dtype=np.double)
        u_out_check = np.array(u_out_batch, dtype=np.double)

        event_out, u_out = st.boost(v, event, u)

        assert np.isclose(event_out, event_out_check).all()
        assert np.isclose(u_out, u_out_check).all()

    # Test `spacetime.boost` with lots of different input shapes
    def test_boost_shapes(self):
        input_shapes = [
            # frame_v_shape, x_shape, u_shape, x_out_shape_check, u_out_shape_check
            # NOTE: If x_out_shape_check or u_out_shape_check is None, it indicates
            # that the corresponding output shape should be the same as the input shape.
            ((2,), (3,), (2,), None, None),
            ((2,), (3,), (1, 2,), None, None),
            ((2,), (3,), (3, 2,), None, None),
            ((2,), (3,), (3, 2,), None, None),
            ((2,), (3,), (1, 3, 2,), None, None),
            ((3,), (4,), (10, 30, 3,), None, None),
            ((2,), (1, 3,), (2,), None, None),
            ((3,), (1, 4,), (3,), None, None),
            ((3,), (10, 4,), (3,), None, None),
            ((3,), (5, 10, 4,), (3,), None, None),
            ((3,), (5, 10, 4,), (30, 3,), None, None),
            ((3,), (5, 10, 4,), (5, 1, 30, 3,), None, None),
            ((10, 3,), (5, 10, 4,), (5, 1, 10, 3,), None, None),
            ((4, 2,), (3,), (2,), (4, 3), (4, 2)),
            ((4, 2,), (1, 3,), (2,), (4, 3), (4, 2)),
            ((4, 2,), (1, 3,), (2,), (4, 3), (4, 2)),
            ((4, 2,), (1, 1, 3,), (2,), (1, 4, 3), (4, 2)),
            ((4, 2,), (3,), (1, 2,), (4, 3), (4, 2)),
            ((4, 2,), (3,), (4, 2,), (4, 3), None),
            ((4, 2,), (3,), (1, 1, 2,), (4, 3), (1, 4, 2)),
            ((4, 2,), (3,), (1, 4, 2,), (4, 3), None),
            ((4, 2,), (4, 3,), (1, 4, 2,), None, None),
        ]

        for frame_v_shape, x_shape, u_shape, x_out_shape_check, u_out_shape_check in input_shapes:
            if x_out_shape_check is None:
                x_out_shape_check = x_shape

            if u_out_shape_check is None:
                u_out_shape_check = u_shape

            frame_v = 0.1 * np.random.randn(*frame_v_shape)
            x = np.random.randn(*x_shape)
            u = 0.1 * np.random.randn(*u_shape)

            x_out, u_out = st.boost(frame_v, x, u)

            self.assertEqual(x_out_shape_check, x_out.shape)
            self.assertEqual(u_out_shape_check, u_out.shape)

    def test_Worldline_eval(self):
        w0 = st.Worldline(
            [[0, 0]],
            [0.9])
        w1 = st.Worldline(
            [[0, 0]],
            vel_past=[0.7],
            vel_future=[-0.1])
        w2 = st.Worldline(
            [[-100, 1], [10, -50]],
            [-0.4])
        w3 = st.Worldline(
            [[-100, 1], [10, -50]],
            vel_past=[0.4],
            vel_future=[-0.1])
        test_cases = [
            # worldline, time, event_check
            (w0, 0, [0, 0]),
            (w0, 1, [1, 0.9]),
            (w0, 2, [2, 1.8]),
            (w0, -1, [-1, -0.9]),
            (w0, -2, [-2, -1.8]),
            (w1, -10, [-10, -7]),
            (w1, 10, [10, -1]),
            (w1, 0, [0, 0]),
            (w2, -200, [-200, 1 + 0.4 * 100]),
            (w2, -100, [-100, 1]),
            (w2, -50, [-50, 1 + (-51 * 50 / 110)]),
            (w2, 10, [10, -50]),
            (w2, 30, [30, -50 + 20 * -0.4]),
            (w3, -200, [-200, 1 - 0.4 * 100]),
            (w3, -100, [-100, 1]),
            (w3, -50, [-50, 1 + (-51 * 50 / 110)]),
            (w3, 10, [10, -50]),
            (w3, 30, [30, -50 + 20 * -0.1]),
        ]
        for w, t, event_check in test_cases:
            assert event_check[0] == t, (
                'test case is invalid, time coordinate does not match')
            event_check = np.array(event_check)
            event = w.eval(t)
            self.assertTrue(np.isclose(event, event_check).all())

    def test_Worldline_proper_time(self):
        w0 = st.Worldline([[0, 0], [1, 0.9], [2, 0]])
        tau0 = (2 ** 2 - 1.8 ** 2) ** 0.5

        test_cases = [
            # worldline, time0, time1, tau_check
            (w0, 0, 2, tau0),
            (w0, 0, 1, tau0 / 2),
            (w0, 1, 2, tau0 / 2),
            (w0, 0.5, 1.5, tau0 / 2),
            (w0, 0.25, 0.75, tau0 / 4),
            (w0, 1.25, 1.75, tau0 / 4),
            (w0, 0, 0, 0),
            (w0, 0, 0.123, 0.123 * tau0 / 2),
            (w0, 0.123, 0.123, 0),
            (w0, 1, 1, 0),
            (w0, 1.234, 1.234, 0),
            (w0, 2, 2, 0),
        ]

        for w, time0, time1, tau_check in test_cases:
            self.assertAlmostEqual(w.proper_time(time0, time1), tau_check)
            self.assertAlmostEqual(w.proper_time(time1, time0), tau_check)

if __name__ == '__main__':
    unittest.main()
