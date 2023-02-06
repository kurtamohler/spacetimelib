import spacetime as st
import numpy as np
import unittest
from itertools import product

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

    def test_norm_s(self):
        test_sizes = [
            (1,),
            (2,),
            (3,),
            (4,),
            (10,),
            (3, 10),
            (3, 4, 5),
        ]

        # Check against a few different implementations
        check_funcs = [
            lambda vec_s: np.linalg.norm(vec_s, axis=-1),
            lambda vec_s: np.sum(vec_s ** 2, axis=-1) ** 0.5,
        ]

        for test_size, check_func in product(test_sizes, check_funcs):
            a = 100 * np.random.randn(*test_size)

            res = st.norm_s(a)
            res_check = check_func(a)

            self.assertTrue(np.isclose(res, res_check).all())

    def test_norm_st2(self):
        test_sizes = [
            (1+1,),
            (2+1,),
            (3+1,),
            (4+1,),
            (10+1,),
            (3, 10+1),
            (3, 4, 5+1),
        ]

        def check_func_0(vec_s):
            metric = np.eye(vec_s.shape[-1])
            metric[0][0] = -1
            return np.sum(np.matmul(vec_s ** 2, metric), axis=-1)

        def check_func_1(vec_s):
            tmp = vec_s ** 2
            tmp[..., 0] *= -1
            return np.sum(tmp, axis=-1)

        # Check against a few different implementations
        check_funcs = [
            check_func_0,
            check_func_1,
        ]

        for test_size, check_func in product(test_sizes, check_funcs):
            a = 100 * np.random.randn(*test_size)

            res = st.norm_st2(a)
            res_check = check_func(a)

            self.assertTrue(np.isclose(res, res_check).all())

    # Test `velocity_st` and `velocity_s`
    def test_velocity(self):
        def velocity_st_check(vel_s):
            speed = np.expand_dims(st.norm_s(vel_s), -1)
            t = np.ones(vel_s.shape[:-1] + (1,), dtype=vel_s.dtype)

            return np.concatenate([t, vel_s], axis=-1) / (1 - speed ** 2)**0.5

        test_cases = [
            # Input, expected result
            (np.asarray([0, 0, 0]), np.asarray([1, 0, 0, 0])),
            (np.asarray([.9]), velocity_st_check(np.array([.9]))),
            (np.asarray([0, -.8]), velocity_st_check(np.array([0, -.8]))),
        ]

        for vel_s, vel_st_check in test_cases:
            vel_st = st.velocity_st(vel_s)
            self.assertTrue(np.isclose(vel_st, vel_st_check).all())

            # The norm of a spacetime-velocity should always be very close to -1
            norm2_vel_st = st.norm_st2(vel_st)
            self.assertTrue(np.isclose(norm2_vel_st, -1).all())

            # Converting back to a space-velocity should give the original value
            vel_s_check = st.velocity_s(vel_st)
            self.assertTrue(np.isclose(vel_s, vel_s_check).all())

        rand_input_sizes = [
            (1,),
            (2,),
            (3,),
            (4,),
            (100,),
            (1, 2),
            (10, 2),
        ]

        for _ in range(10):
            for input_size in rand_input_sizes:
                max_rand_norm = (input_size[-1] * 4) ** 0.5

                rand_vec = np.random.rand(*input_size) - 0.5
                rand_norm_target = 0.9 * np.random.rand(*input_size[:-1])

                vel_s = np.expand_dims(rand_norm_target, -1) * rand_vec / np.expand_dims(st.norm_s(rand_vec), -1)

                vel_st_check = velocity_st_check(vel_s)
                vel_st = st.velocity_st(vel_s)
                self.assertTrue(np.isclose(vel_st, vel_st_check).all())

                # The norm of a spacetime-velocity should always be very close to -1
                norm2_vel_st = st.norm_st2(vel_st)
                self.assertTrue(np.isclose(norm2_vel_st, -1).all())

                # Converting back to a space-velocity should give the original value
                vel_s_check = st.velocity_s(vel_st)
                self.assertTrue(np.isclose(vel_s, vel_s_check).all())

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

    def test_boost_velocity(self):
        test_cases = [
            # vel_s, boost_vel_s, check_res
            (0.1, 0.2, (0.1 - 0.2) / (1 - 0.1 * 0.2)),

            # Boosting 0 vel to anything should result in the negative of the
            # boost vel
            (0, 0.99, -0.99),
            (0, -0.9581, 0.9581),
            ([0], [-0.9581], [0.9581]),

            # When boosting low velocities by a low velocity, result should
            # just be the difference between the two, since it approximates
            # Galilean transformation
            (0.001, -0.001, 0.002),
            (0.001, 0.001, 0),
            (0.001, 0.999, -0.999),

            # At sufficiently high velocity, boosting further in opposite
            # direction gives almost the same result
            (0.99999, -0.99999, 0.99999),

            # Batched boost at low velocities
            (
                [[[0.001], [-0.0008], [0]],
                 [[-.001], [0.0002],  [-0.0001234]]],
                0.001,
                [[[0], [-0.0018], [-0.001]],
                 [[-.002], [-0.0008],  [-0.0011234]]]
            ),
            (
                0.001,
                [0.001, -0.001],
                [0, 0.002]
            ),
        ]

        for vel_s, boost_vel_s, check_res in test_cases:
            vel_s = np.array(vel_s)
            boost_vel_s = np.array(boost_vel_s)
            check_res = np.array(check_res)
            res = st.boost_velocity_s(vel_s, boost_vel_s)
            
            msg = f'res: {res}\ncheck_res: {check_res}'

            self.assertTrue(res.shape == check_res.shape, msg=msg)
            self.assertTrue(np.isclose(res, check_res).all(), msg=msg)

    # Test boosting velocities in one spatial dimension with randomized inputs
    def test_boost_velocity_1D_random(self):
        v_batch = []
        u_batch = []
        u_out_batch = []

        for _ in range(10):
            v = np.random.uniform(low=0.1, high=1.0, size=()).astype(np.double)
            u = np.random.uniform(low=0.1, high=1.0, size=()).astype(np.double)

            u_out_check = check_boost_velocity_1D(v, u)

            u_out = st.boost_velocity_s(u, v)

            assert np.isclose(u_out, u_out_check).all()

            v_batch.append([v])
            u_batch.append([u])
            # TODO: Need to avoid adding a dim to output if inputs are scalar
            u_out_batch.append([u_out])

        # Test batched mode
        v = np.array(v_batch, dtype=np.double)
        u = np.array(u_batch, dtype=np.double)
        u_out_check = np.array(u_out_batch, dtype=np.double)

        u_out = st.boost_velocity_s(u, v)

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

            x_out = st.boost(frame_v, x)
            u_out = st.boost_velocity_s(u, frame_v)

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

    def test_Worldline__proper_time(self):
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
            self.assertAlmostEqual(w._proper_time(time0, time1), tau_check)
            self.assertAlmostEqual(w._proper_time(time1, time0), tau_check)

if __name__ == '__main__':
    unittest.main()
