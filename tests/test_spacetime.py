import spacetimelib as st
import numpy as np
import unittest
from itertools import product

def maybe_arraylike_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    else:
        return (np.asarray(a) == np.asarray(b)).all()

def maybe_arraylike_close(a, b):
    if a is None or b is None:
        return a is None and b is None
    else:
        return np.isclose(np.asarray(a), np.asarray(b)).all()

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

class SpacetimelibTestSuite(unittest.TestCase):

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

    def test_norm2_st(self):
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

            res = st.norm2_st(a)
            res_check = check_func(a)

            self.assertTrue(np.isclose(res, res_check).all())

    def test_proper_time_delta(self):
        with self.assertRaisesRegex(ValueError, r"expected events to have time-like interval"):
            st.proper_time_delta([0, 0], [0, 1])

        with self.assertRaisesRegex(ValueError, r"expected events to have same number"):
            st.proper_time_delta([0, 0], [0, 1, 2])

        with self.assertRaisesRegex(ValueError, r"expected events to have same number"):
            st.proper_time_delta([0], [0])

        res = st.proper_time_delta(np.zeros([1, 10, 3]), np.zeros([1, 20, 1, 3]))
        self.assertEqual(res.shape, (1, 20, 10))

        with self.assertRaisesRegex(ValueError, r"operands could not be broadcast"):
            res = st.proper_time_delta(np.zeros([2, 10, 3]), np.zeros([20, 1, 3]))


        test_cases = [
            # event0, event1, res_expected
            ([0, 0], [0, 0], 0),
            ([-1.2, 4.5], [-1.2, 4.5], 0),
            ([0, 0], [1, 0], 1),
            ([0, 0], [-1, 0], -1),
            ([3.4, -6.7], [-10.2, -4.5], -((-10.2 - 3.4)**2 - (-4.5 + 6.7)**2)**0.5),
            ([1, -2], [4, -3], (3**2 - 1**2)**0.5),
            ([-10, 5], [21, -13], (31**2 - 18**2)**0.5),
            ([999, 21], [777, 12], -(222**2 - 9**2)**0.5),
        ]

        for event0, event1, res_expected in test_cases:
            res = st.proper_time_delta(event0, event1)
            self.assertTrue((res == res_expected).all())


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
            norm2_vel_st = st.norm2_st(vel_st)
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
                norm2_vel_st = st.norm2_st(vel_st)
                self.assertTrue(np.isclose(norm2_vel_st, -1).all())

                # Converting back to a space-velocity should give the original value
                vel_s_check = st.velocity_s(vel_st)
                self.assertTrue(np.isclose(vel_s, vel_s_check).all())

    # Test the new implementation of `st.boost` against the old one
    def test_boost_against_old(self):
        test_cases = [
            # vec_st_size, boost_vel_s_size
            ((2,), (1,)),
            ((4,), (3,)),
            ((10,), (9,)),

            ((1, 2,), (1,)),
            ((1, 4,), (3,)),
            ((1, 10,), (9,)),
            ((10, 2,), (1,)),
            ((10, 4,), (3,)),
            ((10, 10,), (9,)),
            ((1000, 2,), (1,)),
            ((1000, 4,), (3,)),
            ((1000, 10,), (9,)),

            ((2,), (1, 1,)),
            ((4,), (1, 3,)),
            ((10,), (1, 9,)),
            ((2,), (10, 1,)),
            ((4,), (10, 3,)),
            ((10,), (10, 9,)),
            ((2,), (1000, 1,)),
            ((4,), (1000, 3,)),
            ((10,), (1000, 9,)),

            ((1, 2,), (1, 1,)),
            ((1, 4,), (1, 3,)),
            ((1, 10,), (1, 9,)),

            ((100, 2,), (100, 1,)),
            ((100, 4,), (100, 3,)),
            ((100, 10,), (100, 9,)),

            ((100, 1, 2,), (200, 1,)),
            ((100, 1, 4,), (200, 3,)),
            ((100, 1, 10,), (200, 9,)),

            ((100, 2,), (200, 1, 1,)),
            ((100, 4,), (200, 1, 3,)),
            ((100, 10,), (200, 1, 9,)),

            ((1, 7, 1, 2,), (3, 1, 10, 1,)),
            ((1, 7, 1, 4,), (3, 1, 10, 3,)),
            ((1, 7, 1, 10,), (3, 1, 10, 9,)),
        ]

        for vec_st_size, boost_vel_s_size in test_cases:
            ndim = vec_st_size[-1]
            vec_st = 100 * np.random.randn(*vec_st_size)
            boost_vel_s = (2 * np.random.rand(*boost_vel_s_size) - 1) * (1 / ((ndim - 1) ** 0.5)) / 2
            res = st.boost(vec_st, boost_vel_s)
            res_old = st.boost(vec_st, boost_vel_s, _old=True)
            self.assertTrue(np.isclose(res, res_old).all())

    # Test `st.boost_velocity_s` against
    # `st.velocity_s(st.boost(st.velocity_st(...), ...)`
    def test_boost_velocity_s_against_boost(self):
        test_cases = [
            # vel_s_size, boost_vel_s_size
            ((1,), (1,)),
            ((3,), (3,)),
            ((9,), (9,)),

            ((1, 1,), (1,)),
            ((1, 3,), (3,)),
            ((1, 9,), (9,)),
            ((10, 1,), (1,)),
            ((10, 3,), (3,)),
            ((10, 9,), (9,)),
            ((1000, 1,), (1,)),
            ((1000, 3,), (3,)),
            ((1000, 9,), (9,)),

            ((1,), (1, 1,)),
            ((3,), (1, 3,)),
            ((9,), (1, 9,)),
            ((1,), (10, 1,)),
            ((3,), (10, 3,)),
            ((9,), (10, 9,)),
            ((1,), (1000, 1,)),
            ((3,), (1000, 3,)),
            ((9,), (1000, 9,)),

            ((1, 1,), (1, 1,)),
            ((1, 3,), (1, 3,)),
            ((1, 9,), (1, 9,)),

            ((100, 1,), (100, 1,)),
            ((100, 3,), (100, 3,)),
            ((100, 9,), (100, 9,)),

            ((100, 1, 1,), (200, 1,)),
            ((100, 1, 3,), (200, 3,)),
            ((100, 1, 9,), (200, 9,)),

            ((100, 1,), (200, 1, 1,)),
            ((100, 3,), (200, 1, 3,)),
            ((100, 9,), (200, 1, 9,)),

            ((1, 7, 1, 1,), (3, 1, 10, 1,)),
            ((1, 7, 1, 3,), (3, 1, 10, 3,)),
            ((1, 7, 1, 9,), (3, 1, 10, 9,)),
        ]

        for vel_s_size, boost_vel_s_size in test_cases:
            ndim = vel_s_size[-1] + 1
            vel_s = (2 * np.random.rand(*vel_s_size) - 1) * (1 / ((ndim - 1) ** 0.5)) / 2
            boost_vel_s = (2 * np.random.rand(*boost_vel_s_size) - 1) * (1 / ((ndim - 1) ** 0.5)) / 2
            res = st.boost_velocity_s(vel_s, boost_vel_s)
            res_check = st.velocity_s(st.boost(st.velocity_st(vel_s), boost_vel_s))
            self.assertTrue(np.isclose(res, res_check).all())

    # Test boosting events in one spatial dimension with randomized inputs
    def test_boost_event_1D_random(self):
        v_batch = []
        event_batch = []
        event_out_batch = []

        for _ in range(10):
            v = np.random.uniform(low=0.1, high=1.0, size=()).astype(np.double)
            event = np.random.uniform(low=-1000, high=1000, size=(2,)).astype(np.double)

            event_out = st.boost(event, v)

            event_out_check = check_boost_event_1D(v, event)
            assert np.isclose(event_out, event_out_check).all()

            v_batch.append(v)
            event_batch.append(event)
            event_out_batch.append(event_out)

        # Test batched mode
        v = np.array(v_batch, dtype=np.double)
        event = np.array(event_batch, dtype=np.double)
        event_out_check = np.array(event_out_batch, dtype=np.double)

        event_out = st.boost(event, v)

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

            self.assertTrue(np.isclose(u_out, u_out_check).all())

            v_batch.append([v])
            u_batch.append([u])
            # TODO: Need to avoid adding a dim to output if inputs are scalar
            u_out_batch.append([u_out])

        # Test batched mode
        v = np.array(v_batch, dtype=np.double)
        u = np.array(u_batch, dtype=np.double)
        u_out_check = np.array(u_out_batch, dtype=np.double)

        u_out = st.boost_velocity_s(u, v)

        self.assertTrue(np.isclose(u_out, u_out_check).all())

    # Test `spacetimelib.boost` with lots of different input shapes
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

            x_out = st.boost(x, frame_v)
            u_out = st.boost_velocity_s(u, frame_v)

            self.assertEqual(x_out_shape_check, x_out.shape)
            self.assertEqual(u_out_shape_check, u_out.shape)

    def test_Worldline_vertex(self):
        test_cases = [
            [   [0, 0],
                [10, 4.5],
            ],
            [   [-1000, 100],
                [-304, -100],
                [-10, -250],
            ],
            [   [1, 2, 3],
                [10, -2, 4],
                [20, 0, 0],
            ]
        ]
        for vertices in test_cases:
            w = st.Worldline(vertices)
            self.assertEqual(len(vertices), len(w))

            for idx in range(len(w)):
                self.assertTrue((w.vertex(idx) == vertices[idx]).all())

                # Check index wrapping
                self.assertTrue((w.vertex(idx) == w.vertex(-len(w) + idx)).all())

            # Test that error is thrown if index is out of bounds
            with self.assertRaisesRegex(IndexError, r'out of bounds'):
                w.vertex(len(w))

            with self.assertRaisesRegex(IndexError, r'out of bounds'):
                w.vertex(-len(w) - 1)

    def test_Worldline_eval(self):
        w0 = st.Worldline(
            [[0, 0]],
            [0.9])
        w1 = st.Worldline(
            [[0, 0]],
            past_vel_s=[0.7],
            future_vel_s=[-0.1])
        w2 = st.Worldline(
            [[-100, 1], [10, -50]],
            [-0.4])
        w3 = st.Worldline(
            [[-100, 1], [10, -50]],
            past_vel_s=[0.4],
            future_vel_s=[-0.1])
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

    def test_Worldline_proper_time_delta(self):
        w0 = st.Worldline([[0, 0], [1, 0.9], [2, 0]])
        tau0 = (2 ** 2 - 1.8 ** 2) ** 0.5

        test_cases = [
            # worldline, time0, time1, tau_check
            (w0, 0, 2, tau0),
            (w0, 0, 1, tau0 / 2),
            (w0, 1, 0, -tau0 / 2),
            (w0, 1, 2, tau0 / 2),
            (w0, 2, 1, -tau0 / 2),
            (w0, 0.5, 1.5, tau0 / 2),
            (w0, 0.25, 0.75, tau0 / 4),
            (w0, 1.75, 1.25, -tau0 / 4),
            (w0, 0, 0, 0),
            (w0, 0, 0.123, 0.123 * tau0 / 2),
            (w0, 0.123, 0.123, 0),
            (w0, 1, 1, 0),
            (w0, 1.234, 1.234, 0),
            (w0, 2, 2, 0),
        ]

        for w, time0, time1, tau_check in test_cases:
            self.assertAlmostEqual(w.proper_time_delta(time0, time1), tau_check)
            self.assertAlmostEqual(w.proper_time_delta(time1, time0), -tau_check)

    def test_Worldline_boost(self):
        # TODO: Should test `ends_vel_s` too
        test_worldlines = [
            st.Worldline([
                [0, 0],
                [10, 4.5],
            ]),
            st.Worldline([
                [-1000, 100],
                [-304, -100],
                [-10, -250],
            ]),
            st.Worldline([
                [1, 2, 3],
                [10, -2, 4],
                [20, 0, 0],
            ])
        ]

        # TODO: This is probably a generally useful function for testing.
        # Should probably create a place to put testing utils
        def random_vel_s(n_space_dims):
            # Pick a random direction
            direction_s = np.random.randn(n_space_dims)
            direction_s /= np.linalg.norm(direction_s)

            # Pick a random speed, 0.1 to 0.9
            speed = 0.1 + 0.8 * np.random.rand(1)

            return speed * direction_s

        for w0 in test_worldlines:
            boost_vel_s = random_vel_s(len(w0.vertex(0)) - 1)
            w1 = w0.boost(boost_vel_s)

            self.assertEqual(len(w0), len(w1))

            for v_idx in range(len(w0)):
                v0 = w0.vertex(v_idx)
                v1 = w1.vertex(v_idx)

                v1_check = st.boost(v0, boost_vel_s)
                v0_check = st.boost(v1, -boost_vel_s)

                self.assertTrue(np.isclose(v0, v0_check).all())
                self.assertTrue(np.isclose(v1, v1_check).all())

    def test_Worldline_ends_vel(self):
        w0 = st.Worldline([[0, 0]], ends_vel_s=[0.1])
        self.assertTrue((w0.past_vel_s == 0.1).all())
        self.assertTrue((w0.past_vel_st == st.velocity_st(0.1)).all())
        self.assertTrue((w0.future_vel_s == 0.1).all())
        self.assertTrue((w0.future_vel_st == st.velocity_st(0.1)).all())

        w1 = st.Worldline([[0, 0]], past_vel_s=[0.1], future_vel_s=[0.1])
        self.assertTrue((w1.past_vel_s == 0.1).all())
        self.assertTrue((w1.past_vel_st == st.velocity_st(0.1)).all())
        self.assertTrue((w1.future_vel_s == 0.1).all())
        self.assertTrue((w1.future_vel_st == st.velocity_st(0.1)).all())

        w2 = st.Worldline([[0, 0]], past_vel_s=[-0.2], future_vel_s=None)
        self.assertTrue((w2.past_vel_s == -0.2).all())
        self.assertTrue((w2.past_vel_st == st.velocity_st(-0.2)).all())
        self.assertTrue(w2.future_vel_s is None)
        self.assertTrue(w2.future_vel_st is None)

        w3 = st.Worldline([[0, 0]], past_vel_s=None, future_vel_s=[0.3])
        self.assertTrue(w3.past_vel_s is None)
        self.assertTrue(w3.past_vel_st is None)
        self.assertTrue((w3.future_vel_s == 0.3).all())
        self.assertTrue((w3.future_vel_st == st.velocity_st(0.3)).all())

        v4 = [-0.13, 0.1, 0, -0.14]
        w4 = st.Worldline([[0, 0, 0, 0, 0]], ends_vel_s=v4)
        self.assertTrue((w4.past_vel_s == v4).all())
        self.assertTrue((w4.past_vel_st == st.velocity_st(v4)).all())
        self.assertTrue((w4.future_vel_s == v4).all())
        self.assertTrue((w4.future_vel_st == st.velocity_st(v4)).all())

        v5_past = [-0.13, 0.1, 0, -0.14]
        v5_future = [0.16, -0.14, 0.01, -0.046]
        w5 = st.Worldline([[0, 0, 0, 0, 0]], past_vel_s=v5_past, future_vel_s=v5_future)
        self.assertTrue((w5.past_vel_s == v5_past).all())
        self.assertTrue((w5.past_vel_st == st.velocity_st(v5_past)).all())
        self.assertTrue((w5.future_vel_s == v5_future).all())
        self.assertTrue((w5.future_vel_st == st.velocity_st(v5_future)).all())

        v6_past = [-0.13, 0.1, 0, -0.14]
        w6 = st.Worldline([[0, 0, 0, 0, 0]], past_vel_s=v6_past, future_vel_s=None)
        self.assertTrue((w6.past_vel_s == v6_past).all())
        self.assertTrue((w6.past_vel_st == st.velocity_st(v6_past)).all())
        self.assertTrue(w6.future_vel_s is None)
        self.assertTrue(w6.future_vel_st is None)

        v7_past = [-0.13, 0.1, 0, -0.14]
        v7_future = [0.16, -0.14, 0.01, -0.046]
        w7 = st.Worldline([[0, 0, 0, 0, 0]], past_vel_s=None, future_vel_s=v7_future)
        self.assertTrue(w7.past_vel_s is None)
        self.assertTrue(w7.past_vel_st is None)
        self.assertTrue((w7.future_vel_s == v7_future).all())
        self.assertTrue((w7.future_vel_st == st.velocity_st(v7_future)).all())

        w8 = st.Worldline([[0, 0, 0, 0, 0]], past_vel_s=None, future_vel_s=None)
        self.assertTrue(w8.past_vel_s is None)
        self.assertTrue(w8.past_vel_st is None)
        self.assertTrue(w8.future_vel_s is None)
        self.assertTrue(w8.future_vel_st is None)

        w9 = st.Worldline([[0, 0, 0, 0, 0]], ends_vel_s=None)
        self.assertTrue(w9.past_vel_s is None)
        self.assertTrue(w9.past_vel_st is None)
        self.assertTrue(w9.future_vel_s is None)
        self.assertTrue(w9.future_vel_st is None)

    def test_Worldline___eq__(self):
        # These worldlines all differ from each other in different ways
        worldlines = [
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5], [20, 7, 8]]),
            lambda: st.Worldline([[0, 1], [2, 1], [4, 2]]),
            lambda: st.Worldline([[0, 1], [2, 1]]),
            lambda: st.Worldline([[0.1, 1, 2], [10, 4, 5]]),
            lambda: st.Worldline([[0, 1, 2], [10, 3.9, 5]]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], past_vel_s=[0.1, -0.1]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], past_vel_s=[0.2, -0.1]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], future_vel_s=[0.2, 0.3]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], future_vel_s=[0.2, -0.3]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], past_vel_s=[0.1, -0.1], future_vel_s=[0.2, 0.3]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], past_vel_s=[0.1, -0.1], future_vel_s=[0.2, -0.3]),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], proper_time_origin=5),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], proper_time_origin=6),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], proper_time_offset=-1),
            lambda: st.Worldline([[0, 1, 2], [10, 4, 5]], proper_time_offset=1),
        ]

        for idx, gen in enumerate(worldlines):
            worldline = gen()

            for other_idx, other_gen in enumerate(worldlines):
                other = other_gen()

                if idx == other_idx:
                    self.assertEqual(worldline, other)
                else:
                    self.assertNotEqual(worldline, other)

    def test_Worldline_eval_vel_s(self):
        w = st.Worldline([
            [-20, 0, 0],
            [-18, 0.8, -0.1],
            [-16, 0.8 + 1.2, -0.1 + 0.2],
            [0, 2 - 10, 0.1],
            [2, -8, 0.1 - 1.9],
            [4, -8 - 1, -1.8 - 0.1]],
            past_vel_s=[-0.9, 0],
            future_vel_s=[0.7, -0.7])

        test_times = [
            # time, expected_vel
            (-float('inf'), np.array([-0.9, 0])),
            (-100, np.array([-0.9, 0])),
            (-20.0001, np.array([-0.9, 0])),
            (-20, np.array([0.8, -0.1]) / 2),
            (-19, np.array([0.8, -0.1]) / 2),
            (-18.0001, np.array([0.8, -0.1]) / 2),
            (-18, np.array([1.2, 0.2]) / 2),
            (-16.0001, np.array([1.2, 0.2]) / 2),
            (-16, np.array([-10, 0]) / 16),
            (-0.0001, np.array([-10, 0]) / 16),
            (0, np.array([0, -1.9]) / 2),
            (1.9999, np.array([0, -1.9]) / 2),
            (2, np.array([-1, -0.1]) / 2),
            (3.9999, np.array([-1, -0.1]) / 2),
            (4, np.array([0.7, -0.7])),
            (40, np.array([0.7, -0.7])),
            (float('inf'), np.array([0.7, -0.7])),
        ]

        for time, expected_vel in test_times:
            vel = w.eval_vel_s(time)
            self.assertTrue(np.isclose(vel, expected_vel).all())

    def test_Worldline_eval_proper_time(self):
        w0 = st.Worldline([[0, 0, 0]], ends_vel_s=[0, 0])
        w1 = st.Worldline([[-1.12, 1.234, 8.01]], ends_vel_s=[-0.124, 0.234])

        w2 = st.Worldline([
            [-20, 0, 0],
            [-18, 0.8, -0.1],
            [-16, 0.8 + 1.2, -0.1 + 0.2],
            [0, 2 - 10, 0.1],
            [2, -8, 0.1 - 1.9],
            [4, -8 - 1, -1.8 - 0.1]],
            past_vel_s=[-0.9, 0],
            future_vel_s=[0.7, -0.7])

        test_cases = [
            # worldline, time, proper_time_delta
            (w0, -10, 1),
            (w0, 0, 0),
            (w0, 0, 10),
            (w0, 1, 10),

            (w1, -10, 0),
            (w1, -12, 1),
            (w1, -10, 9),
            (w1, -9, 9),
            (w1, -2, 7),
            (w1, 0, 0),
            (w1, 0, 10),
            (w1, 1, 10),
            # Test a case where the result lands on a vertex
            (w1, -9, w1.proper_time(w1.vertex(0)[0]) - w1.proper_time(-9)),

            (w2, -40, 1000),
            (w2, 0, 0.01),
            (w2, -0.01, 2),
            (w2, -0.01, 0.0001),
            (w2, 3.8, 10),
            (w2, 3.8, 0.0001),
            (w2, 4, 10),
            (w2, 5, 1.123),
        ]

        for w, time, proper_time_delta in test_cases:
            event = w.eval_proper_time(time, proper_time_delta)
            proper_time_delta_check = w.proper_time(event[0]) - w.proper_time(time)
            self.assertTrue(np.isclose(proper_time_delta, proper_time_delta_check))
            event_check = w.eval(event[0])
            self.assertTrue(np.isclose(event, event_check).all())

    def test_Frame_add(self):
        worldlines = [
            st.Worldline([[0, 0, 0]]),
            st.Worldline([[1, 3, -4]], ends_vel_s=[0.9, 0]),
            st.Worldline([
                    [-100, 38, -29],
                    [-20, 15, 15],
                    [0, 9.123, 2.6]
                ],
                past_vel_s=[-.7, 0.69],
                proper_time_origin=-900,
                proper_time_offset=123),
        ]

        frame0 = st.Frame(worldlines)

        offsets = [
            np.array([0, 0, 0]),
            np.array([-10, 0, 0]),
            np.array([100, -30, 802]),
        ]

        for offset in offsets:
            frame1 = frame0 + offset

            for idx in range(len(frame0)):
                self.assertEqual(frame0.name(idx), frame1.name(idx))

                w0 = frame0[idx]
                w1 = frame1[idx]

                self.assertTrue(maybe_arraylike_equal(w0.past_vel_s, w1.past_vel_s))
                self.assertTrue(maybe_arraylike_equal(w0.future_vel_s, w1.future_vel_s))
                self.assertTrue(np.isclose(
                    w0.proper_time_origin + offset[0],
                    w1.proper_time_origin))
                self.assertTrue(w0.proper_time_offset == w1.proper_time_offset)

                for vert_idx in range(len(w0)):
                    self.assertTrue(np.isclose(
                        w0.vertex(vert_idx) + offset,
                        w1.vertex(vert_idx)).all())

    def test_Frame_eval(self):
        worldlines = [
            st.Worldline([[0, 0, 0]], ends_vel_s=[0, 0]),
            st.Worldline([[0, 0, 0]], past_vel_s=[-0.123, 0.01], future_vel_s=[0, 0.999]),
            st.Worldline([[1, 3, -4]], ends_vel_s=[0.9, 0]),
            st.Worldline([
                    [-100, 38, -29],
                    [-20, 15, 15],
                    [0, 9.123, 2.6]
                ],
                past_vel_s=[-0.7, 0.69],
                future_vel_s=[-0.4, -0.5],
                proper_time_origin=-900,
                proper_time_offset=123),
            st.Worldline([
                    [-34.23, 60, 70],
                    [-15.234, 59.342, 71.324],
                    [0.1, 61.2309, 69.34902],
                    [10, 58.24, 70.1],
                    [1000, 56.3094, 80.234],
                ],
                ends_vel_s=[0, 0]),
        ]

        eval_times = [
            -1000, -100, -50, -30, -20, -15.23, -10, -10.123, -1, 0, 0.1, 1, 2, 12,
            50, 100, 999, 1000, 10000,
        ]

        frame = st.Frame(worldlines)

        for eval_time in eval_times:
            state = frame.eval(eval_time)

            for idx, (name, event, proper_time) in enumerate(state):
                self.assertEqual(frame.name(idx), name)

                event_expected = worldlines[idx].eval(eval_time)
                self.assertTrue((event == event_expected).all())

                proper_time_expected = worldlines[idx].proper_time(eval_time)
                self.assertEqual(proper_time, proper_time_expected)

    def test_Frame_boost(self):
        worldlines = [
            st.Worldline([[0, 0, 0]]),
            st.Worldline([[0, 0, 0]], ends_vel_s=[0, 0]),
            st.Worldline([[0, 0, 0]], past_vel_s=[-0.123, 0.01], future_vel_s=[0, 0.999]),
            st.Worldline([[1, 3, -4]], ends_vel_s=[0.9, 0]),
            st.Worldline([
                    [-100, 38, -29],
                    [-20, 15, 15],
                    [0, 9.123, 2.6]
                ],
                past_vel_s=[-.7, 0.69],
                proper_time_origin=-900,
                proper_time_offset=123),
            st.Worldline([
                    [-34.23, 60, 70],
                    [-15.234, 59.342, 71.324],
                    [0.1, 61.2309, 69.34902],
                    [10, 58.24, 70.1],
                    [1000, 56.3094, 80.234],
                ],
                ends_vel_s=[0, 0]),
        ]

        frame0 = st.Frame(worldlines)

        boost_vels = [
            [0, 0],
            [0.9, 0],
            [0, -0.99],
            [0.734, -0.6342],
        ]

        event_deltas = [
            # event_delta_pre, event_delta_post
            (None, None),
            ([0, 0, 0], [0, 0, 0]),
            ([1, 2, 3], None),
            (None, [1, 2, 3]),
            ([-123, 39, -23], [53.1, 89.2, -23.55]),
        ]

        for boost_vel_s, (event_delta_pre, event_delta_post), batched in product(boost_vels, event_deltas, [True, False]):
            frame1 = frame0.boost(
                boost_vel_s,
                event_delta_pre,
                event_delta_post,
                _batched=batched)

            self.assertEqual(len(frame0), len(frame1))

            for idx in range(len(frame1)):
                self.assertEqual(frame0.name(idx), frame1.name(idx))
                w0 = frame0[idx]
                w1 = frame1[idx]

                if event_delta_pre is not None:
                    w0_ = w0 + event_delta_pre
                else:
                    w0_ = w0

                w1_check_ = w0_.boost(boost_vel_s)

                if event_delta_post is not None:
                    w1_check = w1_check_ + event_delta_post
                else:
                    w1_check = w1_check_

                self.assertEqual(w1, w1_check)

if __name__ == '__main__':
    unittest.main()
