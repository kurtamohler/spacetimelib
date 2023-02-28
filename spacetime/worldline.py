from .basic_ops import boost, _proper_time
from .error_checking import check, internal_assert

import numpy as np

class Worldline:
    '''
    The worldline of a particle is represented as a finite list of events,
    called "vertices", through which the particle passes in an inertial
    reference frame. The particle moves in straight lines between the vertices.

    Continuous worldlines can only be approximated, with accuracy proportional
    to the density of vertices.

    We can evaluate events along the worldline at any time between the vertices
    with :func:`Worldline.eval`.

    We can also boost an entire worldline into a different reference frame
    with :func:`Worldline.boost`.
    '''

    # TODO: Would be cool to add an option to enable infinite loops over the
    # positions in the vertices.

    # TODO: Investigate using SymPy to enable continuous worldlines.

    def __init__(self, vertices, vel_ends=None, *, vel_past=None, vel_future=None):
        '''
        Args:

          vertices (array_like):
            A list of spacetime-vectors to use as the vertices of the worldline.

            Events must be sorted by increasing time coordinate.

            Adjacent vertices must be separated by time-like or light-like
            intervals, since particles are limited by the speed of light. In
            other words, ``st.norm2_st(vertices[i+1] - vertices[i]) <= 0`` for
            all ``i``.

            By default, the first and last vertices are treated as end point
            boundaries, past which events simply cannot be evaluated. The
            ``vel_ends``, ``vel_past``, or ``vel_future`` arguments can be
            specified to enable linear extrapolation of events that fall
            outside of these boundaries.

            Size: (M, N+1) for M vertices that each have N+1 dimensions

          vel_ends (array_like, optional):
            Space-velocity of the worldline before and after the first and last
            vertices. This enables the extrapolation of events that occur
            before and after the first and last ``vertices``.

            If specified, ``vel_past`` and ``vel_future`` must be ``None``.

            Default: ``None``

        Keyword args:

          vel_past (array_like, optional):
            Space-velocity of the worldline before the first vertex. If
            specified, ``vel_ends`` must be ``None``.

            Default: ``None``

          vel_future (array_like, optional):
            Space-velocity of the worldline after the last vertex. If
            specified, ``vel_ends`` must be ``None``.

            Default: ``None``
        '''
        vertices = np.array(vertices)

        check(vertices.ndim == 2, ValueError,
            f"expected 'vertices' to have 2 dims, but got {vertices.ndim}")
        check(vertices.shape[-1] >= 2, ValueError,
            f"expected 'vertices.shape[-1] >= 2', but got {vertices.shape[-1]}")

        prev_event = vertices[0]

        # Check that each pair of vertices is in order of increasing time
        # coordinates and has time-like displacement
        for event_idx in range(1, vertices.shape[0]):
            cur_event = vertices[event_idx]

            # Time dimension must increase for each pair
            check(cur_event[0] > prev_event[0], ValueError,
                "expected 'vertices' to be ordered by increasing time coordinate")

            tau = _proper_time(prev_event, cur_event)

            check(tau >= 0, ValueError,
                "expected 'vertices' to all have time-like separation")

            prev_event = cur_event

        self._vertices = vertices

        num_spatial_dims = vertices.shape[-1] - 1

        def check_vel_end(arg_name, v):
            check(v.shape == (num_spatial_dims,), ValueError,
                f"expected `{arg_name}.shape == ({num_spatial_dims},)`, ",
                f"since `events` has {num_spatial_dims} spatial dimensions, but got ",
                f"`{v.shape}` instead")
            speed = np.linalg.norm(v)
            check(speed <= 1, ValueError,
                f"expected `{arg_name}` to have speed less than or equal ",
                f"to the speed of light, 1, but got {speed} instead")

        if vel_ends is not None:
            check(vel_past is None and vel_future is None, ValueError,
                "expected `vel_past` and `vel_future` to be None, since `vel_ends` was given")
            vel_ends = np.array(vel_ends)
            check_vel_end('vel_ends', vel_ends)
            self._vel_ends = [vel_ends, vel_ends]
        else:
            self._vel_ends = [None, None]

            if vel_past is not None:
                vel_past = np.array(vel_past)
                check_vel_end('vel_past', vel_past)
                self._vel_ends[0] = vel_past

            if vel_future is not None:
                vel_future = np.array(vel_future)
                check_vel_end('vel_future', vel_future)
                self._vel_ends[1] = vel_future

    def _find_surrounding_vertices(self, time):
        '''
        Find the two closest vertices surrounding a specified time coorindate.

        Args:

          time (number):
            The time coordinate.

        Returns:
          tuple of int: (idx_before, idx_after), indices into ``self._vertices``.

            If a vertex is at exactly ``time``, then it is returned as both
            ``idx_before`` and ``idx_after``, so ``idx_before == idx_after``.

            If there is no vertex before ``time``, then ``idx_before = None``.

            If there is no vertex after ``time``, then ``idx_after = None``.
        '''
        idx_after = np.searchsorted(self._vertices[..., 0], time)

        if idx_after >= len(self._vertices):
            return idx_after - 1, None

        elif self._vertices[idx_after][0] == time:
            return idx_after, idx_after

        elif idx_after == 0:
            return None, idx_after

        else:
            return idx_after - 1, idx_after

    def eval(self, time, return_indices=False):
        '''

        Calculates the event at a specified time on the worldline. The particle
        travels in straight lines between vertices, so we use simple linear
        interpolation for this.

        To evaluate times that are before or after all of the vertices,
        ``vel_ends``, ``vel_past``, or ``vel_future`` must have been
        specified in :func:`Worldline`.

        Args:

          time (number):
            Time at which to evaluate the worldline.

          return_indices (bool, optional):
            Whether to return the indices of vertices surrounding the specified time.

            Default: ``False``

        Returns:
          ``ndarray`` or ``tuple(ndarray, tuple(int, int))``:

            If ``return_indices == False``, just the evaluated event is
            returned.  If ``return_indices == True``, returns a 2-tuple
            containing the event combined with a 2-tuple of ints for the
            indices of vertices surrounding the specified ``time``.
        '''
        idx_before, idx_after = self._find_surrounding_vertices(time)

        if idx_before is None or idx_after is None:
            internal_assert(idx_before != idx_after)

            if idx_before is None:
                vel_ends = self._vel_ends[0]
                check(vel_ends is not None, ValueError,
                    f"time '{time}' is before the first event on the worldline at ",
                    f"time '{self._vertices[0][0]}'")
                vert = self._vertices[0]
            else:
                vel_ends = self._vel_ends[1]
                check(vel_ends is not None, ValueError,
                    f"time '{time}' is after the last event on the worldline at ",
                    f"time '{self._vertices[-1][0]}'")
                vert = self._vertices[-1]

            event = np.concatenate([[time],
                vert[1:] + vel_ends * (time - vert[0])])

        elif idx_before == idx_after:
            event = self._vertices[idx_before]

        else:
            v0 = self._vertices[idx_before]
            v1 = self._vertices[idx_after]
            delta_ratio = (time - v0[0]) / (v1[0] - v0[0])
            event = (v1 - v0) * delta_ratio + v0

        internal_assert(np.isclose(event[0], time).all())

        if return_indices:
            return event, (idx_before, idx_after)
        else:
            return event

    def proper_time(self, time0, time1):
        '''
        Measure the proper time along a section of the worldline between
        two specified time coordinates.

        Args:

          time0 (number):
            First time coordinate

          time1 (number):
            Second time coordinate

        Returns:
          number:
        '''
        if time0 > time1:
            tmp = time0
            time0 = time1
            time1 = tmp

        first_event, first_indices = self.eval(time0, return_indices=True)
        last_event, last_indices = self.eval(time1, return_indices=True)

        if first_indices == last_indices:
            return _proper_time(first_event, last_event)

        else:
            res = 0
            if first_indices[0] != first_indices[1]:
                res += _proper_time(first_event, self._vertices[first_indices[1]])

            for idx0 in range(first_indices[1], last_indices[0]):
                idx1 = idx0 + 1
                v0 = self._vertices[idx0]
                v1 = self._vertices[idx1]
                res += _proper_time(v0, v1)

            if last_indices[0] != last_indices[1]:
                res += _proper_time(self._vertices[last_indices[0]], last_event)

            return res

    def boost(self, frame_velocity):
        '''
        Boost the worldline to a different inertial reference frame.

        Args:

          frame_velocity (array_like):
            Velocity to boost the worldline by.

        Returns:
          :class:`spacetime.Worldline`:
        '''
        vertices = boost(self._vertices, frame_velocity)
        vel_ends = [None, None]

        for idx in [0, 1]:
            if self._vel_ends[idx] is not None:
                vel_ends[idx] = boost_velocity_s(
                    self._vel_ends[idx],
                    frame_velocity)

        return Worldline(vertices, vel_past=vel_ends[0], vel_future=vel_ends[1])

    def __add__(self, event_delta):
        '''
        Add a displacement to all events in the worldline.

        Args:

          event_delta (array_like):
            Displacements to add to each dimension.

        Returns:
          :class:`spacetime.Worldline`:
        '''
        return Worldline(
            self._vertices + event_delta,
            self._vel_ends)

    def __sub__(self, event_delta):
        '''
        Subtract a displacement from all events in the worldline.

        Args:

          event_delta (array_like):
            Displacements to subtract from each dimension.

        Returns:
          :class:`spacetime.Worldline`:
        '''
        return self + (-event_delta)

    # TODO: Should probably take a list (or array_like potentially) of dim
    # indices instead, to support extracting fewer or more dims than two,
    # if that is ever useful for people.
    def plot(self, dim0=1, dim1=0):
        '''
        Get an array that can be given directly to `matplotlib.pyplot.plot()
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_,
        to generate a 2D plot of the worldline in two of its N+1 dimensions.

        You will need to expand the array with `*`, like so:
        ``matplotlib.pyplot.plot(*worldline.plot())``

        Args:

          dim0 (int):
            First dimension to plot

            Default: 1

          dim1 (int):
            Second dimension to plot

            Default: 0

        Returns:

          ndarray: Size: (2, M) for M vertices
        '''
        check(isinstance(dim0, int), TypeError,
            f'dim0 must be an int, but got {type(dim0)}')
        check(isinstance(dim1, int), TypeError,
            f'dim1 must be an int, but got {type(dim1)}')

        num_dims = self._vertices.shape[1]
        check(
            dim0 >= 0 and dim0 < num_dims,
            TypeError,
            f'dim0 must be >=0 or <{num_dims}, but got {dim0}')
        check(
            dim1 >= 0 and dim1 < num_dims,
            TypeError,
            f'dim1 must be >=0 or <{num_dims}, but got {dim1}')

        vertices_t = self._vertices.transpose()

        return np.stack([
            vertices_t[dim0],
            vertices_t[dim1]])


