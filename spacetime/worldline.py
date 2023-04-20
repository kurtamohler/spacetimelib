from .basic_ops import boost, _proper_time, boost_velocity_s
from .error_checking import check, internal_assert

import numpy as np
import numbers

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

    def __init__(self,
            vertices,
            vel_ends=None,
            *,
            vel_past=None,
            vel_future=None,
            proper_time_origin=None,
            proper_time_offset=0):
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

            Size: (N+1)

            Default: ``None``

        Keyword args:

          vel_past (array_like, optional):
            Space-velocity of the worldline before the first vertex. If
            specified, ``vel_ends`` must be ``None``.

            Size: (N+1)

            Default: ``None``

          vel_future (array_like, optional):
            Space-velocity of the worldline after the last vertex. If
            specified, ``vel_ends`` must be ``None``.

            Size: (N+1)

            Default: ``None``

          proper_time_origin (number, optional):
            The proper time origin for the worldline. This is the coordinate
            time at which a stopwatch traveling along the worldline has the
            value ``proper_time_offset``. By default, the coordinate time of
            the first vertex in the worldline is chosen.

            Default: ``vertices[0][0]``

          proper_time_offset (number, optional):
            The value that a stopwatch traveling along the worldline has at
            the coordinate time ``proper_time_origin``.

            Default: ``0``
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

            # TODO: I may want to rename `_vel_ends` since its meaning is actually
            # not the same as `vel_ends`. `vel_ends` is one velocity, and `_vel_ends`
            # is two velocities
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

        if proper_time_origin is None:
            proper_time_origin = self._vertices[0][0].item()
        else:
            check(isinstance(proper_time_origin, numbers.Number), TypeError,
                "expected 'proper_time_origin' to be float or int, but got ",
                f"{type(proper_time_origin)}")

        # Need to check that the proper time origin is actually within the bounds of
        # worldline
        first_time = -float('inf') if self._vel_ends[0] is not None else self._vertices[0][0].item()
        last_time = float('inf') if self._vel_ends[1] is not None else self._vertices[-1][0].item()

        check(proper_time_origin >= first_time and proper_time_origin <= last_time,
            ValueError,
            f"expected 'proper_time_origin' to be between {first_time} and ",
            f"{last_time}, the first and last time coordinates in the worldline, ",
            f"but got {proper_time_origin}")

        self._proper_time_origin = proper_time_origin

        check(isinstance(proper_time_offset, numbers.Number), TypeError,
            "expected 'proper_time_offset' to be float or int, but got ",
            f"{type(proper_time_offset)}")
        self._proper_time_offset = proper_time_offset

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

    # TODO: I suppose `proper_time` should now just take one arg and find the
    # proper time between `proper_time_origin` and the specified time. Then to
    # get the proper time between two arbitrary points on the worldline, you
    # can just call `proper_time` twice with the two different coord times and
    # take the difference between them. Probably should make `proper_time_diff`
    # function that does that.
    def proper_time(self, time):
        '''
        Measure the proper time along a section of the worldline between
        :attr:``Worldline.proper_time_origin`` and a specified time coordinate.

        Args:

          time (number):
            Time coordinate

        Returns:
          number:
        '''
        self.proper_time_diff(self.proper_time_origin, time)

    def proper_time_diff(self, time0, time1):
        '''
        Measure the proper time along a section of the worldline between two
        specified time coordinates. Note that if ``time1 < time0``, result is
        negative.

        Args:

          time0 (number):
            First time coordinate

          time1 (number):
            Second time coordinate

        Returns:
          number:
        '''
        sign = 1

        # TODO: I'm not a fan of this. Make it simpler
        if time0 > time1:
            tmp = time0
            time0 = time1
            time1 = tmp
            sign = -1

        first_event, first_indices = self.eval(time0, return_indices=True)
        last_event, last_indices = self.eval(time1, return_indices=True)

        if first_indices == last_indices:
            return sign * _proper_time(first_event, last_event)

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

            return sign * res

    def boost(self, boost_vel_s):
        '''
        Boost the worldline to a different inertial reference frame. The vertices,
        past and future velocities, and :attr:``Worldline.proper_time_origin`` are
        all boosted.

        Args:

          boost_vel_s (array_like):
            Space-velocity to boost the worldline by.

        Returns:
          :class:`spacetime.Worldline`:
        '''
        vertices = boost(self._vertices, boost_vel_s)
        vel_ends = [None, None]

        for idx in [0, 1]:
            if self._vel_ends[idx] is not None:
                vel_ends[idx] = boost_velocity_s(
                    self._vel_ends[idx],
                    boost_vel_s)


        return Worldline(
            vertices,
            # TODO: I guess only evaluating this once would be better
            proper_time_origin=boost(self.eval(self._proper_time_origin), boost_vel_s)[0].item(),
            vel_past=vel_ends[0],
            vel_future=vel_ends[1])

    def __add__(self, event_delta):
        '''
        Add a displacement to all events in the worldline.

        Args:

          event_delta (array_like):
            Displacements to add to each dimension.

        Returns:
          :class:`spacetime.Worldline`:
        '''
        event_delta = np.asarray(event_delta)
        check(event_delta.shape == self._vertices[0].shape, ValueError,
            f"'event_delta' must have shape {self._vertices[0].shape}, but got ",
            f"{event_delta.shape}")

        return Worldline(
            self._vertices + event_delta,
            vel_past=self._vel_ends[0],
            vel_future=self._vel_ends[1],
            proper_time_origin=self._proper_time_origin + event_delta[0].item())

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

    def __len__(self):
        '''
        Get the number of vertices in the worldline.

        Returns:
          int:
        '''
        return len(self._vertices)

    def vertex(self, idx):
        '''
        Get the vertex at the specified index.

        Args:

            idx (int):
              The index of the vertex to get.

        Returns:
          ndarray: Size: (N+1,)
        '''
        check(isinstance(idx, int), TypeError,
            f'idx must be an int, but got {type(idx)}')
        return self._vertices[idx].copy()

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

    @property
    def proper_time_origin(self):
        '''
        Get the proper time origin of the worldline.

        Returns:
          int or float:
        '''
        return self._proper_time_origin

    @property
    def proper_time_offset(self):
        '''
        Get the proper time offset of the worldline.

        Returns:
          int or float:
        '''
        return self._proper_time_offset

    @property
    def ndim(self):
        '''
        Get the number of spatial plus time dimensions, N+1, for this worldline.

        Returns:
            int: N+1
        '''
        return self._vertices.shape[-1]
