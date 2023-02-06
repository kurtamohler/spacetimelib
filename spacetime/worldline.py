from .basic_ops import boost, _proper_time
from .error_checking import check, internal_assert

import numpy as np

class Worldline:
    '''
    The worldline of an object. It is represented as a finite set of events,
    called vertices, within an inertial reference frame connected together by
    straight time-like line segments. Events between a pair of vertices are
    evaluated using basic linear interpolation. Continuous worldlines can be
    approximated, with accuracy proportional to the density of vertices.

    By default, the first and last vertices are treated as end point
    boundaries, past which events cannot be evaluated. Alternatively, the
    ``vel_ends``, ``vel_past``, or ``vel_future`` arguments can be specified to
    enable linear extrapolation of events that fall outside of these
    boundaries.
    '''

    # TODO: Would be cool to add an option to enable infinite loops over the
    # positions in the vertices.

    # TODO: Investigate using SymPy to enable continuous worldlines.

    def __init__(self, vertices, vel_ends=None, *, vel_past=None, vel_future=None):
        '''
        Args:

          vertices (array_like):
            Set of events to use as the vertices of the worldline. Events
            must be sorted by increasing time coordinate, and each pair of
            events must have time-like separation.

          vel_ends (array_like, optional):
            Velocity of the worldline before and after the first and last
            vertices. This enables the extrapolation of events that occur
            before and after the first and last ``vertices``.

            Default: ``None``

        Keyword args:

          vel_past (array_like, optional):
            Velocity of the worldline before the first vertex. If specified,
            ``vel_ends`` must be ``None``.

            Default: ``None``

          vel_future (array_like, optional):
            Velocity of the worldline after the last vertex. If specified,
            ``vel_ends`` must be ``None``.

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
        Returns the event at a specified time on the worldline.

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

    def _proper_time(self, time0, time1):
        '''
        Measure the proper time span across a section of the worldline between
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
          spacetime.Worldline:
        '''
        vertices = boost(frame_velocity, self._vertices)
        vel_ends = [None, None]

        for idx in [0, 1]:
            if self._vel_ends[idx] is not None:
                vel_ends[idx] = boost_velocity_s(
                    self._vel_ends[idx],
                    frame_velocity)

        return Worldline(vertices, vel_ends)

    def __add__(self, event_delta):
        '''
        Add a displacement to all events in the worldline.

        Args:

          event_delta (array_like):
            Displacements to add to each dimension.

        Returns:
          spacetime.Worldline:
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
          spacetime.Worldline:
        '''
        return self + (-event_delta)
