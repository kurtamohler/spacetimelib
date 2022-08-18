from .basic_ops import boost, proper_time
from .error_checking import check, internal_assert

import numpy as np

# The worldline of an object in spacetime. It is represented as a finite set of
# events, called vertices, in a common inertial reference frame, sorted by
# increasing time coordinate, all connected together by time-like straight line
# segments.  Continuous worldlines can be approximated, and the the accuracy of
# the approximation is proportional to the density of vertices and inversely
# proportional to the complexity of the actual worldline being approximated.
#
# Since a `Worldline` is just a set of vertices, it can be transformed in all the
# ways that an event can be transformed. It can be Lorentz boosted or
# displaced, as long as the same transformation is applied to every
# single event in the `Worldline`.
#
# The first and last of the vertices in a `Worldline` can either be a finite
# end point or it can be a straight line of some arbitrary velocity for
# infinite time in the appropriate direction. TODO: Think of a better way to
# word this.
#
# TODO: Would be great to offer different interpolation methods. I think
# `scipy.interp` probably has everything I would need.
class Worldline:

    # Args:
    #
    #   vertices : array
    #       Set of events to use as the vertices of the worldline.
    #
    # Keyword args:
    #
    #   end_velocities : 2-tuple of None or array
    #       The velocities to use for evaluation of time coordinates before or
    #       after all the events in `vertices`. `end_velocities[0]` is used
    #       extrapolate into the past, and `end_velocities[1]` is used to
    #       extrapolate into the future. If `end_velocities[i] is None`, then
    #       evaluating into that region throws an error.
    def __init__(self, vertices, *, end_velocities=(None, None)):
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

            tau = proper_time(prev_event, cur_event)

            check(tau >= 0, ValueError,
                "expected 'vertices' to all have time-like separation")

            prev_event = cur_event

        self._vertices = vertices

        num_spatial_dims = vertices.shape[-1] - 1

        # TODO: I don't like the name `end_velocities`
        check(isinstance(end_velocities, (tuple, list)), TypeError,
            f"`end_velocities` must be a tuple, but got {type(end_velocities)} instead")
        check(len(end_velocities) == 2, IndexError,
            f"expected `len(end_velocities) == 2`, but got {len(end_velocities)} instead")

        self._end_velocities = [None, None]

        for idx, v in enumerate(end_velocities):
            if v is not None:
                v = np.array(v)
                check(v.shape == (num_spatial_dims,), ValueError,
                    f"expected `end_velocities[{idx}].shape == ({num_spatial_dims},)`, ",
                    f"since `events` has {num_spatial_dims} spatial dimensions, but got ",
                    f"`{v.shape}` instead")
                speed = np.linalg.norm(v)
                check(speed <= 1, ValueError,
                    f"expected `end_velocities[{idx}]` to have speed less than or equal ",
                    f"to the speed of light, 1, but got {speed} instead")
                self._end_velocities[idx] = v


    # Find the two closest vertices surrounding a specified time coorindate.
    #
    # Arguments:
    #
    #   time : number
    #       The time coordinate.
    #
    # Returns:
    #   idx_before, idx_after : tuple of int
    #       Indices into `self._vertices`.
    #
    # If a vertex is at exactly `time`, then it is returned as both `idx_before`
    # and `idx_after`, so `idx_before == idx_after`.
    #
    # If there is no vertex before `time`, then `idx_before = None`.
    #
    # If there is no vertex after `time`, then `idx_after = None`.
    def _find_surrounding_vertices(self, time):
        idx_after = np.searchsorted(self._vertices[..., 0], time)

        if idx_after >= len(self._vertices):
            return idx_after - 1, None

        elif self._vertices[idx_after][0] == time:
            return idx_after, idx_after

        elif idx_after == 0:
            return None, idx_after

        else:
            return idx_after - 1, idx_after


    # Returns the event at a specified time on the worldline.
    #
    # Args:
    #
    #   time : number
    #       Time at which to evaluate the worldline.
    #
    #   return_indices : bool, optional
    #       Whether to return the indices of vertices surrounding the specified time.
    #       Default: False
    #
    # Returns:
    #
    #   If `return_indices == False`:
    #       event : array
    #
    #   If `return_indices == True:
    #       event, (idx_before, idx_after) : array, (int, int)
    def eval(self, time, return_indices=False):
        idx_before, idx_after = self._find_surrounding_vertices(time)

        if idx_before is None or idx_after is None:
            internal_assert(idx_before != idx_after)

            if idx_before is None:
                end_velocity = self._end_velocities[0]
                check(end_velocity is not None, ValueError,
                    f"time '{time}' is before the first event on the worldline at ",
                    f"time '{self._vertices[0][0]}'")
                vert = self._vertices[0]
            else:
                end_velocity = self._end_velocities[1]
                check(end_velocity is not None, ValueError,
                    f"time '{time}' is after the last event on the worldline at ",
                    f"time '{self._vertices[-1][0]}'")
                vert = self._vertices[-1]

            event = np.concatenate([[time],
                vert[1:] + end_velocity * (time - vert[0])])

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

    # Measure the proper time span across a section of the worldline between
    # two specified time coordinates.
    def proper_time(self, time0, time1):
        if time0 > time1:
            tmp = time0
            time0 = time1
            time1 = tmp

        first_event, first_indices = self.eval(time0, return_indices=True)
        last_event, last_indices = self.eval(time1, return_indices=True)

        if first_indices == last_indices:
            return proper_time(first_event, last_event)

        else:
            res = 0
            if first_indices[0] != first_indices[1]:
                res += proper_time(first_event, self._vertices[first_indices[1]])

            for idx0 in range(first_indices[1], last_indices[0]):
                idx1 = idx0 + 1
                v0 = self._vertices[idx0]
                v1 = self._vertices[idx1]
                res += proper_time(v0, v1)

            if last_indices[0] != last_indices[1]:
                res += proper_time(self._vertices[last_indices[0]], last_event)

            return res

    def boost(self, frame_velocity):
        vertices = boost(frame_velocity, self._vertices)
        end_velocities = [None, None]

        for idx in [0, 1]:
            if self._end_velocities[idx] is not None:
                _, end_velocities[idx] = boost(
                    frame_velocity,
                    np.zeros_like(vertices[0]),
                    self._end_velocities[idx])

        return Worldline(vertices, end_velocities=end_velocities)

    def __add__(self, event_delta):
        return Worldline(
            self._vertices + event_delta,
            end_velocities=self._end_velocities)

    def __sub__(self, event_delta):
        return self + (-event_delta)
