from .transformations import boost, time_distance, check

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
# `scipy.interpolate` probably has everything I would need.
class Worldline:
    def __init__(self, vertices, *, end_velocities=None):
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

            dist = time_distance(prev_event, cur_event)

            check(dist >= 0, ValueError,
                  "expected 'vertices' to all have time-like separation")

            prev_event = cur_event

        self._vertices = vertices

        # TODO: I don't like the name `end_velcities`
        if end_velocities is not None:
            raise NotImplementedError
            end_velocities = np.array(end_velocities)
            ndim_spatial = vertices.shape[-1] - 1
            check(end_velocities.shape[-1] == ndim_spatial, ValueError, (
                  f"Expected 'end_velocities.shape[-1]' to be equal to the number of "
                  f"spatial dims in 'vertices', {ndim_spatial}, but got "
                  f"{end_velocities.shape[-1]}"))
        else:
            self._velocity_before = None
            self._velocity_after = None

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


    # Approximate the event on the worldline at a specified time
    def interpolate(self, time, return_indices=False):
        idx_before, idx_after = self._find_surrounding_vertices(time)

        if idx_before is None:
            check(self._velocity_before is not None, ValueError,
                f"time '{time}' is before the first event on the worldline at "
                f"time '{self._vertices[0][0]}'")
            raise NotImplementedError
        elif idx_after is None:
            check(self._velocity_after is not None, ValueError,
                f"time '{time}' is after the last event on the worldline at "
                f"time '{self._vertices[-1][0]}'")
            raise NotImplementedError
        elif idx_before == idx_after:
            if return_indices:
                return self._vertices[idx_before], (idx_before, idx_after)
            else:
                return self._vertices[idx_before]

        v0 = self._vertices[idx_before]
        v1 = self._vertices[idx_after]
        delta_ratio = (time - v0[0]) / (v1[0] - v0[0])

        event = (v1 - v0) * delta_ratio + v0

        if return_indices:
            return event, (idx_before, idx_after)
        else:
            return event

    # Measure the proper time span across the section of the worldline between
    # two specified time coordinates.
    # TODO: Should choose a better name for this, probably
    def proper_time_range(self, time0, time1):
        first_event, first_indices = self.interpolate(time0, return_indices=True)
        last_event, last_indices = self.interpolate(time1, return_indices=True)

        proper_time = 0
        if first_indices[0] != first_indices[1]:
            proper_time += time_distance(first_event, self._vertices[first_indices[1]])

        for idx0 in range(first_indices[1], last_indices[0]):
            idx1 = idx0 + 1
            v0 = self._vertices[idx0]
            v1 = self._vertices[idx1]
            proper_time += time_distance(v0, v1)

        if last_indices[0] != last_indices[1]:
            proper_time += time_distance(self._vertices[last_indices[0]], last_event)

        return proper_time


