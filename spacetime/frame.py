import numpy as np
from copy import copy

from .basic_ops import boost, boost_velocity_s
from .worldline import Worldline
from .error_checking import check


class Frame:
    '''
    An inertial reference frame in Minkowski spacetime.
    Only supports 2+1 spacetime at the moment.
    '''

    def __init__(self, worldlines=None):
        if worldlines is None:
            self._worldlines = []
        else:
            assert isinstance(worldlines, (list, tuple))
            for w in worldlines:
                assert isinstance(w, Worldline)
            self._worldlines = copy(worldlines)

    def append(self, worldline):
        assert isinstance(worldline, Worldline)
        self._worldlines.append(worldline)

    def get_state_at_time(self, time):
        '''
        Returns the proper times and events for all worldlines at the specified time.
        '''
        state = []

        # TODO: I'd like to parallelize this into one batched operation, but
        # I don't think that's very possible since worldlines can have all
        # different numbers of vertices. Still, there could be something that
        # would give better performance here.
        for w in self._worldlines:
            proper_time = w.proper_time(time)
            event = w.eval(time)
            state.append((proper_time, event))

        return state


    # TODO: `event_delta` should be None by default. Actually, probably shouldn't
    # even be here--instead, add an addition function and use that? But that would
    # give worse performance though...
    def boost(self, event_delta, velocity_delta):
        '''
        Transform the frame, applying a time and position translations first,
        then applying a velocity transformation.
        '''
        # Check `event_delta` arg
        event_delta = np.array(event_delta)
        assert event_delta.shape == (3,)

        # Check `velocity_delta` arg
        if velocity_delta is None:
            velocity_delta = np.array([0, 0])
        else:
            velocity_delta = np.array(velocity_delta)
            assert velocity_delta.shape == (2,)

        speed = np.linalg.norm(velocity_delta)
        # Don't allow faster than light transformations
        assert speed <= 1

        new_worldlines = []

        batched = False

        # TODO: While this impl of batching is roughly 3x faster on my machine
        # than the non-batched path while uniformly accelerating in
        # `examples/clock_grid.py`, it is still pretty inefficient to copy all
        # the events and velocities out of the worldlines and then back into
        # the new worldlines. It would be much more efficient to always keep
        # a batched representation of the frame cached. This would be very
        # simple to implement, just separate the batched representation
        # creation into a new function and use the `lru_cache` decorator (or
        # whatever it's called) to auto-cache it. Another possibility is that
        # events and velocities of the individual worldlines could potentially
        # be views into the batched representation, but that seems harder to
        # implement while preventing broken views.
        if batched:
            vertex_count = []
            vertices = []
            proper_time_origin_events = []

            batched_velocities = []

            # Map the worldline index to an index into `velocities`, for both the
            # past and future velocity
            past_velocity_idx_map = {}
            future_velocity_idx_map = {}

            # Need to grab all event vertices and velocities from all
            # worldlines and combine them into two batched event and velocity
            # arrays that can be boosted. Need to keep track of which vertices
            # and velocities belong to which worldlines.
            for w_idx, w in enumerate(self._worldlines):
                vertex_count.append(len(w._vertices))
                vertices += [vertex for vertex in w._vertices]

                if w._vel_ends[0] is not None:
                    past_velocity_idx_map[w_idx] = len(batched_velocities)
                    batched_velocities.append(w._vel_ends[0])

                if w._vel_ends[1] is not None:
                    future_velocity_idx_map[w_idx] = len(batched_velocities)
                    batched_velocities.append(w._vel_ends[1])

                proper_time_origin_events.append(w.eval(w.proper_time_origin))

            batched_events = np.concatenate([vertices, proper_time_origin_events])

            new_batched_events = boost(
                batched_events - event_delta,
                velocity_delta)

            new_batched_velocities = boost_velocity_s(
                batched_velocities,
                velocity_delta)

            new_vertices = new_batched_events[:len(vertices)]
            new_proper_time_origin_events = new_batched_events[len(vertices):]

            cur_vertices_idx = 0

            for w_idx, w in enumerate(self._worldlines):
                if w_idx in past_velocity_idx_map:
                    past_velocity = new_batched_velocities[past_velocity_idx_map[w_idx]]
                else:
                    past_velocity = None

                if w_idx in future_velocity_idx_map:
                    future_velocity = new_batched_velocities[future_velocity_idx_map[w_idx]]
                else:
                    future_velocity = None

                num_vertices = vertex_count[w_idx]

                # TODO: Make Worldline accept 1-element ndarray to avoid this cast
                new_proper_time_origin = float(new_proper_time_origin_events[w_idx][0])
                new_w = Worldline(
                        new_vertices[cur_vertices_idx : cur_vertices_idx + num_vertices],
                        vel_past=past_velocity,
                        vel_future=future_velocity,
                        proper_time_origin=new_proper_time_origin,
                        proper_time_offset=w.proper_time_offset)

                cur_vertices_idx += num_vertices

                new_worldlines.append(new_w)

        else:
            for w_idx, w in enumerate(self._worldlines):
                new_w = (w - event_delta).boost(velocity_delta)

                new_worldlines.append(new_w)

        return Frame(new_worldlines)
