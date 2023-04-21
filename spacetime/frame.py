import numpy as np

from .basic_ops import boost, boost_velocity_s
from .worldline import Worldline
from .error_checking import check


class Frame2D:
    '''
    An inertial reference frame in Minkowski spacetime with two spatial
    dimensions.
    '''

    def __init__(self, clocks=None):
        if clocks is None:
            self._clocks = []
        else:
            assert isinstance(clocks, (list, tuple))
            for clock in clocks:
                assert isinstance(clock, Clock)
            self._clocks = clocks

    def append(self, clock):
        self._clocks.append(clock)

    def get_state_at_time(self, time):
        '''
        Returns the face times and events for all clocks at the specified time.
        '''
        state = []

        # TODO: I'd like to parallelize this into one batched operation, but
        # I don't think that's very possible since worldlines can have all
        # different numbers of vertices. Still, there could be something that
        # would give better performance here.
        for clock in self._clocks:
            face_time, event = clock.get_state_at_time(time)
            state.append((face_time, event))

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

        new_clocks = []

        batched = True

        # TODO: While this impl of batching is roughly 3x faster on my machine
        # than the non-batched path while uniformly accelerating in
        # `examples/clock_grid.py`, it is still pretty inefficient to copy all
        # the events and velocities out of the clocks and then back into the
        # new clocks. It would be much more efficient to always keep a batched
        # representation of the frame cached. This would be very simple to
        # implement, just separate the batched representation creation into
        # a new function and use the `lru_cache` decorator (or whatever it's
        # called) to auto-cache it.  Another possibility is that events and
        # velocities of the individual clocks could potentially be views into
        # the batched representation, but that seems harder to implement and
        # prevent broken views.
        if batched:
            vertex_count = []
            vertices = []
            proper_time_origin_events = []

            batched_velocities = []

            # Map the clock index to an index into `velocities`, for both the
            # past and future velocity
            past_velocity_idx_map = {}
            future_velocity_idx_map = {}

            # Need to grab all event vertices and velocities from all worldlines and combine
            # them into two batched event and velocity arrays that can be boosted. Need
            # to keep track of which vertices and velocities belong to which clocks.
            for clock_idx, clock in enumerate(self._clocks):
                worldline = clock._worldline
                vertex_count.append(len(worldline._vertices))
                vertices += [vertex for vertex in worldline._vertices]

                if worldline._vel_ends[0] is not None:
                    past_velocity_idx_map[clock_idx] = len(batched_velocities)
                    batched_velocities.append(worldline._vel_ends[0])

                if worldline._vel_ends[1] is not None:
                    future_velocity_idx_map[clock_idx] = len(batched_velocities)
                    batched_velocities.append(worldline._vel_ends[1])

                proper_time_origin_events.append(worldline.eval(worldline.proper_time_origin))

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

            for clock_idx, clock in enumerate(self._clocks):
                if clock_idx in past_velocity_idx_map:
                    past_velocity = new_batched_velocities[past_velocity_idx_map[clock_idx]]
                else:
                    past_velocity = None

                if clock_idx in future_velocity_idx_map:
                    future_velocity = new_batched_velocities[future_velocity_idx_map[clock_idx]]
                else:
                    future_velocity = None

                num_vertices = vertex_count[clock_idx]

                # TODO: Make Worldline accept 1-element ndarray to avoid this cast
                new_proper_time_origin = float(new_proper_time_origin_events[clock_idx][0])
                new_proper_time_offset = float(clock._clock_time0)

                # TODO: Why does this fail?
                #if clock._clock_time0 != worldline.proper_time_offset:
                #    print(f'{clock_idx}: {clock._clock_time0}, {worldline.proper_time_offset}')

                new_worldline = Worldline(
                        new_vertices[cur_vertices_idx : cur_vertices_idx + num_vertices],
                        vel_past=past_velocity,
                        vel_future=future_velocity,
                        proper_time_origin=new_proper_time_origin,
                        proper_time_offset=new_proper_time_offset)
                        #proper_time_offset=worldline.proper_time_offset)

                cur_vertices_idx += num_vertices

                new_clocks.append(Clock(
                    new_worldline,
                    new_proper_time_offset))

        else:
            for clock_idx, clock in enumerate(self._clocks):
                worldline = clock._worldline
                new_worldline = (worldline - event_delta).boost(velocity_delta)

                # TODO: Why does this fail?
                #if clock._clock_time0 != new_worldline.proper_time_offset:
                #    print(f'{clock_idx}: {clock._clock_time0}, {new_worldline.proper_time_offset}')

                new_clocks.append(Clock(
                    new_worldline,
                    clock._clock_time0))
                    #new_worldline.proper_time_offset))

        return Frame2D(new_clocks)


class Clock:
    '''
    A clock that moves at a constant velocity, and exists over the entire
    time axis of a reference frame.
    '''

    def __init__(self, worldline, clock_time0):
        check(isinstance(worldline, Worldline), TypeError,
            "expected `worldline` to be a `Worldline` type, but got ",
            f"{type(worldline)} instead")

        # Check `clock_time0` arg
        clock_time0 = np.array(clock_time0)
        check(clock_time0.ndim == 0, ValueError,
            "expected `clock_time0` to be a scalar, but got array of size ",
            f"{clock_time0.shape}")

        self._worldline = worldline
        self._clock_time0 = clock_time0

    def get_state_at_time(self, time):
        '''
        Gives the event and face time of the clock at a particular time.
        '''
        event = self._worldline.eval(time)

        tau = self._worldline.proper_time(time)

        clock_time = self._clock_time0 + tau

        #if self._clock_time0 != self._worldline.proper_time_offset:
        #    print(f'{self._clock_time0}, {self._worldline.proper_time_offset}')

        return clock_time, event
