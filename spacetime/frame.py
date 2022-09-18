import numpy as np

from .basic_ops import boost
from .worldline import Worldline
from .error_checking import check


# An inertial reference frame in Minkowski spacetime with two spatial
# dimensions.
class Frame2D:
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

    # Returns the face times and events for all clocks at the specified time
    def get_state_at_time(self, time):
        state = []

        # TODO: I'd like to parallelize this into one batched operation, but
        # I don't think that's very possible since worldlines can have all
        # different numbers of vertices. Still, there could be something that
        # would give better performance here.
        for clock in self._clocks:
            face_time, event = clock.get_state_at_time(time)
            state.append((face_time, event))

        return state


    # Transform the frame, applying a time and position translations first,
    # then applying a velocity transformation
    # TODO: `event_delta` should be None by default. Actually, probably shouldn't
    # even be here--instead, add an addition function and use that? But that would
    # give worse performance though...
    def boost(self, event_delta, velocity_delta):
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
            clock_time0_events = []

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

                clock_time0_events.append(worldline.eval(clock._time0))

            batched_events = np.concatenate([vertices, clock_time0_events])

            new_batched_events, new_batched_velocities = boost(
                velocity_delta,
                batched_events - event_delta,
                batched_velocities)

            new_vertices = new_batched_events[:len(vertices)]
            new_clock_time0_events = new_batched_events[len(vertices):]

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
                
                new_worldline = Worldline(
                        new_vertices[cur_vertices_idx : cur_vertices_idx + num_vertices],
                        vel0=past_velocity,
                        velN=future_velocity)

                cur_vertices_idx += num_vertices

                new_clocks.append(Clock(
                    new_worldline,
                    new_clock_time0_events[clock_idx][0],
                    clock._clock_time0))

        else:
            for clock in self._clocks:
                worldline = clock._worldline
                new_worldline = (worldline - event_delta).boost(velocity_delta)
                #new_time0 = (worldline.eval(clock._time0) - event_delta).boost(velocity_delta)[0]

                new_time0 = boost(
                    velocity_delta,
                    worldline.eval(clock._time0) - event_delta)[0]

                clock_time0 = clock._clock_time0
                new_clocks.append(Clock(
                    new_worldline,
                    new_time0,
                    clock_time0))

        return Frame2D(new_clocks)


# A clock that moves at a constant velocity, and exists over the entire
# time axis of a reference frame.
class Clock:
    #def __init__(self, face_time0, event0, velocity):

    def __init__(self, worldline, time0, clock_time0):
        check(isinstance(worldline, Worldline), TypeError,
            "expected `worldline` to be a `Worldline` type, but got ",
            f"{type(worldline)} instead")

        # Check `time0` arg
        time0 = np.array(time0)
        check(time0.ndim == 0, ValueError,
            "expected `_time0` to be a scalar, but got array of size ",
            f"{time0.shape}")

        # Check `clock_time0` arg
        clock_time0 = np.array(clock_time0)
        check(clock_time0.ndim == 0, ValueError,
            "expected `clock_time0` to be a scalar, but got array of size ",
            f"{clock_time0.shape}")

        self._worldline = worldline
        self._time0 = time0
        self._clock_time0 = clock_time0

    # Gives the event and face time of the clock at a particular time
    def get_state_at_time(self, time):
        #position0 = self._event0[1:]
        #time0 = self._event0[0]
        #event = np.concatenate((
        #    [time],
        #    position0 + self._velocity * (time - time0)))

        #face_time = self._clock_time0 + (time - time0) * self._dtau_dt

        #return face_time, event

        event = self._worldline.eval(time)

        tau = self._worldline.proper_time(
            self._time0,
            time)

        # TODO: Should probably have an option in `proper_time` to preserve sign,
        # to avoid this conditional?
        if time > self._time0:
            clock_time = self._clock_time0 + tau
        else:
            clock_time = self._clock_time0 - tau

        return clock_time, event




if __name__ == '__main__':
    frame = Frame2D()
    frame.append(Clock(
        0,
        (0, -10, 0),
        (0, 0)))
    frame.append(Clock(
        0,
        (0, -5, 0),
        (0, 0)))
    frame.append(Clock(
        0,
        (0, 0, 0),
        (0, 0)))
    frame.append(Clock(
        0,
        (0, 5, 0),
        (0, 0)))
    frame.append(Clock(
        0,
        (0, 10, 0),
        (0, 0)))

    print(frame.get_state_at_time(0))

    frame_ = frame.boost(
        (0, 0, 0),
        (0, 0))
    print(frame_.get_state_at_time(0))



