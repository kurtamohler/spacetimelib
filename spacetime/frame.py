import numpy as np

from .transformations import boost


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

        # TODO: Storing the properties of the clocks together in `ndarray`s
        # would make transformations much more performant because we'd replace
        # this loop with just a couple numpy calls
        for clock in self._clocks:
            face_time, event = clock.get_state_at_time(time)
            state.append((face_time, event))

        return state


    # Transform the frame, applying a time and position translations first,
    # then applying a velocity transformation
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

        # TODO: While this impl of batching is roughly 4x faster on my machine
        # than the non-batched path while uniformly accelerating in
        # `examples/clock_grid.py`, it is still pretty inefficient to copy all
        # the events and velocities out of the clocks and then back into the
        # new clocks. It would be much more efficient to always keep a batched
        # representation of the frame. The events and velocities of the
        # individual clocks could potentially be views into the batched
        # representation, if it makes sense to do so. Otherwise, they could
        # just be copies. But I would like to have some way to automatically
        # update between them if one of the copies or views changes. Probably
        # automating that would be easier if the coordinates are copied rather
        # than shared. But deciding for sure will require more thought.
        if batched:
            event0_batch = np.array([clock._event0 for clock in self._clocks])
            velocity_batch = np.array([clock._velocity for clock in self._clocks])

            event0_batch_out, velocity_batch_out = boost(
                velocity_delta,
                event0_batch - event_delta,
                velocity_batch)

            for clock_idx, clock in enumerate(self._clocks):
                new_clocks.append(Clock(
                    clock._face_time0,
                    event0_batch_out[clock_idx],
                    velocity_batch_out[clock_idx]))

        else:
            for clock in self._clocks:
                event0 = clock._event0
                velocity = clock._velocity

                event0_, velocity_ = boost(
                    velocity_delta,
                    event0 - event_delta,
                    velocity)

                new_clocks.append(Clock(
                    clock._face_time0,
                    event0_,
                    velocity_))

        return Frame2D(new_clocks)


# A clock that moves at a constant velocity, and exists over the entire
# time axis of a reference frame.
class Clock:
    def __init__(self, face_time0, event0, velocity):
        # Check `face_time0` arg
        face_time0 = np.array(face_time0)
        assert face_time0.shape == ()

        # Check `event0` arg
        event0 = np.array(event0)
        assert event0.shape == (3,)

        # Check `velocity` arg
        if velocity is None:
            velocity = np.array([0, 0])
        else:
            velocity = np.array(velocity)
            assert velocity.shape == (2,)

        # Limit the speed of a clock to the speed of light
        # TODO: This is probably not a very good idea
        speed = np.linalg.norm(velocity)
        if speed > 1:
            velocity = velocity / speed
            speed = 1

        self._event0 = event0
        self._face_time0 = face_time0
        self._velocity = velocity

        # The change in the clock's proper time with respect to time
        # in the rest frame. Could be called a "time dilation factor"
        self._dtau_dt = np.sqrt(1 - speed**2)

    # Gives the event and face time of the clock at a particular time
    def get_state_at_time(self, time):
        position0 = self._event0[1:]
        time0 = self._event0[0]
        event = np.concatenate((
            [time],
            position0 + self._velocity * (time - time0)))

        face_time = self._face_time0 + (time - time0) * self._dtau_dt

        return face_time, event



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


