import numpy as np

# An inertial reference frame in Minkowski spacetime with two spatial
# dimensions.
class Frame2D:
    def __init__(self):
        self._clocks = []

    def add_clock(self, clock):
        self._clocks.append(clock)

    def get_state_at_time(self, time):
        state = []

        # TODO: Keeping the properties of the clocks together in `ndarray`s
        # would make transformations much more performant because we'd avoid
        # looping through them
        for clock in self._clocks:
            face_time, event = clock.get_state_at_time(time)
            state.append((clock, face_time, event))

        return state


    # Transform the frame, applying a time and position translations first,
    # then applying a velocity transformation
    def transform(self, event_delta, velocity):
        speed = np.norm(velocity)


# A clock that moves at a constant velocity, and exists over the entire
# time axis of a reference frame.
class Clock:
    def __init__(self, face_time0, event0, velocity):
        # Check `face_time0` arg
        assert isinstance(face_time0, (np.ndarray, float, int))
        face_time0 = np.ndarray(face_time0)
        assert face_time0.shape == ()

        # Check `event0` arg
        assert isinstance(event0, np.ndarray)
        assert event0.shape == (3,)
        # TODO: Get rid of this requirement
        assert event0[0] == 0

        # Check `velocity` arg
        assert isinstance(velocity, np.ndarray)
        assert velocity.shape == (2,)
        # Limit the speed of a clock to the speed of light
        speed = np.linalg.norm(velocity)
        assert speed <= 1

        self._event0 = event0
        self._face_time0 = face_time0
        self._velocity = velocity

        # The change in the clock's proper time with respect to time
        # in the rest frame. Could be called a "time dilation factor"
        self._dtau_dt = np.sqrt(1 - speed**2)

    # Gives the event and face time of the clock at a particular time
    def get_state_at_time(time):
        event = np.concatenate((
            time,
            # TODO: This line would need to change if `event0[0] != 0`
            self._event0[1:] + self._velocity * time))

        face_time = self.face_time0 + time * self._dtau_dt

        return face_time, event






