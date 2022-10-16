from .error_checking import check

import numpy as np

def boost(frame_velocity, event, velocity=None, light_speed=1):
    '''
    Transforms an event from one inertial frame, F, to a another inertial frame,
    F', using the Lorentz vector transformations:
    https://en.wikipedia.org/wiki/Lorentz_transformation#Vector_transformations

    This function works for any number of spatial dimensions, N.

    NOTE:
      The arguments of this function can be batched, in order to perform
      multiple transformations with one call. This gives better performance than
      calling the function many individual times. The `...` in the shape for each
      argument below means that those parts of their shapes must all be
      broadcastable together. NumPy broadcasting semantics are used:
      https://numpy.org/doc/stable/user/basics.broadcasting.html

      TODO: The actual batching behavior is wonky. Need to think carefully
      about what behavior would be most useful and understandable.

    Args:

      frame_velocity : array_like
          Velocity of frame F' relative to F.
          Shape: (..., N), or () if `N == 1`

      event : array_like or None
          N+1 dimensional event for the particle in frame F. Given as
          [t, x_1, x_2, ..., x_N], where t is the time coordinate and x_i is the
          spatial coordinate for dimension i. If `None`, then `velocity` must not
          be `None`.
          Shape: (..., N+1)

      velocity : array_like, optional
          Velocity of the particle for each spatial dimension with respect to
          time in frame F. If given, the output `velocity_` will be the
          Lorentz transformed velocity of the particle in frame F'.
          Shape: (..., N)
          Default: None

      light_speed : array_like, optional scalar Speed of light. Default: 1

    Returns:

      `event_`, `velocity_` : tuple of ndarray
    '''
    check(event is not None or velocity is not None, ValueError,
        "expected either `event` or `velocity` to be given, but both are `None`")
    frame_velocity = np.array(frame_velocity)
    light_speed = np.array(light_speed)

    if frame_velocity.ndim == 0:
        frame_velocity = np.array([frame_velocity])

    if event is not None:
        event = np.array(event)
        check(event.ndim > 0, ValueError,
            "expected 'event' to have one or more dimensions, ",
            f"but got {event.ndim}")

        # TODO: Need to think more about the logic here. It might be a bit wrong
        if event.shape[-1] == 2 and frame_velocity.shape[-1] > 1:
            frame_velocity = np.expand_dims(frame_velocity, -1)
        else:
            check(event.shape[-1] - 1 == frame_velocity.shape[-1], ValueError,
                "expected 'event.shape[-1] - 1 == frame_velocity.shape[-1]', but ",
                f"got '{event.shape[-1]} - 1 != {frame_velocity.shape[-1]}'")

    frame_speed = np.linalg.norm(frame_velocity, axis=-1)

    # TODO: If frame_velocity is batched, we should only print out the
    # first speed in frame_speed that is greater than light_speed
    check((frame_speed < light_speed).all(), ValueError,
        "the norm of 'frame_velocity' must be less than ",
        f"'light_speed' ({light_speed}), but got {frame_speed}")

    # TODO: Would batching the speed of light be useful at all? Probably best to
    # wait and see before adding batching.
    check(light_speed.ndim == 0, ValueError,
        "expected 'light_speed' to have 0 dimensions, ",
        f"but got {light_speed.ndim}")
    check(light_speed > 0, ValueError,
        f"expected 'light_speed' to be positive, but got {light_speed}")

    dtype = np.find_common_type([frame_velocity.dtype, light_speed.dtype], [])

    if event is not None:
        dtype = np.find_common_type([event.dtype, dtype], [])

    if velocity is not None:
        velocity = np.array(velocity)
        if velocity.ndim == 0:
            velocity = np.array([velocity])

        # TODO: Need to think more about the logic here. It might be a bit wrong
        if event is not None:
            if event.shape[-1] == 2 and velocity.shape[-1] > 1:
                velocity = np.expand_dims(velocity, -1)
            else:
                check(event.shape[-1] - 1 == velocity.shape[-1], ValueError,
                    "expected 'event.shape[-1] - 1 == velocity.shape[-1]', but ",
                    "got '{event.shape[-1]} - 1 != {velocity.shape[-1]'")

        speed = np.linalg.norm(velocity, axis=-1)
        check((speed <= light_speed).all(), ValueError,
            "the norm of 'velocity' must be less than or equal to ",
            f"'light_speed' ({light_speed}), but got {speed}")

        dtype = np.find_common_type([dtype, velocity.dtype], [])

    # Change dtypes to match each other
    frame_velocity = frame_velocity.astype(dtype)
    light_speed = light_speed.astype(dtype)
    if event is not None:
        event = event.astype(dtype)
    if velocity is not None:
        velocity = velocity.astype(dtype)

    # TODO: Need to check up front whether the args can broadcast with each other.

    if frame_speed.ndim == 0:
        if frame_speed == 0:
            if velocity is None:
                return event
            else:
                return event, velocity
    else:
        # TODO: This case should be supported, but will require a condition
        # below to prevent the division by zero
        check((frame_speed > 0).all(), ValueError,
            f"'frame_velocity' must be nonzero, but got {frame_velocity}")

    # γ = 1 / √(1 - v ⋅ v / c²)
    lorentz_factor = 1 / np.sqrt(1 - np.square(frame_speed / light_speed))
    
    if event is not None:
        position = event[..., 1:]
        time = event[..., 0]

        # r' = r + v ((r ⋅ v) (γ - 1) / v² - tγ)
        position_ = position + frame_velocity * np.expand_dims(
            np.sum(position * frame_velocity, axis=-1) * (lorentz_factor - 1)
                / np.square(frame_speed)    # TODO: fix division by zero case
            - time * lorentz_factor,
            axis=-1)

        # t' = γ (t - (r ⋅ v) / c²)
        time_ = lorentz_factor * (time - np.sum(position * frame_velocity, axis=-1)
            / np.square(light_speed))

        event_ = np.empty(time_.shape + (position_.shape[-1] + 1,), dtype=dtype)

        event_[..., 0] = time_
        event_[..., 1:] = position_

    else:
        event_ = None

    if velocity is not None:
        # u' = (u / γ - v + (γ (u ⋅ v) v) / (c² (γ + 1)))
        #      / (1 - u ⋅ v / c²)

        u_dot_v = np.expand_dims(
            np.sum(velocity * frame_velocity, axis=-1),
            axis=-1)

        outer_factor = 1 / (1 - (u_dot_v / (light_speed**2)))
        inner_factor = np.expand_dims(
            (lorentz_factor / (lorentz_factor + 1)) / (light_speed**2),
            axis=-1)

        # TODO: Probably should expand this above, where it's first calculated
        L = np.expand_dims(lorentz_factor, axis=-1)

        velocity_ = outer_factor * (velocity / L - frame_velocity + inner_factor * u_dot_v * frame_velocity)

        # TODO: Need to broadcast `velocity_` and `event_` together here

    else:
        velocity_ = None

    if velocity is None:
        return event_
    else:
        return event_, velocity_


# TODO: Need to get rid of this, in favor of just taking the difference between the
# two events and calling `proper_time_squared` on the difference.
def _proper_time(event0, event1):
    '''
    Calculate the proper time between two events.

    Args:
        event0 : array
            First event

        event1 : array
            Second event

    Returns: number
    '''
    event0 = np.array(event0)
    event1 = np.array(event1)

    check(event0.shape == event1.shape, ValueError,
        "expected both events to have same shape")
    check(event0.ndim == 1, ValueError, "expected exactly two dimensions")
    check(event0.shape[0] >= 2, ValueError, "expected at least 2 dims")

    event_diff_sq = (event1 - event0) ** 2

    return np.sqrt(event_diff_sq[0] - np.sum(event_diff_sq[1:]))

def proper_time_squared(four_vector):
    '''
    Calculate the proper time squared between the origin and some four_vector.

    Args:
        four_vector : array
            A four-vector.
    '''
    four_vector = np.array(four_vector)

    return four_vector[..., 0]**2 - np.sum(four_vector[..., 1:]**2, axis=-1)


def four_velocity(three_velocity, light_speed=1):
    '''
    Creates a four-velocity vector from a three-velocity vector. However, this
    function is generalized to any N+1 Minkowski spacetime, so it can also
    create an (N+1)-velocity vector from an N-velocity vector.

    Args:

      three_velocity : array_like
          Three-velocity of a particle for each spacetime dimension with respect to
          proper time.
          Shape: (..., N)

      light_speed : array_like, optional scalar Speed of light. Default: 1
    '''

    if light_speed != 1:
        raise NotImplementedError('light_speed must be 1')

    three_velocity = np.array(three_velocity)
    if three_velocity.ndim == 0:
        velocity = np.array([three_velocity])

    speed = np.linalg.norm(three_velocity, axis=-1)
    check((speed <= light_speed).all(), ValueError,
        "the norm of 'three_velocity' must be less than or equal to ",
        f"'light_speed' ({light_speed}), but got {speed}")

    shape = list(three_velocity.shape)
    shape[-1] += 1
    four_velocity = np.empty(shape, dtype=three_velocity.dtype)

    four_velocity[..., 0] = 1 / np.sqrt(1 - speed**2)
    four_velocity[..., 1:] = three_velocity / np.sqrt(1 - np.expand_dims(speed, -1)**2)

    # TODO: Find out if this should be light_speed**2 or light_speed**-2 or whatever.
    if not np.allclose(proper_time_squared(four_velocity), light_speed, atol=0.01):
        raise ValueError(
            'Due to floating point error, one of the given three-velocities '
            'gave a four-velocity whose proper time is not 1. I hope this '
            'error never happens.')

    return four_velocity

def three_velocity(four_velocity):
    '''
    Creates a three-velocity vector from a four-velocity vector. However, this
    function is generalized to any N+1 Minkowski spacetime, so it can also
    create an N-velocity vector from an (N+1)-velocity vector.

    Args:

      four_velocity : array_like
          Four-velocity of a particle for each spacetime dimension with respect to
          proper time.
          Shape: (..., N+1)
    '''
    return four_velocity[..., 1:] / np.expand_dims(four_velocity[..., 0], -1)
