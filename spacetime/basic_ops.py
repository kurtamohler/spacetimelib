from .error_checking import check

import numpy as np

def boost(boost_velocity, event, velocity=None, light_speed=1):
    '''
    Boost the coordinates of one or more events by a specified velocity.
    Optionally, velocities can also be boosted.

    This operation uses the Lorentz vector transformations as described here:
    https://en.wikipedia.org/wiki/Lorentz_transformation#Vector_transformations

    Batched inputs are supported, to allow boosting multiple events and velocities
    by multiple boost velocities in one call.

    A single event is given by a 1-D array of N+1 elements, where N is the
    number of spatial dimensions. The first element is the time coordinate, and
    the remaining elements are spatial coordinates. For example, if the event
    is `(t, x, y, z)`, `t` is the time coordinate and `x`, `y`, and `z` are
    spatial coordinates in a three dimensional space.

    A single velocity is given by a 1-D array of N elements, derivatives of the
    spatial dimensions with respect to coordinate time. For example, the
    velocity `(v_x, v_y, v_z)` can also be written `(dx/dt, dy/dt, dz/dt)`.

    Args:

      boost_velocity : array_like
          Boost velocity to use for the transformation.
          Shape: (..., N)

      event : array_like or None
          Coordinates of events (or any 4-vector) to be boosted.
          Shape: (..., N+1)

      velocity : array_like, optional
          Velocities to be boosted.
          Shape: (..., N)
          Default: None

      light_speed : array_like, optional scalar speed of light. Default: 1

    Returns:

      If `velocity is None`:
          `event_boosted` : ndarray

      If `velocity is not None`:
          (`event_boosted`, `velocity_boosted`) : tuple of ndarray

    '''
    check(event is not None or velocity is not None, ValueError,
        "expected either `event` or `velocity` to be given, but both are `None`")
    boost_velocity = np.array(boost_velocity)
    light_speed = np.array(light_speed)

    if boost_velocity.ndim == 0:
        boost_velocity = np.array([boost_velocity])

    if event is not None:
        event = np.array(event)
        check(event.ndim > 0, ValueError,
            "expected 'event' to have one or more dimensions, ",
            f"but got {event.ndim}")

        # TODO: Need to think more about the logic here. It might be a bit wrong
        if event.shape[-1] == 2 and boost_velocity.shape[-1] > 1:
            boost_velocity = np.expand_dims(boost_velocity, -1)
        else:
            check(event.shape[-1] - 1 == boost_velocity.shape[-1], ValueError,
                "expected 'event.shape[-1] - 1 == boost_velocity.shape[-1]', but ",
                f"got '{event.shape[-1]} - 1 != {boost_velocity.shape[-1]}'")

    frame_speed = np.linalg.norm(boost_velocity, axis=-1)

    # TODO: If boost_velocity is batched, we should only print out the
    # first speed in frame_speed that is greater than light_speed
    check((frame_speed < light_speed).all(), ValueError,
        "the norm of 'boost_velocity' must be less than ",
        f"'light_speed' ({light_speed}), but got {frame_speed}")

    # TODO: Would batching the speed of light be useful at all? Probably best to
    # wait and see before adding batching.
    check(light_speed.ndim == 0, ValueError,
        "expected 'light_speed' to have 0 dimensions, ",
        f"but got {light_speed.ndim}")
    check(light_speed > 0, ValueError,
        f"expected 'light_speed' to be positive, but got {light_speed}")

    dtype = np.find_common_type([boost_velocity.dtype, light_speed.dtype], [])

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
    boost_velocity = boost_velocity.astype(dtype)
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
            f"'boost_velocity' must be nonzero, but got {boost_velocity}")

    # γ = 1 / √(1 - v ⋅ v / c²)
    lorentz_factor = 1 / np.sqrt(1 - np.square(frame_speed / light_speed))
    
    if event is not None:
        position = event[..., 1:]
        time = event[..., 0]

        # r' = r + v ((r ⋅ v) (γ - 1) / v² - tγ)
        position_ = position + boost_velocity * np.expand_dims(
            np.sum(position * boost_velocity, axis=-1) * (lorentz_factor - 1)
                / np.square(frame_speed)    # TODO: fix division by zero case
            - time * lorentz_factor,
            axis=-1)

        # t' = γ (t - (r ⋅ v) / c²)
        time_ = lorentz_factor * (time - np.sum(position * boost_velocity, axis=-1)
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
            np.sum(velocity * boost_velocity, axis=-1),
            axis=-1)

        outer_factor = 1 / (1 - (u_dot_v / (light_speed**2)))
        inner_factor = np.expand_dims(
            (lorentz_factor / (lorentz_factor + 1)) / (light_speed**2),
            axis=-1)

        # TODO: Probably should expand this above, where it's first calculated
        L = np.expand_dims(lorentz_factor, axis=-1)

        velocity_ = outer_factor * (velocity / L - boost_velocity + inner_factor * u_dot_v * boost_velocity)

        # TODO: Need to broadcast `velocity_` and `event_` together here

    else:
        velocity_ = None

    if velocity is None:
        return event_
    else:
        return event_, velocity_


# TODO: Probably get rid of this, in favor of just taking the difference between the
# two events and calling `norm_st2` on the difference.
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

    return np.sqrt(-norm_st2(event1 - event0))

def norm_s(vec_s):
    '''
    Calculate the norm of a space-vector. This is simply the Euclidean norm, or
    more formally, the [L-2 norm](https://mathworld.wolfram.com/L2-Norm.html).

    Given an N-dimensional space-vector `a = (a1, ..., aN)`, the norm is
    `norm_s(a) = sqrt(a1^2 + ... + aN^2)`.

    For instance, if the space-vector is the difference between the coordinates
    of two positions in space, then its norm is the distance between the events.

    Args:
        vec_s : array
            Any space-vector
            Shape: (..., N)
    '''
    return np.linalg.norm(vec_s, axis=-1)

def norm_st2(vec_st):
    '''
    Calculate the square of the norm of a spacetime-vector.

    Given an N+1-dimensional spacetime-vector `a = (a0, a1, ..., aN)`, the
    square norm is `norm_st2(a) = - a0^2 + a1^2 + ... + aN^2`.

    This is traditionally called the [squared norm of
    a four-vector](https://mathworld.wolfram.com/Four-VectorNorm.html)

    For instance, if the spacetime-vector is the difference between the
    coordinates of two events, then its squared norm is the square of proper
    distance (or the space-like interval) between the events. The negative of
    the squared norm is the proper time (or the time-like interval) between the
    events.

    Args:
        vec_st : array
            Any spacetime-vector
            Shape: (..., N+1)
    '''
    vec_st = np.array(vec_st)
    return -vec_st[..., 0]**2 + np.linalg.norm(vec_st[..., 1:], axis=-1)**2

def velocity_st(vel_s, light_speed=1):
    '''
    Calculates the spacetime-velocity vector from a space-velocity vector.

    Given a space-velocity `v = (v1, ..., vN)`, the spacetime-velocity is
    calculated by `(1 , v1, ..., vN) / sqrt(1 - |v|**2)`.

    Spacetime-velocity is traditionally called four-velocity in the context of
    3+1 Minkowski spacetime.

    This is the reverse of [`spacetime.velocity_s`](spacetime.velocity_s).

    Args:

      vel_s : array_like
          Space-velocity of a particle, given by the derivative of each space
          dimension with respect to coordinate time. If the norm of the
          space-velocity is equal to or greater than the speed of light, then
          the spacetime-velocity is undefined and an error will raise.
          Shape: (..., N)

      light_speed : array_like, optional scalar Speed of light. Default: 1
    '''

    if light_speed != 1:
        raise NotImplementedError('light_speed must be 1')

    vel_s = np.array(vel_s)
    if vel_s.ndim == 0:
        vel_s = np.array([vel_s])

    speed = np.linalg.norm(vel_s, axis=-1)
    check((speed < light_speed).all(), ValueError,
        "the norm of 'vel_s' must be less than ",
        f"'light_speed' ({light_speed}), but got {speed}")

    shape = list(vel_s.shape)
    shape[-1] += 1
    vel_st = np.empty(shape, dtype=vel_s.dtype)

    vel_st[..., 0] = 1
    vel_st[..., 1:] = vel_s
    vel_st = vel_st / np.sqrt(1 - np.expand_dims(speed, -1)**2)

    # TODO: Find out if this should be light_speed**2 or light_speed**-2 or whatever.
    if not np.allclose(-norm_st2(vel_st), light_speed, atol=0.01):
        raise ValueError(
            'Due to floating point error, one of the given space-velocities '
            'gave a spacetime-velocity whose proper time is not 1. I hope this '
            'error never happens.')

    return vel_st

def velocity_s(vel_st):
    '''
    Calculates the space-velocity vector from a spacetime-velocity vector.

    This is the reverse of [`spacetime.velocity_st`](spacetime.velocity_st).

    Args:

      vel_st : array_like
          Spacetime-velocity of a particle, given by the derivative of each
          dimension of spacetime (coordinate time dimension comes first) with
          respect to proper time.
          Shape: (..., N+1)
    '''
    return vel_st[..., 1:] / np.expand_dims(vel_st[..., 0], -1)
