import numpy as np

from .error_checking import check

def boost(vec_st, boost_vel_s, light_speed=1):
    '''
    Boost a spacetime-vector by a specified space-velocity.

    This operation uses the Lorentz vector transformations as described here:
    `Lorentz Transformation: Vector transformations - Wikipedia
    <https://en.wikipedia.org/wiki/Lorentz_transformation#Vector_transformations>`_

    Batched inputs are supported, to allow boosting multiple spacetime-vectors
    or space-velocities by multiple boost space-velocities in one call.

    A single spacetime-vector is given by a 1-D array of N+1 elements, where
    N is the number of spatial dimensions. The first element is the time
    coordinate, and the remaining elements are spatial coordinates. For
    example, if the spacetime-vector represents an event ``(t, x, y, z)`` in
    3+1 spacetime, ``t`` is the time coordinate and ``x``, ``y``, and ``z`` are
    spatial coordinates in a three dimensional space.

    A single space-velocity is given by a 1-D array of N elements, derivatives
    of the spatial dimensions with respect to coordinate time. For example, the
    space-velocity ``(v_x, v_y, v_z)`` can also be written ``(dx/dt, dy/dt,
    dz/dt)``.

    Args:

      vec_st (array_like):
        Spacetime-vectors to be boosted.

        Shape: (..., N+1)

      boost_vel_s (array_like):
        Space-velocity to use as the boost velocity of the Lorentz
        transformation.

        Shape: (..., N)

      light_speed (array_like, optional):
        Scalar speed of light.

        Default: 1

    Returns:
      ndarray: The boosted spacetime-vector
    '''
    boost_vel_s = np.array(boost_vel_s)
    light_speed = np.array(light_speed)

    if boost_vel_s.ndim == 0:
        boost_vel_s = np.array([boost_vel_s])

    vec_st = np.array(vec_st)
    check(vec_st.ndim > 0, ValueError,
        "expected 'vec_st' to have one or more dimensions, ",
        f"but got {vec_st.ndim}")

    # TODO: Need to think more about the logic here. It might be a bit wrong
    if vec_st.shape[-1] == 2 and boost_vel_s.shape[-1] > 1:
        boost_vel_s = np.expand_dims(boost_vel_s, -1)
    else:
        check(vec_st.shape[-1] - 1 == boost_vel_s.shape[-1], ValueError,
            "expected 'vec_st.shape[-1] - 1 == boost_vel_s.shape[-1]', but ",
            f"got '{vec_st.shape[-1]} - 1 != {boost_vel_s.shape[-1]}'")

    frame_speed = np.linalg.norm(boost_vel_s, axis=-1)

    # TODO: If boost_vel_s is batched, we should only print out the
    # first speed in frame_speed that is greater than light_speed
    check((frame_speed < light_speed).all(), ValueError,
        "the norm of 'boost_vel_s' must be less than ",
        f"'light_speed' ({light_speed}), but got {frame_speed}")

    # TODO: Would batching the speed of light be useful at all? Probably best to
    # wait and see before adding batching.
    check(light_speed.ndim == 0, ValueError,
        "expected 'light_speed' to have 0 dimensions, ",
        f"but got {light_speed.ndim}")
    check(light_speed > 0, ValueError,
        f"expected 'light_speed' to be positive, but got {light_speed}")

    dtype = np.find_common_type([boost_vel_s.dtype, light_speed.dtype], [])

    dtype = np.find_common_type([vec_st.dtype, dtype], [])

    # Change dtypes to match each other
    boost_vel_s = boost_vel_s.astype(dtype)
    light_speed = light_speed.astype(dtype)
    vec_st = vec_st.astype(dtype)

    # TODO: Need to check up front whether the args can broadcast with each other.

    if frame_speed.ndim == 0:
        if frame_speed == 0:
            return vec_st
    else:
        # TODO: This case should be supported, but will require a condition
        # below to prevent the division by zero
        check((frame_speed > 0).all(), ValueError,
            f"'boost_vel_s' must be nonzero, but got {boost_vel_s}")

    # γ = 1 / √(1 - v ⋅ v / c²)
    lorentz_factor = 1 / np.sqrt(1 - np.square(frame_speed / light_speed))
    
    position = vec_st[..., 1:]
    time = vec_st[..., 0]

    # r' = r + v ((r ⋅ v) (γ - 1) / v² - tγ)
    position_ = position + boost_vel_s * np.expand_dims(
        np.sum(position * boost_vel_s, axis=-1) * (lorentz_factor - 1)
            / np.square(frame_speed)    # TODO: fix division by zero case
        - time * lorentz_factor,
        axis=-1)

    # t' = γ (t - (r ⋅ v) / c²)
    time_ = lorentz_factor * (time - np.sum(position * boost_vel_s, axis=-1)
        / np.square(light_speed))

    event_ = np.empty(time_.shape + (position_.shape[-1] + 1,), dtype=dtype)

    event_[..., 0] = time_
    event_[..., 1:] = position_

    return event_

def boost_velocity_s(vel_s, boost_vel_s, light_speed=1):
    '''
    Boost a space-velocity by a specified space-velocity.

    This operation uses the Lorentz vector transformations as described here:
    `Lorentz Transformation: Transformation of velocities - Wikipedia
    <https://en.wikipedia.org/wiki/Lorentz_transformation#Transformation_of_velocities>`_

    Batched inputs are supported, to allow boosting multiple space-velocities
    by multiple boost space-velocities in one call.

    A single space-velocity is given by a 1-D array of N elements, derivatives
    of the spatial dimensions with respect to coordinate time. For example, the
    space-velocity ``(v_x, v_y, v_z)`` can also be written ``(dx/dt, dy/dt,
    dz/dt)``.

    Args:

      vel_s (array_like, optional):
        Space-velocities to be boosted.

        Shape: (..., N)

        Default: None

      boost_vel_s (array_like):
        Space-velocity to use as the boost velocity of the Lorentz
        transformation.

        Shape: (..., N)

      light_speed (array_like, optional):
        Scalar speed of light.

        Default: 1

    Returns:
      ndarray: The boosted space-velocity
    '''
    boost_vel_s = np.array(boost_vel_s)
    light_speed = np.array(light_speed)
    vel_s = np.array(vel_s)

    is_scalar = (boost_vel_s.ndim == 0) and (vel_s.ndim == 0)

    if boost_vel_s.ndim == 0:
        boost_vel_s = np.array([boost_vel_s])

    # TODO: I don't like messing around with changing the sizes for scalar
    # inputs. Maybe I should just disallow scalar vel_s at first
    if vel_s.ndim == 0:
        vel_s = np.array([vel_s])

    frame_speed = np.linalg.norm(boost_vel_s, axis=-1)

    # TODO: If boost_vel_s is batched, we should only print out the
    # first speed in frame_speed that is greater than light_speed
    check((frame_speed < light_speed).all(), ValueError,
        "the norm of 'boost_vel_s' must be less than ",
        f"'light_speed' ({light_speed}), but got {frame_speed}")

    # TODO: Would batching the speed of light be useful at all? Probably best to
    # wait and see before adding batching.
    check(light_speed.ndim == 0, ValueError,
        "expected 'light_speed' to have 0 dimensions, ",
        f"but got {light_speed.ndim}")
    check(light_speed > 0, ValueError,
        f"expected 'light_speed' to be positive, but got {light_speed}")

    dtype = np.find_common_type([
        boost_vel_s.dtype,
        light_speed.dtype,
        vel_s.dtype,
    ], [])

    speed = np.linalg.norm(vel_s, axis=-1)
    check((speed <= light_speed).all(), ValueError,
        "the norm of 'vel_s' must be less than or equal to ",
        f"'light_speed' ({light_speed}), but got {speed}")

    # Change dtypes to match each other
    boost_vel_s = boost_vel_s.astype(dtype)
    light_speed = light_speed.astype(dtype)
    vel_s = vel_s.astype(dtype)

    # TODO: Need to check up front whether the args can broadcast with each other.

    if frame_speed.ndim == 0:
        if frame_speed == 0:
            return vel_s
    else:
        # TODO: This case should be supported, but will require a condition
        # below to prevent the division by zero
        check((frame_speed > 0).all(), ValueError,
            f"'boost_vel_s' must be nonzero, but got {boost_vel_s}")

    # γ = 1 / √(1 - v ⋅ v / c²)
    lorentz_factor = 1 / np.sqrt(1 - np.square(frame_speed / light_speed))

    # u' = (u / γ - v + (γ (u ⋅ v) v) / (c² (γ + 1)))
    #      / (1 - u ⋅ v / c²)
    u_dot_v = np.expand_dims(
        np.sum(vel_s * boost_vel_s, axis=-1),
        axis=-1)

    outer_factor = 1 / (1 - (u_dot_v / (light_speed**2)))
    inner_factor = np.expand_dims(
        (lorentz_factor / (lorentz_factor + 1)) / (light_speed**2),
        axis=-1)

    # TODO: Probably should expand this above, where it's first calculated
    L = np.expand_dims(lorentz_factor, axis=-1)

    vel_s_ = outer_factor * (vel_s / L - boost_vel_s + inner_factor * u_dot_v * boost_vel_s)

    if is_scalar:
        return vel_s_[0]
    else:
        return vel_s_

def proper_time_delta(event0, event1):
    '''
    Calculate the proper time between two events.

    The result preserves the sign of the difference between the time
    coordinates of the two events. So if ``event1`` is in the future with
    respect to ``event0``, the result is positive; otherwise, the result is
    negative.

    Args:

      event0 (array_like):
        First event
        Shape: (..., N+1)

      event1 (array_like):
        Second event
        Shape: (..., N+1)

    Returns:
      ndarray:
        The proper time between the two events.
        Shape: (...)
    '''
    event0 = np.array(event0)
    event1 = np.array(event1)

    check(event0.shape[-1] >= 2 and event0.shape[-1] == event1.shape[-1], ValueError,
        "expected events to have same number of spacetime dims, and to be at "
        f"least 2, but got event0: {event0.shape[0]}, event1: {event1.shape[0]}")

    diff = event1 - event0
    norm2 = -norm2_st(diff)

    check((norm2 >= 0).all(), ValueError,
        "expected events to have time-like interval, but got space-like")
    norm = np.sqrt(norm2)

    # Preserve the sign from time coordinate difference
    sign = np.where(np.signbit(diff[..., 0]), -1, 1)

    return sign * norm

def norm_s(vec_s):
    '''
    Calculate the norm of a space-vector. This is simply the Euclidean norm, or
    more formally, the `L-2 norm <https://mathworld.wolfram.com/L2-Norm.html>`_.

    Given an N-dimensional space-vector ``a = (a1, ..., aN)``, the norm is
    ``norm_s(a) = sqrt(a1^2 + ... + aN^2)``.

    For instance, if the space-vector is the difference between the coordinates
    of two positions in space, then its norm is the distance between the events.

    Args:

      vec_s (array_like):
        Any space-vector

        Shape: (..., N)
    '''
    return np.linalg.norm(vec_s, axis=-1)

def norm2_st(vec_st):
    '''
    Calculate the square of the norm of a spacetime-vector.

    Given an N+1-dimensional spacetime-vector ``a = (a0, a1, ..., aN)``, the
    square norm is ``norm2_st(a) = - a0^2 + a1^2 + ... + aN^2``.

    This is traditionally called the `squared norm of a four-vector
    <https://mathworld.wolfram.com/Four-VectorNorm.html>`_

    For instance, if the spacetime-vector is the difference between the
    coordinates of two events, then its squared norm is the square of proper
    distance (or the space-like interval) between the events. The negative of
    the squared norm is the proper time (or the time-like interval) between the
    events.

    Args:

      vec_st (array_like):
        Any spacetime-vector

        Shape: (..., N+1)
    '''
    vec_st = np.array(vec_st)
    return -vec_st[..., 0]**2 + np.linalg.norm(vec_st[..., 1:], axis=-1)**2

def velocity_st(vel_s, light_speed=1):
    '''
    Calculates the spacetime-velocity vector from a space-velocity vector.

    Given a space-velocity ``v = (v1, ..., vN)``, the spacetime-velocity is
    calculated by ``(1 , v1, ..., vN) / sqrt(1 - |v|**2)``.

    Spacetime-velocity is traditionally called four-velocity in the context of
    3+1 Minkowski spacetime.

    This is the reverse of :func:`spacetime.velocity_s`.

    Args:

      vel_s (array_like):
        Space-velocity of a particle, given by the derivative of each space
        dimension with respect to coordinate time. If the norm of the
        space-velocity is equal to or greater than the speed of light, then
        the spacetime-velocity is undefined and an error will raise.

        Shape: (..., N)

      light_speed (array_like, optional):
        Scalar Speed of light.

        Default: 1
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
    if not np.allclose(-norm2_st(vel_st), light_speed, atol=0.01):
        raise ValueError(
            'Due to floating point error, one of the given space-velocities '
            'gave a spacetime-velocity whose proper time is not 1. I hope this '
            'error never happens.')

    return vel_st

def velocity_s(vel_st):
    '''
    Calculates the space-velocity vector from a spacetime-velocity vector.

    This is the reverse of :func:`spacetime.velocity_st`.

    Args:

      vel_st (array_like):
        Spacetime-velocity of a particle, given by the derivative of each
        dimension of spacetime (coordinate time dimension comes first) with
        respect to proper time.

        Shape: (..., N+1)
    '''
    return vel_st[..., 1:] / np.expand_dims(vel_st[..., 0], -1)
