import numpy as np
from numbers import Number

def check(condition, error_type, message):
    if not condition:
        raise error_type(message)

# Transforms an event from one inertial frame, S, to a another inertial frame,
# S', using the Lorentz vector transformations
# (https://en.wikipedia.org/wiki/Lorentz_transformation#Vector_transformations).
# This function works for any number of spatial dimensions.
#
# Arguments:
#   
#   frame_velocity : array_like
#       Velocity of frame S' relative to S
#
#   position : array_like
#       Spatial position of the particle
#
#   time : array_like
#       Temporal position of the particle
#
#   velocity : array_like, optional
#       Velocity of the particle. If given, the output `velocity_` will be
#       the Lorentz transformed velocity of the particle in frame S'.
#       Default: None
#
#   light_speed : array_like, optional
#       Speed of light. Default: 1
#
# Returns:
#
#   position_, time_, velocity_ : tuple of ndarray
#
# Shape:
#
#   * frame_velocity: (..., N)
#
#   * position: (..., N)
#
#   * time: (...)
#
#   * velocity: same as `position`
#
#   * light_speed: ()
#
#   where N is the number of spatial dimensions.
#
#   `position`, `time`, and `frame_velocity` can all be batched, to calculate
#   multiple transformations. In this case, NumPy broadcasting semantics are
#   used: https://numpy.org/doc/stable/user/basics.broadcasting.html
#   
def transform(frame_velocity, position, time, velocity=None, light_speed=1):
    position = np.array(position)
    time = np.array(time)
    frame_velocity = np.array(frame_velocity)
    light_speed = np.array(light_speed)

    check(position.ndim > 0, ValueError,
          "expected 'position' to have one or more dimensions, "
          f"but got {position.ndim}")
    check(frame_velocity.ndim > 0, ValueError,
          "expected 'frame_velocity' to have one or more dimensions, "
          f"but got {frame_velocity.ndim}")
    check(position.shape[-1] == frame_velocity.shape[-1], ValueError,
          "expected 'position.shape[-1]' and 'frame_velocity.shape[-1]' to be "
          "equal, but got {position.shape[-1]} and {frame_velocity.shape[-1]'")
    check(light_speed.ndim == 0, ValueError,
          "expected 'light_speed' to have 0 dimensions, "
          f"but got {light_speed.ndim}")
    check(light_speed > 0, ValueError,
          f"expected 'light_speed' to be positive, but got {light_speed}")

    if velocity is not None:
        velocity = np.array(velocity)
        check(position.shape == velocity.shape, ValueError,
              "expected 'position' and 'velocity' to have the same shape, "
              f"but got {position.shape} and {velocity.shape}")
        speed = np.linalg.norm(velocity, axis=-1)
        check((speed <= light_speed).all(), ValueError,
              "the norm of 'velocity' must be less than or equal to "
              f"'light_speed' ({light_speed}), but got {speed}")

    frame_speed = np.linalg.norm(frame_velocity, axis=-1)

    # TODO: If frame_velocity is batched, we should only print out the
    # first speed in frame_speed that is greater than light_speed
    check((frame_speed < light_speed).all(), ValueError,
          "the norm of 'frame_velocity' must be less than "
          f"'light_speed' ({light_speed}), but got {frame_speed}")

    if frame_speed.ndim == 0:
        if frame_speed == 0:
            return position, time, velocity
    else:
        # TODO: This case should be supported, but will require a condition
        # below to prevent the division by zero
        check((frame_speed > 0).all(), ValueError,
              f"'frame_velocity' must be nonzero, but got {frame_velocity}")

    # γ = 1 / √(1 - v ⋅ v / c²)
    lorentz_factor = 1 / np.sqrt(1 - np.square(frame_speed / light_speed))

    # r' = r + v ((r ⋅ v) (γ - 1) / v² - tγ)
    position_ = position + frame_velocity * np.expand_dims(
        np.dot(position, frame_velocity) * (lorentz_factor - 1)
            / np.square(frame_speed)    # TODO: fix division by zero case
        - time * lorentz_factor,
        axis=-1)

    # t' = γ (t - (r ⋅ v) / c²)
    time_ = lorentz_factor * (time - np.dot(position, frame_velocity)
        / np.square(light_speed))

    if velocity is not None:
        # u' = (u / γ - v + (γ (u ⋅ v) v) / (c² (γ + 1)))
        #      / (1 - u ⋅ v / c²)

        u_dot_v = np.dot(velocity, frame_velocity)

        outer_factor = 1 / (1 - (u_dot_v / (light_speed**2)))
        inner_factor = (lorentz_factor / (lorentz_factor + 1)) / (light_speed**2)

        velocity_ = outer_factor * (velocity / lorentz_factor - frame_velocity + inner_factor * u_dot_v * frame_velocity)

    else:
        velocity_ = None

    return position_, time_, velocity_
