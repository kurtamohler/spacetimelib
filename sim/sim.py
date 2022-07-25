import itertools
import pygame
import numpy as np

from frame import Frame2D, Clock
import lorentz

rest_frame = Frame2D()

# Create a grid of stationary clocks around the origin
for i, j in itertools.product(range(21), range(21)):
    rest_frame.append(Clock(
        0,
        20 * np.array([0, i-10, j-10]),
        (0, 0)))

# Always keep the displacement of the current instantaneous
# observer frame
observer_frame_disp = np.array([0, 0, 0])

observer_frame_velocity = np.array([0, 0])
observer_frame_time = 0

# Add a clock that will represent the observer's clock
# TODO: I should improve the interface for this kind of thing. Maybe allow
# clocks to be named, and they can be accessed from the frame by name
rest_frame.append(Clock(
    0, 
    observer_frame_disp,
    observer_frame_velocity))

observer_frame = rest_frame.transform(
    observer_frame_disp,
    observer_frame_velocity)


pygame.init()

screen = pygame.display.set_mode([800, 800])


pygame_clock = pygame.time.Clock()

running = True

display_scale = 10

# Measure the time in seconds since last frame update
time_delta = pygame_clock.tick(20) * 0.001

while running:
    # Detect quit
    for pygame_event in pygame.event.get():
        if pygame_event.type == pygame.QUIT:
            running = False

    # Detect key presses for changing velocity
    keys_pressed = pygame.key.get_pressed()
    add_velocity = None
    if keys_pressed[pygame.K_LEFT]:
        add_velocity = np.array([-0.1, 0])
    elif keys_pressed[pygame.K_RIGHT]:
        add_velocity = np.array([0.1, 0])
    elif keys_pressed[pygame.K_DOWN]:
        add_velocity = np.array([0, -0.1])
    elif keys_pressed[pygame.K_UP]:
        add_velocity = np.array([0, 0.1])

    observer_frame_state = observer_frame.get_state_at_time(observer_frame_time)
    observer_clock_face_time = observer_frame_state[-1][0]

    if add_velocity is not None:
        # Find the new position and velocity of the observer clock in the
        # rest frame
        clock_velocity_ = add_velocity
        clock_event_ = observer_frame_state[-1][1]
        clock_position_ = clock_event_[1:]
        clock_time_ = clock_event_[0]

        clock_position, clock_time, clock_velocity = lorentz.transform(
            -rest_frame._clocks[-1]._velocity,
            clock_position_,
            clock_time_,
            clock_velocity_)

        # Need to add the current observer frame's displacement to get the
        # correct event from the rest frame's perspective
        clock_event = observer_frame_disp + np.concatenate([[clock_time], clock_position])

        # Now that we have a new event and velocity for the observer clock,
        # create a new clock and replace the old one in the rest frame
        new_observer_clock = Clock(
            observer_clock_face_time,
            clock_event,
            clock_velocity)

        rest_frame._clocks[-1] = new_observer_clock

        observer_frame = rest_frame.transform(
            clock_event,
            clock_velocity)
        observer_frame_time = 0
        observer_frame_disp = clock_event


    # Display everything
    screen.fill((0, 0, 0))
    for idx, (face_time, event) in enumerate(observer_frame_state):
        if idx == len(observer_frame_state) - 1:
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        pygame.draw.circle(
            screen,
            color,
            (
                display_scale * event[1] + 400,
                -display_scale * event[2] + 400
            ),
            2)

    pygame.display.flip()
    time_delta = pygame_clock.tick(20) * 0.001

    # Iterate the observer frame's time
    observer_frame_time = observer_frame_time + time_delta


pygame.quit()

