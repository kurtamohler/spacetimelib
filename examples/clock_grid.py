import itertools
import pygame
import numpy as np

from spacetime import Frame, boost, boost_velocity_s, Worldline

rest_frame = Frame()

render_proper_times = True

demo_number = 0

if demo_number == 0:
    # Create a grid of stationary worldlines around the origin
    for i, j in itertools.product(range(7), range(7)):
        rest_frame.append(Worldline(
            [20 * np.array([0, i-3, j-3])],
            [0, 0],
            proper_time_origin=0))

elif demo_number == 1:
    R = 20
    N = 30

    for i in range(N):
        angle = 2 * np.pi * i / 20

        rest_frame.append(Worldline(
            [R * np.array([0, np.sin(angle), np.cos(angle)])],
            [0, 0],
            proper_time_origin=0))

elif demo_number == 2:
    num_charges = 100
    for i in range(num_charges + 1):
        for direction in [-1, 1]:
            rest_frame.append(Worldline(
                [(0, 10 * direction, (i - num_charges/2) * 5)],
                [0, 0.5 * direction],
                proper_time_origin=0))

elif demo_number == 3:
    N = 100
    spacing = 4

    for i in range(N):
        rest_frame.append(Worldline(
            # TODO: If I use light speed line segments here, there's an NaN
            # error when calculating proper time
            np.array([
                [0, -4, 0],
                [10, 4, 0],
                [20, -4, 0],
                [30, 4, 0],
                [40, -4, 0],
                [50, 4, 0],
                [60, -4, 0],
                [70, 4, 0],
                [80, -4, 0],
                [90, 4, 0],
            ]) + (-40.5, 0, spacing * i - spacing * (N//2)),
            [0, 0],
            proper_time_origin=0))

elif demo_number == 4:
        rest_frame.append(Worldline(
            # TODO: If I use light speed line segments here, there's an NaN
            # error when calculating proper time
            np.array([
                [0, 0, 0]
            ]),
            vel_ends=[0, 0],
            proper_time_origin=0,
            proper_time_offset=123))

# Always keep the displacement of the current instantaneous
# observer frame
observer_frame_disp = np.array([0, 0, 0])
observer_frame_time = 0

# Add a worldline for the observer
# TODO: I should improve the interface for this kind of thing. Maybe allow
# worldlines to be named, and they can be accessed from the frame by name
rest_frame.append(
    Worldline(
        [observer_frame_disp],
        np.array([0, 0]),
        proper_time_origin=observer_frame_disp[0]),
    'observer')

observer_frame = rest_frame.boost(
    rest_frame['observer'].vertex(0),
    rest_frame['observer'].vel_future)


pygame.init()
pygame.font.init()

screen = pygame.display.set_mode([800, 800])
pygame_clock = pygame.time.Clock()
running = True
display_scale = 10

my_font = pygame.font.SysFont('ubuntumono', 16)

is_clock_ticking = True

observer_frame_state = observer_frame.eval(observer_frame_time)

while running:
    # Detect quit
    for pygame_event in pygame.event.get():
        if pygame_event.type == pygame.QUIT:
            running = False

        elif pygame_event.type == pygame.KEYDOWN:
            new_worldline_velocity_ = None

            if pygame_event.key == pygame.K_SPACE:
                is_clock_ticking = not is_clock_ticking

            elif pygame_event.key == pygame.K_w:
                new_worldline_velocity_ = np.array([0, 0.9])

            elif pygame_event.key == pygame.K_s:
                new_worldline_velocity_ = np.array([0, -0.9])

            elif pygame_event.key == pygame.K_a:
                new_worldline_velocity_ = np.array([-0.9, 0])

            elif pygame_event.key == pygame.K_d:
                new_worldline_velocity_ = np.array([0.9, 0])

            if new_worldline_velocity_ is not None:
                # TODO: Fix this hack
                velocity = rest_frame['observer'].vel_past
                event0_ = observer_frame['observer'].eval(observer_frame_time)

                new_worldline_event0 = boost(
                    event0_,
                    -velocity)

                new_worldline_velocity = boost_velocity_s(new_worldline_velocity_, -velocity)



                event0 = new_worldline_event0 + rest_frame['observer'].vertex(0)

                rest_frame.append(Worldline(
                    [event0],
                    new_worldline_velocity,
                    proper_time_origin=event0[0]))

                observer_frame = rest_frame.boost(
                    rest_frame['observer'].vertex(0),
                    velocity)




    # Detect key presses for changing velocity
    keys_pressed = pygame.key.get_pressed()
    add_velocity_direction = np.array([0, 0])

    control_speed = 0.05

    # TODO: There's something wrong about how I'm adding velocities to the
    # observer. If the observer is going at velocity, say (0.1, 0.1) wrt the
    # rest frame and shoots a particle to the right, then tries to catch up to
    # it by accelerating to the right, the particle starts moving vertically
    # upward. That should not happen, since I want the acceleration direction
    # to map correctly to the observer's frame. This problem doesn't happen
    # when the orthogonal component of the observer's velocity in the rest
    # frame is zero. The issue doesn't seem like it could be just floating point
    # error, since it happens with these fairly small velocities.

    # UPDATE: Actually, after thinking about it enough, I do think it must be
    # a numerical error. That would explain why we don't get this effect when
    # the observer is at the origin. I should be able to solve it by centering
    # the rest frame on the observer before performing the acceleration. This
    # way, a worldline that has the same spatial position and velocity as the
    # observer in, say, the x-axis will be locked in at exactly 0 for the
    # x-position and x-velocity, regardless of the relative y-positions and
    # y-velocities of the observer and woldline. I'm almost completely
    # convinced that the numerical error is coming from an initially small
    # difference in the x-position that come from displacing and boosting two
    # worldlines that differ in y and t but match in x with a nonzero value,
    # and with boost velocity where x-direction is 0. In such a case, even
    # though mathematically the x positions of the worldlines should end up
    # being equal, I'm pretty sure we can get a small numerical error. That
    # small numerical error then becomes magnified as the observer continues to
    # accelerate toward the other worldline, due to length contraction making
    # any worldline with a slightly different trajectory rotate away. So,
    # I can still keep the rest frame at a fixed velocity (to avoid
    # numerical errors from boosting repeatedly, as I originally intended) but
    # keep the rest frame's origin on the observer.

    if keys_pressed[pygame.K_LEFT]:
        add_velocity_direction += np.array([-1, 0])

    if keys_pressed[pygame.K_RIGHT]:
        add_velocity_direction += np.array([1, 0])

    if keys_pressed[pygame.K_DOWN]:
        add_velocity_direction += np.array([0, -1])

    if keys_pressed[pygame.K_UP]:
        add_velocity_direction += np.array([0, 1])

    if (add_velocity_direction == 0).all():
        add_velocity = None
    else:
        norm = np.linalg.norm(add_velocity_direction)
        add_velocity = control_speed * (add_velocity_direction / norm)

    # Reset observer velocity to 0 wrt rest frame
    if keys_pressed[pygame.K_r]:
        add_velocity = -rest_frame['observer'].vel_future

    if add_velocity is not None:
        # Create a new worldline for the observer, boost it back to the rest
        # frame, replace the old observer worldline in the rest frame, and then
        # boost the rest frame to get the new observer frame
        
        event = observer_frame['observer'].eval(observer_frame_time)
        proper_time = observer_frame['observer'].proper_time(observer_frame_time)

        new_observer_worldline_ = Worldline(
            [event],
            vel_ends=add_velocity,
            proper_time_origin=float(event[0]),
            proper_time_offset=proper_time)

        new_observer_worldline = new_observer_worldline_.boost(
            -rest_frame['observer'].vel_future) + rest_frame['observer'].vertex(0)

        rest_frame['observer'] = new_observer_worldline
        observer_frame_time = 0
        
        observer_frame = rest_frame.boost(
            rest_frame['observer'].vertex(0),
            rest_frame['observer'].vel_future)

    observer_frame_state = observer_frame.eval(observer_frame_time)
    observer_worldline_face_time = observer_frame['observer'].eval(observer_frame_time)[0]

    # Display everything
    screen.fill((0, 0, 0))
    for name, event, face_time in observer_frame_state:
        draw_position = (
            display_scale * event[1] + 400,
            -display_scale * event[2] + 400)

        if name == 'observer':
            dot_color = (255, 255, 255)
            text_color = (255, 255, 255)
            text_position = (
                draw_position[0],
                draw_position[1] - 20)

            # Find the observer worldline's position and velocity in the rest frame
            # to be displayed
            # TODO: This is a bit of a hack. Should probably add a method to
            # `Worldline` that gives the velocity at a particular time
            velocity = rest_frame['observer'].vel_future
            rest_pos = boost(
                event,
                -velocity
            # TODO: I thought this should be a subtraction, but addition
            # gives the correct result?? Need to figure out why
            )[..., 1:] + rest_frame['observer'].vertex(0)[1:]

            screen.blit(
                my_font.render(
                    f'position x: {rest_pos[0]}', False, (255, 255, 255)),
                (10, 10)
            )
            screen.blit(
                my_font.render(
                    f'position y: {rest_pos[1]}', False, (255, 255, 255)),
                (10, 25)
            )

            screen.blit(
                my_font.render(
                    f'velocity x: {velocity[0]}', False, (255, 255, 255)),
                (10, 50)
            )
            screen.blit(
                my_font.render(
                    f'velocity y: {velocity[1]}', False, (255, 255, 255)),
                (10, 65)
            )
        else:
            dot_color = (0, 160, 0)
            text_color = (150, 255, 150)
            text_position = (
                draw_position[0],
                draw_position[1] + 5)

        pygame.draw.circle(
            screen,
            dot_color,
            draw_position,
            2)

        if render_proper_times:
            text = my_font.render(f'{int(face_time)}', False, text_color)
            screen.blit(text, text_position)

    if not is_clock_ticking:
        screen.blit(
            my_font.render(
                'time is frozen [spacebar] to unfreeze', False, (255, 255, 255)),
            (10, 90)
        )

    pygame.display.flip()

    # Measure the time in seconds since last frame update
    time_delta = pygame_clock.tick(20) * 0.001

    # Iterate the observer frame's time
    if is_clock_ticking:
        observer_frame_time = observer_frame_time + time_delta


pygame.quit()

