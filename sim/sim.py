import itertools
import pygame
import numpy as np

from frame import Frame2D, Clock
from transformations import boost

rest_frame = Frame2D()

render_clock_times = False

demo_number = 0

if demo_number == 0:
    # Create a grid of stationary clocks around the origin
    for i, j in itertools.product(range(21), range(21)):
        rest_frame.append(Clock(
            0,
            20 * np.array([0, i-10, j-10]),
            (0, 0)))

    rest_frame.append(Clock(
        0,
        (0, 0, 0),
        (0.9, 0)))

    rest_frame.append(Clock(
        0,
        (0, 0, 0),
        (1, 0)))

    render_clock_times = True

elif demo_number == 1:
    # Demonstration of one of the connections between electromagnetic inductance
    # and special relativity
    num_charges = 100
    for i in range(num_charges + 1):
        rest_frame.append(Clock(
            0,
            (0, -10, (i - num_charges/2) * 5),
            (0, -.9)))

        rest_frame.append(Clock(
            0,
            (0, 10, (i - num_charges/2) * 5),
            (0, .9)))

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

observer_frame = rest_frame.boost(
    observer_frame_disp,
    observer_frame_velocity)


pygame.init()
pygame.font.init()

screen = pygame.display.set_mode([800, 800])
pygame_clock = pygame.time.Clock()
running = True
display_scale = 10

my_font = pygame.font.SysFont('ubuntumono', 16)

is_clock_ticking = True

observer_frame_state = observer_frame.get_state_at_time(observer_frame_time)

while running:
    # Detect quit
    for pygame_event in pygame.event.get():
        if pygame_event.type == pygame.QUIT:
            running = False

        elif pygame_event.type == pygame.KEYDOWN:
            new_clock_velocity_ = None

            if pygame_event.key == pygame.K_SPACE:
                is_clock_ticking = not is_clock_ticking

            elif pygame_event.key == pygame.K_w:
                new_clock_velocity_ = np.array([0, 0.9])

            elif pygame_event.key == pygame.K_s:
                new_clock_velocity_ = np.array([0, -0.9])

            elif pygame_event.key == pygame.K_a:
                new_clock_velocity_ = np.array([-0.9, 0])

            elif pygame_event.key == pygame.K_d:
                new_clock_velocity_ = np.array([0.9, 0])

            if new_clock_velocity_ is not None:
                velocity = rest_frame._clocks[-1]._velocity
                event0_ = observer_frame_state[-1][1]

                new_clock_event0, new_clock_velocity = boost(
                    -rest_frame._clocks[-1]._velocity,
                    event0_,
                    new_clock_velocity_)

                event0 = new_clock_event0 + observer_frame_disp

                rest_frame._clocks.insert(
                    -1,
                    Clock(
                        0,
                        event0,
                        new_clock_velocity))

                observer_frame = rest_frame.boost(
                    observer_frame_disp,
                    velocity)




    # Detect key presses for changing velocity
    keys_pressed = pygame.key.get_pressed()
    add_velocity_direction = np.array([0, 0])

    control_speed = 0.05

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
        add_velocity = (control_speed / norm) * add_velocity_direction 

    observer_frame_state = observer_frame.get_state_at_time(observer_frame_time)
    observer_clock_face_time = observer_frame_state[-1][0]

    if add_velocity is not None:
    #if True:
        # TODO: At the moment, the observer's clock face is correct, but all
        # the other clocks are getting reset to 0. Need to fix that.
        # A possible way to try to solve the problem is to make this branch
        # always run even when `add_velocity is None`. I tried it and
        # a few things are very wonky. When this branch works correctly,
        # it must give the same exact result as not taking the branch
        # if we're not accelerating. In other words, applying zero acceleration
        # must give the same result as not applying any acceleration.
        # The wonky things are that only the observer clock ticks at all and
        # when you stop accelerating, your velocity goes right down to zero.
        # Okay, I fixed the clock face thing. But I still don't know why
        # the observer stops completely when not accelerating.


        # Find the new position and velocity of the observer clock in the
        # rest frame
        clock_velocity_ = add_velocity
        clock_event_ = observer_frame_state[-1][1]

        clock_event, clock_velocity = boost(
            -rest_frame._clocks[-1]._velocity,
            clock_event_,
            clock_velocity_)

        # Need to add the current observer frame's displacement to get the
        # correct event from the rest frame's perspective
        clock_event = observer_frame_disp + clock_event

        # Now that we have a new event and velocity for the observer clock,
        # create a new clock and replace the old one in the rest frame
        new_observer_clock = Clock(
            observer_clock_face_time,
            clock_event,
            clock_velocity)

        rest_frame._clocks[-1] = new_observer_clock

        observer_frame = rest_frame.boost(
            clock_event,
            clock_velocity)
        observer_frame_time = 0

        observer_frame_disp = clock_event


    # Display everything
    screen.fill((0, 0, 0))
    for idx, (face_time, event) in enumerate(observer_frame_state):


        draw_position = (
            display_scale * event[1] + 400,
            -display_scale * event[2] + 400)
        
        if idx == len(observer_frame_state) - 1:
            dot_color = (255, 255, 255)
            text_color = (255, 255, 255)
            text_position = (
                draw_position[0],
                draw_position[1] - 20)

            # Find the observer clock's position and velocity in the rest frame
            # to be displayed
            observer_clock = rest_frame._clocks[-1]
            velocity = observer_clock._velocity
            rest_pos = boost(
                -velocity,
                event
            # TODO: I thought this should be a subtraction, but addition
            # gives the correct result?? Need to figure out why
            )[0][..., 1:] + observer_frame_disp[1:]

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

        if render_clock_times:
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

