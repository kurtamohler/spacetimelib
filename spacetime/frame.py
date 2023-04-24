import numpy as np
from copy import copy
from collections import OrderedDict
from itertools import islice
import re

from .basic_ops import boost, boost_velocity_s
from .worldline import Worldline
from .error_checking import check, internal_assert, maybe_wrap_index

class Frame:
    '''
    An inertial reference frame is represented as a set of :class:`Worldline`s.

    Each :class:`Worldline` can be given a name. Otherwise a default name of the
    format "wl<number>" with be automatically assigned to it.

    :class:`Worldline`s can be accessed either by name
    '''

    def __init__(self, worldlines=None, names=None):
        self._worldlines = OrderedDict()
        self._next_default_name_id = 0

        # TODO: Separate this into an `_append_multiple` function, and move the
        # current `append` impl  to `_append_single`. Then make `append`
        # conditionally call either the single or multiple append func
        if worldlines is not None:
            check(isinstance(worldlines, (list, tuple)), TypeError,
                "Expected 'worldlines' to be a list or tuple, but got ",
                f"{type(worldlines)}")

            if names is not None:
                check(isinstance(names, (list, tuple)), TypeError,
                    "Expected 'names' to be a list or tuple, but got ",
                    f"{type(names)}")
                check(len(names) == len(worldlines), ValueError,
                    "Expected 'len(names) == len(worldlines)', but got '",
                    f"{len(names)} != {len(worldlines)}'")

            for idx, worldline in enumerate(worldlines):
                name = names[idx] if names is not None else None
                self.append(worldline, name)

        else:
            check(names is None, ValueError, "'names' cannot be given without 'worldlines'")

    def _gen_default_name(self):
        cur_id = self._next_default_name_id
        name = f'wl{cur_id}'
        internal_assert(name not in self._worldlines.keys())
        self._next_default_name_id += 1
        return name

    def append(self, worldline, name=None):
        assert isinstance(worldline, Worldline)

        if name is not None:
            assert isinstance(name, str)
            check(name not in self._worldlines.keys(), ValueError,
                f"The name '{name}' is already used by a different worldline")

            # If the user-defined name matches the default name format, maybe
            # reset the next default ID to avoid generating a duplicate later
            if re.match(r'^wl[0-9]*$', name):
                name_id = int(name[2:])
                self._next_default_name_id = max(name_id + 1, self._next_default_name_id)
        else:
            name = self._gen_default_name()

        self._worldlines.update({name: worldline})

    def get_state_at_time(self, time):
        '''
        Returns the proper times and events for all worldlines at the specified time.
        '''
        state = []

        # TODO: Look into parallelizing this into one batched operation, as
        # in `Frame.boost`.
        for _, w in self._worldlines.items():
            proper_time = w.proper_time(time)
            event = w.eval(time)
            state.append((proper_time, event))

        return state

    def __getitem__(self, key):
        check(isinstance(key, (int, str)), TypeError,
            f"key must be either int or str, but got {type(key)}")

        if isinstance(key, int):
            idx = maybe_wrap_index(key, len(self))
            return next(islice(self._worldlines.values(), idx, None))

        else:
            return self._worldlines[key]

    def __setitem__(self, key, value):
        check(isinstance(key, (int, str)), TypeError,
            f"key must be either int or str, but got {type(key)}")
        check(isinstance(value, Worldline), TypeError,
            f"value must be a Worldline, but got {type(value)}")

        if isinstance(key, int):
            idx = maybe_wrap_index(key, len(self))
            #return next(islice(self._worldlines.values(), idx, None))
            check(False, NotImplementedError, "")

        else:
            self._worldlines[key] = value

    def name(self, idx):
        check(isinstance(idx, int), TypeError,
            f"idx must be an int, but got {type(idx)}")
        idx_wrapped = maybe_wrap_index(idx, len(self))
        return next(islice(self._worldlines.keys(), idx, None))

    # TODO: The complexity of this operation is O(N), so I should either figure
    # out a way to decrease the complexity or remove the need for this function
    # in `examples/clock_grid.py`. I should probably do the latter regardless,
    # and make the return of `get_state_at_time` include the worldline names
    def index(self, name):
        check(isinstance(name, str), TypeError,
            f"name must be a str, but got {type(name)}")
        check(name in self._worldlines, ValueError,
            f"no worldline named '{name}' was found")
        return list(self._worldlines.keys()).index(name)

    def __len__(self):
        return len(self._worldlines)

    def __str__(self):
        return f'Frame(worldlines={self._worldlines})'

    def __repr__(self):
        return str(self)

    # TODO: `event_delta` should be None by default. Actually, probably shouldn't
    # even be here--instead, add an addition function and use that? But that would
    # give worse performance though...
    def boost(self, event_delta, velocity_delta):
        '''
        Transform the frame, applying a time and position translations first,
        then applying a velocity transformation.
        '''
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

        new_worldlines = []

        batched = False

        # TODO: While this impl of batching is roughly 3x faster on my machine
        # than the non-batched path while uniformly accelerating in
        # `examples/clock_grid.py`, it is still pretty inefficient to copy all
        # the events and velocities out of the worldlines and then back into
        # the new worldlines. It would be much more efficient to always keep
        # a batched representation of the frame cached. This would be very
        # simple to implement, just separate the batched representation
        # creation into a new function and use the `lru_cache` decorator (or
        # whatever it's called) to auto-cache it. Another possibility is that
        # events and velocities of the individual worldlines could potentially
        # be views into the batched representation, but that seems harder to
        # implement while preventing broken views.
        if batched:
            vertex_count = []
            vertices = []
            proper_time_origin_events = []

            batched_velocities = []

            # Map the worldline index to an index into `velocities`, for both the
            # past and future velocity
            past_velocity_idx_map = {}
            future_velocity_idx_map = {}

            # Need to grab all event vertices and velocities from all
            # worldlines and combine them into two batched event and velocity
            # arrays that can be boosted. Need to keep track of which vertices
            # and velocities belong to which worldlines.
            for w_idx, w in enumerate(self._worldlines.values()):
                vertex_count.append(len(w._vertices))
                vertices += [vertex for vertex in w._vertices]

                if w.vel_past is not None:
                    past_velocity_idx_map[w_idx] = len(batched_velocities)
                    batched_velocities.append(w.vel_past)

                if w.vel_future is not None:
                    future_velocity_idx_map[w_idx] = len(batched_velocities)
                    batched_velocities.append(w.vel_future)

                proper_time_origin_events.append(w.eval(w.proper_time_origin))

            batched_events = np.concatenate([vertices, proper_time_origin_events])

            new_batched_events = boost(
                batched_events - event_delta,
                velocity_delta)

            new_batched_velocities = boost_velocity_s(
                batched_velocities,
                velocity_delta)

            new_vertices = new_batched_events[:len(vertices)]
            new_proper_time_origin_events = new_batched_events[len(vertices):]

            cur_vertices_idx = 0

            for w_idx, w in enumerate(self._worldlines.values()):
                if w_idx in past_velocity_idx_map:
                    past_velocity = new_batched_velocities[past_velocity_idx_map[w_idx]]
                else:
                    past_velocity = None

                if w_idx in future_velocity_idx_map:
                    future_velocity = new_batched_velocities[future_velocity_idx_map[w_idx]]
                else:
                    future_velocity = None

                num_vertices = vertex_count[w_idx]

                # TODO: Make Worldline accept 1-element ndarray to avoid this cast
                new_proper_time_origin = float(new_proper_time_origin_events[w_idx][0])
                new_w = Worldline(
                        new_vertices[cur_vertices_idx : cur_vertices_idx + num_vertices],
                        vel_past=past_velocity,
                        vel_future=future_velocity,
                        proper_time_origin=new_proper_time_origin,
                        proper_time_offset=w.proper_time_offset)

                cur_vertices_idx += num_vertices

                new_worldlines.append(new_w)

        else:
            for w_idx, w in enumerate(self._worldlines.values()):
                new_w = (w - event_delta).boost(velocity_delta)

                new_worldlines.append(new_w)

        return Frame(new_worldlines, list(self._worldlines.keys()))
