import numpy as np
from collections import OrderedDict
from itertools import islice
import re

import spacetime as st
from .error_checking import check, internal_assert, maybe_wrap_index

class Frame:
    '''
    An inertial reference frame is represented as a set of :class:`Worldline`
    s, all in the same coordinate system. All worldlines in the same frame must
    have the same number of N+1 dimensions.

    :class:`Worldline` s can be accessed either by name or by index. Index
    values correspond with the order in which worldlines were added to the
    frame.

    When a :class:`Worldline` is appended to the frame, it can be given a name.
    If no name is given, a default name of the format "wl<number>" is
    automatically assigned to it.
    '''

    def __init__(self, worldlines=None, names=None):
        self._worldlines = OrderedDict()
        self._next_default_name_id = 0

        if worldlines is None:
            check(names is None, ValueError,
                "'names' cannot be given without 'worldlines'")

        else:
            self.append(worldlines, names)


    def _gen_default_name(self):
        cur_id = self._next_default_name_id
        name = f'wl{cur_id}'
        internal_assert(name not in self._worldlines.keys())
        self._next_default_name_id += 1
        return name

    @property
    def ndim(self):
        if len(self) == 0:
            return None
        else:
            return self[0].ndim

    def _append_multiple(self, worldlines, names=None):
        # TODO: Separate this into an `_append_multiple` function, and move the
        # current `append` impl  to `_append_single`. Then make `append`
        # conditionally call either the single or multiple append func
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

        # TODO: Add ndim checks here
        for idx, worldline in enumerate(worldlines):
            if idx == 0:
                if self.ndim is not None:
                    check(worldline.ndim == self.ndim, ValueError,
                        f"expected worldlines[{idx}] to have {self.ndim} dims, ",
                        f"but got {worldline.ndim}")
            else:
                check(worldline.ndim == worldlines[0].ndim, ValueError,
                    f"expected worldlines[{idx}] to have {worldlines[0].ndim} dims, ",
                    f"but got {worldline.ndim}")


        for idx, worldline in enumerate(worldlines):
            name = names[idx] if names is not None else None
            self._append_single(worldline, name)

    def _append_single(self, worldline, name=None):
        check(isinstance(worldline, st.Worldline), TypeError,
            f"expected an object of type Worldline, but got '{type(worldline)}'")

        if self.ndim is not None:
            check(worldline.ndim == self.ndim, ValueError,
                f"expected worldline to have {self.ndim} dims, but got {worldline.ndim}")

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

    def append(self, worldline, name=None):
        if isinstance(worldline, (list, tuple)):
            self._append_multiple(worldline, name)
        else:
            self._append_single(worldline, name)

    def eval(self, time):
        '''
        Calculates the event coordinates and proper times for all worldlines at
        a specified time.

        Args:

          time (number):
            Time at which to evaluate the frame.

        Returns:
          ``list[3-tuple(str, ndarray, float)]``:
            A list containing an entry corresponding to each worldline in the
            frame, in the same order that worldlines are stored in the
            :class:`Frame`. Each entry contains the name, event, and proper
            time, in that order, for one of the worldlines at the specified
            time.
        '''

        state = []

        # TODO: Look into parallelizing this into one batched operation, as
        # in `Frame.boost`.
        for name, worldline in self._worldlines.items():
            proper_time = worldline.proper_time(time)
            event = worldline.eval(time)
            state.append((name, event, proper_time))

        return state

    def __getitem__(self, key):
        check(isinstance(key, (int, str)), TypeError,
            f"key must be either int or str, but got {type(key)}")

        if isinstance(key, int):
            idx = maybe_wrap_index(key, len(self))
            return next(islice(self._worldlines.values(), idx, None))

        else:
            check(key in self._worldlines, KeyError,
                f"worldline of name '{key}' not found")
            return self._worldlines[key]

    def __setitem__(self, key, value):
        check(isinstance(key, (int, str)), TypeError,
            f"key must be either int or str, but got {type(key)}")
        check(isinstance(value, st.Worldline), TypeError,
            f"value must be a Worldline, but got {type(value)}")

        if isinstance(key, int):
            idx = maybe_wrap_index(key, len(self))
            name = next(islice(self._worldlines.keys(), idx, None))

        else:
            check(key in self._worldlines, KeyError,
                f"worldline of name '{key}' not found")
            name = key

        check(value.ndim == self.ndim, ValueError,
            f"expected 'value.ndim' to be {self.ndim}, but got {value.ndim}")
        self._worldlines[name] = value

    def name(self, idx):
        check(isinstance(idx, int), TypeError,
            f"idx must be an int, but got {type(idx)}")
        idx_wrapped = maybe_wrap_index(idx, len(self))
        return next(islice(self._worldlines.keys(), idx, None))

    # TODO: The complexity of this operation is O(N), so I should either figure
    # out a way to decrease the complexity or remove the need for this function
    # in `examples/clock_grid.py`. I should probably do the latter regardless,
    # and make the return of `Frame.eval` include the worldline names
    def index(self, name):
        check(isinstance(name, str), TypeError,
            f"name must be a str, but got {type(name)}")
        check(name in self._worldlines, ValueError,
            f"no worldline named '{name}' was found")
        return list(self._worldlines.keys()).index(name)

    def __len__(self):
        return len(self._worldlines)

    def __str__(self):
        return f'Frame(worldlines={list(self._worldlines.items())})'

    def __repr__(self):
        return str(self)

    # TODO: This interface needs to be consistent with `st.boost` and `Worldline.boost`.
    # I suppose an event delta maybe should be added to those interfaces as well, because
    # here, we get a significant performance improvement by including the boost and
    # the offset in the same batch
    def boost(self, boost_vel_s, event_delta_pre=None, event_delta_post=None, _batched=True):
        '''
        Boost each worldline in the frame by the specified velocity.

        The :attr:`event_delta_pre` and :attr:`event_delta_post` arguments
        can be used to add optional spacetime-vector offsets to the frame
        before and after boosting. This performs the same calculation as
        calling :func:`Frame.__add__` before and after boosting, but it is
        usually significantly faster to use this combined operation.

        Args:

          boost_vel_s (array_like):
            Space-velocity to boost the frame by.

          event_delta_pre (optional, array_like):
            Displacement to add to each worldline before boosting.

            Default: ``None``

          event_delta_post (optional, array_like):
            Displacement to add to each worldline after boosting.

            Default: ``None``

        Returns:
          :class:`Frame`:
        '''
        if len(self) == 0:
            return Frame()

        # Check `event_delta_*` args
        if event_delta_pre is not None:
            event_delta_pre = np.array(event_delta_pre)
            check(event_delta_pre.shape == (self.ndim,), ValueError,
                f"'event_delta_pre' must have shape {(self.ndim,)}, "
                f"but got {event_delta_pre.shape}")

        if event_delta_post is not None:
            event_delta_post = np.array(event_delta_post)
            check(event_delta_post.shape == (self.ndim,), ValueError,
                f"'event_delta_post' must have shape {(self.ndim,)}, "
                f"but got {event_delta_post.shape}")

        boost_vel_s = np.array(boost_vel_s)
        check(boost_vel_s.shape == (self.ndim - 1,), ValueError,
            f"'boost_vel_s' must have shape {(self.ndim - 1,)}, "
            f"but got {boost_vel_s.shape}")

        speed = np.linalg.norm(boost_vel_s)
        # Don't allow faster than light transformations
        assert speed <= 1

        new_worldlines = []

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
        if _batched:
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

                if w.past_vel_s is not None:
                    past_velocity_idx_map[w_idx] = len(batched_velocities)
                    batched_velocities.append(w.past_vel_s)

                if w.future_vel_s is not None:
                    future_velocity_idx_map[w_idx] = len(batched_velocities)
                    batched_velocities.append(w.future_vel_s)

                proper_time_origin_events.append(w.eval(w.proper_time_origin))

            batched_events = np.concatenate([vertices, proper_time_origin_events])

            if event_delta_pre is not None:
                batched_events = batched_events + event_delta_pre

            new_batched_events = st.boost(
                batched_events,
                boost_vel_s)

            if event_delta_post is not None:
                new_batched_events = new_batched_events + event_delta_post

            new_batched_velocities = st.boost_velocity_s(
                batched_velocities,
                boost_vel_s)

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
                new_w = st.Worldline(
                        new_vertices[cur_vertices_idx : cur_vertices_idx + num_vertices],
                        past_vel_s=past_velocity,
                        future_vel_s=future_velocity,
                        proper_time_origin=new_proper_time_origin,
                        proper_time_offset=w.proper_time_offset)

                cur_vertices_idx += num_vertices

                new_worldlines.append(new_w)

        else:
            for w_idx, w in enumerate(self._worldlines.values()):
                if event_delta_pre is not None:
                    w = w + event_delta_pre

                new_w = w.boost(boost_vel_s)

                if event_delta_post is not None:
                    new_w = new_w + event_delta_post

                new_worldlines.append(new_w)

        return Frame(new_worldlines, list(self._worldlines.keys()))

    def __add__(self, event_delta):
        '''
        Add a displacement to all worldlines in the frame.

        Args:

          event_delta (array_like):
            Displacements to add to each dimension.

        Returns:
          :class:`Frame`:
        '''
        event_delta = np.array(event_delta)
        result = Frame()

        for idx in range(len(self)):
            name = self.name(idx)
            worldline = self[idx]
            result.append(worldline + event_delta, name)

        return result

    def __sub__(self, event_delta):
        '''
        Subtract a displacement from all worldlines in the frame.

        Args:

          event_delta (array_like):
            Displacements to subtract from each dimension.

        Returns:
          :class:`Frame`:
        '''
        return self + (-event_delta)
