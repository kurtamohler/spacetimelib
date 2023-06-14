import numpy as np

import spacetime as st
from .error_checking import check, check_type

class ObserverSim:
    '''

    :class:`ObserverSim` simulates an observer's non-inertial reference frame
    over time. This is done by calculating planes of simultaneity at increments
    of proper time along an observer's worldline. Each plane of simultaneity
    corresponds to the measurable state of all the particles in an inertial
    reference frame that instanteneously coincides with the observer's
    worldline at a particular event and velocity.

    It is possible to update the future worldline of the observer at any
    simulation step. This allows the simulation to be run on real time data.
    For example, :class:`ObserverSim` can be used to create an interactive game
    that follows the rules of special relativity.

    In order to preserve floating point accuracy, :class:`ObserverSim` holds
    onto a rest frame which is never boosted, though it may be shifted. If we
    boost a reference frame repeatedly by some series of velocities, and then
    we repeatedly boost it again by the negatives of each in reverse order, the
    output is significantly different than the first frame. So to preserve
    information that would otherwise be lost with successive boosts, we keep
    a rest frame whose velocity never changes.

    '''
    def __init__(self, frame_rest, observer_name='observer'):
        '''
        Args:

          frame_rest (:class:`Frame`):
            A frame to use as a rest frame whose velocity never changes. The
            simulation begins where the rest frame's plane of simultaneity at
            time 0 intersects with the observer's worldline.

          observer_name (str):
            Name of the observer :class:`Worldline`, which must be included in
            :attr:`frame_rest`.
        '''
        check_type(frame_rest, st.Frame, 'frame_rest')
        check_type(observer_name, str, 'observer_name')

        self.observer_name_ = observer_name

        # Center the rest frame's origin onto the observer
        # TODO: Consider not only doing space offsets, so time coordinates are
        # maintained.
        offset = frame_rest[self.observer_name_].eval(0)
        self.frame_rest_ = frame_rest - offset

        self.cur_observer_vel_s_ = self.frame_rest_[self.observer_name_].eval_vel_s(0)

        self.frame_observer_ = self.frame_rest_.boost(self.cur_observer_vel_s_)

    def step(self, proper_time_delta):
        # Find the event, in the observer's frame, that is `proper_time_delta`
        # into the future. Then, boost that event's coordinates to the rest
        # frame. Then center the rest frame on the event. Finally, boost the
        # rest frame by the new

        event_observer = self.frame_observer_[self.observer_name_].eval_proper_time(0, proper_time_delta)

        offset = st.boost(event_observer, -self.cur_observer_vel_s_)
        self.frame_rest_ = self.frame_rest_ - offset
        self.cur_observer_vel_s_ = self.frame_rest_[self.observer_name_].eval_vel_s(0)
        self.frame_observer_ = self.frame_rest_.boost(self.cur_observer_vel_s_)

    def eval(self):
        # TODO: This is a bad way to preserve the time offset
        time_offset = np.zeros(self.frame_observer_.ndim)
        time_offset[0] = self.frame_observer_[self.observer_name_].proper_time(0)
        frame_observer_adjusted = self.frame_observer_ + time_offset
        return frame_observer_adjusted.eval(time_offset[0])
