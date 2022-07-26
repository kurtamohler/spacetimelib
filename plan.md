# Special relativity simulation tool

## Description

A tool for simulating motion in Minkowski spacetime.


## Minimum Viable Product

These features should be done first, before thinking too much about further
steps.

* There needs to be a representation of an inertial reference frame. Objects
  and their paths can be added to it. It should have a function that returns
  a new frame transformed by an arbitrary velocity, position, and time.

* Ability to create a stationary clock. Clocks can be added to a frame. A clock
  has a constant velocity for the entire time axis. The clock is initialized
  with a velocity, an event, and an initial time-reading for the clock.  For
  instance, if the event is `(t=0, x=1, y=3)` and the initial time is `10`
  seconds, then at time `t=0` in the frame it was added to, the clock is at
  spatial position `(x=1, y=3)`, and it reads `10` seconds. At different times in
  the frame, forward and back, the clock moves in a straight line according to
  the constant velocity it was given, and the reading on the clock face changes
  according to the dτ/dt that corresponds with the clock's velocity.

* Create an interactive 2-D topdown demo. It shows the spatial positions and
  readings of all the clocks at some particular time t' in an observer frame.
  The player controls the observer frame, which can change velocity. There should
  probably be a little ship in the middle of the screen or something. The change
  in proper time of the observer frame should always match that of the player in
  real life. In other words, the player should feel as if they are in the same
  reference frame as the observer frame in the simulation. The plane of
  simultaneity must change proportionally to the change in velocity.
  The motions of all the clocks should be defined ahead of time in a rest
  frame that does not change at all during the simulation.


## Extra ideas

I would love to be able to rotate the observer frame in the demo.

Game engines usually have a hierarchical tree structure which serves as the
main interface for adding and manipulating objects in the space.  For a special
relativity engine, a tree like this might also be useful. Each node could
potentially be its own full reference frame object. I need to think more about
this.

I want it to be possible for a clock's path to be finite. Instead of defining
just one point through which an infinitely long line passes, you would define
two end-points. Both endpoints could have an option whether it's a finite end
or not--which would allow for the line being infinitely long in one direction,
but not another. Maybe would also be a good idea to allow the path to be
defined by any number of points, like I experimented with in the past, but
I still worry about performance issues. I could look into ways of simplifying
the parts of paths of objects that are outside of the future lightcone of the
observer. And I could just completely delete parts of paths that fall inside
the observer's past lightcone, since there's no way for the observer's plane of
simultaneity to intersect with those events ever again-- that is, if I don't
want to support time travel. These garbage collection features could just be
offered as options, so that you can support time travel.


