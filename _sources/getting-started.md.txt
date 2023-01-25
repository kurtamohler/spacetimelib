# Getting Started

## Welcome to SpacetimeLib

SpacetimeLib is an open source Python library for special relativity physics.
You can use it for things like solving textbook special relativity problems and
creating physics simulations that follow relativistic principles like time
dilation, length contraction, and the relativity of simultaneity.

This document describes the main concepts required to start using SpacetimeLib.
A few of the fundamental ideas of special relativity are explained briefly, but
we will not start from the very beginning or do any derivations--there are many
exising resources for that. (TODO: Create and link to page with recommended
learning resources)

Readers who haven't studied special relativity before can probably learn
a little bit by reading some of this document. However, it is recommended to
have a basic prior understanding of the subject. If you know how to calculate
Lorentz transformations with one spatial dimension, then you probably know
enough, or almost enough, to learn everything in this document.

## Installing SpacetimeLib

TODO: Fill this in after adding package to conda and pip

## How to import SpacetimeLib

To access SpacetimeLib, import it into your Python code:

```python
>>> import spacetime as st
```

We shorten the imported name to `st` for better readability of code using
SpacetimeLib, but you can name it whatever you want or just use the unshortened
name.

## Spacetime

SpacetimeLib is named after the mathematical concept of spacetime, which is
a type of topological space that has one time dimension and one or more spatial
dimensions. If a spacetime has N spatial dimensions, it can be called an N+1
spacetime. The "+1" is for the time dimension.

Special relativity focuses on a particular type of N+1 spacetime called
a Minkowski spacetime. A defining rule of Minkowski spacetime is that the speed
of light has a limit and nothing can travel faster than light.

SpacetimeLib provides operations for manipulating events, velocities, and
worldlines in an N+1 Minkowski spacetime.

## Event vectors

An event is a single point in spacetime. We can locate an event using
a coordinate system that tracks the position and time of the event, which we
combine into a vector of N+1 elements. The first element is the time
coordinate and the remaining elements are coordinates in each dimension of
space. For example, given the event `(t, x, y, z)`, `t` is the time of the
event and `x`, `y`, and `z` are spatial coordinates along the x-, y-, and
z-axes in a 3+1 spacetime.

In SpacetimeLib, we represent an event with a one dimensional arraylike
(`list`, `tuple`, or `numpy.ndarray`) containing real numbers. For example:

```python
>>> q = (10, 2, 0, 0)
```

`q` is an event at time `t = 10` and spatial position `x = 2`, `y = 0`, `z = 0`.

## Velocity vectors

A velocity vector keeps track of how fast something is moving a particular
instant in time. It is a vector of N elements. Each element is the derivative
of a spatial dimension with respect to time. For example, if a particle has the
velocity vector `(v_x, v_y, v_z)`, then it is moving with velocity `v_x` along
the x-axis, `v_y` along the y-axis, and `v_z` along the z-axis. We can show
these in Leibniz notation as `dx/dt = v_x`, `dy/dt = v_y`, and `dz/dt = v_z`.

A velocity vector is represented in SpacetimeLib as a one dimensional arraylike
containing real numbers. For example:

```python
>>> v = (0.5, 0, 0)
```

A particle with the above velocity is moves along the x-axis with velocity
`0.5` and stays fixed along the y- and z-axes, since the corresponding elements
are `0`.

## Units of measurement

By convention, SpacetimeLib uses a speed of light of `1`, so you may choose any
units of measurement that give a speed of light equal to `1`.

For example, let's say we want to measure time in seconds. We'll have to choose
a unit of distance such that light travels one of those units of distance over
one second. There is only one possible choice--it's called a light-second, the
distance that light travels in one second. Since velocity is a distance over
a time, then the our units of velocity would be light-seconds per second.

## The speed of light limit

The velocities of particles and the relative velocities between two reference
frames must always be less than or equal to the speed of light, `1`. If you
attempt to perform some operation using a velocity whose magnitude is greater
than the speed of light, an error will be thrown.

You can always use NumPy's `norm` function to find the magnitude of a velocity
vector like so:

```python
>>> import numpy as np

>>> v0 = (0.5, -0.1, 0)
>>> np.linalg.norm(v0)
0.5099019513592785

>>> v1 = (-0.9, 0.9, -0.9)
>>> np.linalg.norm(v1)
1.5588457268119895
```

Notice that `v0` is a valid velocity vector because its magnitude is less than `1`,
but `v1` is not valid because its magnitude is greater than `1`.

## Frames of reference

As we said before, an event vector is described by a coordinate within
a particular frame of reference. A frame of reference is a description of
the point of view of an observer. An observer moving at a constant velocity
describes their frame of reference as a coordinate system with themself
positioned at the spatial origin, x=0, y=0, etc., for all time coordinates.

From the point of view of two different observers moving at different
velocities, the same event will be described by different event vectors. This
is just because of the fact that two different observers have different
coordinate systems. The observers are moving with respect to each other, so
they cannot both remain at the origin in both frames, and since they are each
fixed at the origin in their own coordinate systems, their frames have to be
different.

## Boost

Let's say we have two different reference frames that are moving at a known
constant velocity with respect to each other and which share the same origin.
If we know the coordinates of some event in one of the frames, we can use the
[Lorentz transformation](https://en.wikipedia.org/wiki/Lorentz_transformation)
to find the event's coordinates in the other frame. In SpacetimeLib, the
[`spacetime.boost`](spacetime.boost) function performs a rotation-free Lorentz
transformation, also known as a Lorentz boost. A Lorentz boost is just
a Lorentz transformation in which the two reference frames' corresponding
spatial dimensions are pointed in the same directions. In other words, the
frames have different relative velocities, but not different relative spatial
rotations.

### Boost example: Superhuman footrace

Let's look at an example problem that [`spacetime.boost`](spacetime.boost) can
solve for us. Alice is watching a footrace of superhuman athletes. She stands
still at the starting line for the whole race, at `x = 0`, and the finish line
is 10 light-seconds away from her, at `x = 10`. The race begins at time `t = 0`
seconds. Bob immediately starts running at a speed of 0.8 light-seconds per
second toward the finish line. He maintains this speed for the whole race and
reaches the finish line at `t = (10 / 0.8) = 12.5` seconds.

In Alice's frame, the coordinates of the event when Bob passes the finish line
are given by the vector `(12.5, 10)`. But what are the coordinates of this same
event from Bob's perspective? Let's solve this with
[`spacetime.boost`](spacetime.boost).  We'll plug in the coordinates of the
event in Alice's reference frame and boost it by Bob's velocity, `0.8`, to get
the coordinates of the event in Bob's reference frame.

```python
>>> bob_vel = 0.8
>>> finish_coords = (12.5, 10)
>>> st.boost(bob_vel, finish_coords)
array([7.5, 0. ])
```

So in Bob's reference frame, the coordinates of the event where he passes the
finish line is `(7.5, 0)`, or `t = 7.5`, `x = 0`. It makes perfect sense that
this event occurs at `x = 0`, since Bob remains at rest at `x = 0` in his own
reference frame--he is not moving with respect to himself. But why does the
event occur at `t = 7.5` seconds, when Alice saw the race take 10 seconds
total? The answer is [time
dilation](https://en.wikipedia.org/wiki/Time_dilation). In Alice's frame of
reference, Bob's wrist watch ticks 7.5 seconds within the time that it takes
Alice's wrist watch to tick 10 seconds.

## Worldlines

A [worldline](https://en.wikipedia.org/wiki/World_line) is the path that an
object takes through spacetime. A worldline is a continuous set of events
through which an object passes. As such, we can describe a worldline by the
coordinates of each event along the worldline in a particular frame of
reference. Since every object travels at or below the speed of light, the
combined magnitude of the derivative of the wordline's position coordinates
with respect to the time coordinate must be less than or equal to the speed of
light at every point along the worldline.

In general, a worldline's position coordinates can be described by any type of
continuous function that depends on the time coordinate and obeys the speed of
light. However, in SpacetimeLib, we represent a worldline with a finite number
of event coordinates, called vertices. Between each vertex, we evaluate the
coordinate using linear interpolation. So we still have a continuous function
to describe the worldline, such that we can determine a definite position at
any moment in time between any two vertices, but the first derivative of the
worldline with respect to time is discrete.
