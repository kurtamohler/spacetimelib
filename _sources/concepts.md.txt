# Concepts

## Welcome to SpacetimeLib

SpacetimeLib is an open source Python library for special relativity physics.
You can use it for things like solving textbook special relativity problems and
creating physics simulations that exhibit relativistic effects like time
dilation, length contraction, and the relativity of simultaneity.

This page describes the main concepts, terminology, and library features
required to start using SpacetimeLib. A few of the ideas of special
relativity are explained briefly, but we will not start from the very beginning
or do any derivations--there are many exising resources for that. (TODO: Create
and link to page with recommended learning resources)

Readers who haven't studied special relativity before may learn something by
reading this page and experimenting with SpacetimeLib in the interactive Python
CLI. However, it would certainly be helpful to have a basic understanding of
the subject. If you know how to calculate Lorentz transformations with one
spatial dimension, then you probably know enough to understand everything in
this document.

## Spacetime

SpacetimeLib is named after the mathematical concept of
[spacetime](https://en.wikipedia.org/wiki/Spacetime), which is a type of
topological space that has one time dimension and number of spatial dimensions.
If a spacetime has N spatial dimensions, it can be called an N+1 spacetime. The
"+1" is for the time dimension.

Special relativity focuses on a particular type of N+1 spacetime called
[Minkowski spacetime](https://en.wikipedia.org/wiki/Minkowski_space).
A defining rule of Minkowski spacetime is that a light ray always travels at
the same constant speed in every inertial reference frame, and nothing can
travel faster than light.

SpacetimeLib provides operations for manipulating objects in N+1 Minkowski
spacetimes.

## Events

An event is a single point in spacetime. We can describe the location of an
event in spacetime using a coordinate system embedded in the N+1 dimensions of
the spacetime. An event's N+1 coordinates are combined into a vector. The first
element is the time coordinate and the remaining elements are coordinates in
each dimension of space. For example, given the event `(t, x, y, z)`, `t` is
the time coordinate of the event and `x`, `y`, and `z` are spatial coordinates
along the x-, y-, and z-axes in a 3+1 spacetime.

In SpacetimeLib, we represent an event with a one dimensional arraylike
(`list`, `tuple`, or `numpy.ndarray`) containing N+1 real numbers. For example:

```python
>>> q = [10, 2, 0, 0]
```

`q` is an event at time `t = 10` and spatial position `x = 2`, `y = 0`, `z = 0`.

## Spacetime-vectors and space-vectors (classically called four-vectors and three-vectors)

In SpacetimeLib, we use the term "spacetime-vector" to mean any vector that has
N+1 elements, each corresponding to a different time or space dimension of an
N+1 spacetime. An event vector is an example of a spacetime-vector. Typically
in classical special relativity, the real world is modelled as a 3+1 Minkowski
spacetime, so in that context, a spacetime-vector is called
a ["four-vector"](https://en.wikipedia.org/wiki/Four-vector). Breaking from
traditional terminology in this case makes it easier to think with any number
of spatial dimensions.

Similarly, we use the term "space-vector" to mean any vector that has
N elements, each corresponding to a different space dimension in an N+1
spacetime. So, for instance, if we want to indicate a position in space without
including a time coordinate, we would use a space-vector. In classical physics,
a space-vector is called a "three-vector", and again we break from traditional
terminology.

## Distance and norms

To find the distance between two points, we take the norm of the difference between
them, and in general, we can take the norm of any vector.

We can use [`spacetime.norm_s`](spacetime.norm_s) to calculate the norm of any
space-vector.

```python
>>> st.norm_s([3, 4])
5.0

# Distance between two points in space
>>> a = st.asarray([-2, -1])
>>> b = st.asarray([-6, 2])
>>> st.norm_s(a - b)
5.0
```

We can use [`spacetime.norm2_st`](spacetime.norm2_st) to calculate the square
norm of any spacetime-vector. The square norm is more convenient than the norm
itself, because the square norm of a spacetime-vector can be negative.

```python
>>> st.norm2_st([1, 3, 4])
24.0

# Square of proper distance between two points in space
>>> a = st.asarray([0, 0])
>>> b = st.asarray([3, 5])
>>> st.norm2_st(a - b)
16.0
```


## Velocity

If some particle is moving through space over time relative to some other
entity (like another particle, a fixed position in space, etc), then the
particle has a nonzero velocity with respect to that entity. We can choose to
quantify the relative velocity of a particle using either a space-vector which
we call a "space-velocity", or a spacetime-vector which we call
a "spacetime-velocity". Both types of velocity vectors convey the same
information, but they have different mathematical properties, so choosing which
one to use will depend on the context.

In traditional terminology, a space-velocity is called a "three-velocity" and
a spacetime-velocity is called
a ["four-velocity"](https://en.wikipedia.org/wiki/Four-velocity).

Each element of a space-velocity is the derivative of each space dimension with
respect to coordinate time. For example, if a particle in a 3+1 spacetime has
the space-velocity `(v_x, v_y, v_z)`, then it is moving with velocity `v_x`
along the x-axis, `v_y` along the y-axis, and `v_z` along the z-axis. We can
also write the space-velocity in Leibniz notation as `(dx/dt, dy/dt, dz/dt)`.
The norm of a space-velocity must always be less than or equal to the speed of
light.

Each element of a spacetime-velocity is the derivative of each dimension of
spacetime with respect to proper time. For instance, the space-velocity
`(dx/dt, dy/dt, dz/dt)` can be converted to the spacetime-velocity `(dt/dτ,
dx/dτ, dy/dτ, dz/dτ)`, where `τ` is the proper time.

We won't go through the derivation, but there is a simple way to calculate the
spacetime-velocity from a space-velocity vector. Given a space-velocity `v
= (v1, ..., vN)`, the spacetime-velocity is calculated by `(1 , v1, ...,
vN) / sqrt(1 - norm_s(v)**2)`.

The squared norm of a spacetime-velocity must always be equal to the negative
of the speed of light.

SpacetimeLib has the functions [`spacetime.velocity_st`](spacetime.velocity_st)
and [`spacetime.velocity_s`](spacetime.velocity_s) for converting
between the two kinds of velocity vectors.

```python
# Velocity of 0.1 in the x-dimension and -0.2 in the y-dimension
>>> vel_s = [0.1, -0.2]

# The norm of the space-velocity must be less than 1, the speed of light
>>> st.norm_s(vel_s)
0.223606797749979

# Convert the space-velocity into a spacetime-velocity
>>> vel_st = st.velocity_st(vel_s)
>>> vel_st
array([ 1.02597835,  0.10259784, -0.20519567])

# The square norm of the spacetime-velocity is equal to 1, with some small error
>>> st.norm2_st(vel_st)
-1.0000000000000002

# Converting the spacetime-velocity back into a space-velocity gives the same
# space-velocity that we started with, with some small error
>>> st.velocity_s(vel_st)
array([ 0.1, -0.2])
```

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

## Frames of reference

A frame of reference is a description of the state of a spacetime from point of
view of an observer. An observer moving at a constant velocity describes their
frame of reference as a coordinate system with themself positioned at the
spatial origin, x=0, y=0, etc., for all time coordinates.

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
>>> st.boost(finish_coords, bob_vel)
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
