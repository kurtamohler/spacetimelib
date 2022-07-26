# SpacetimeLib

Very early stages of a tool that can calculate motion of objects in special relativity.


## Install

Install Miniconda: [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Run the following to create and activate an environment with all dependencies.

```bash
conda env create -f environment.yaml -n spacetime && conda activate spacetime
```

Then install SpacetimeLib.

```bash
python setup.py install
```

## Run

There is just one small interactive demo at the moment. Use the arrow keys to
move around. Click the spacebar to freeze time. You can also shoot projectiles
with the WASD keys. Be careful shooting too many though, performance decreases
with each new object.

```bash
python examples/clock_grid.py
```

In this demo, you control a space ship in a Minkowski spacetime with two
spatial dimensions. Your ship is the dot in the middle of the screen, and you
carry a clock with you that always ticks at the same rate as the clock on your
computer. You add horizontal and vertical velocity to your ship with the arrow
keys.

There are a bunch of clocks floating in space at rest around you, all
synchronized to the same time. Two more clocks shoot off to
your right at different speeds, one is 90% the speed of light and the other
is exactly at the speed of light.

What you see on the screen are the positions and time readings of the clocks
from your ship's frame of reference, on its plane of simultaneity.

![Animated gif of a topdown view of a Minkowski space with two spatial dimensions. In the animation, space and time warp when an observer frame changes velocity.](./images/demo_animation0.gif)

## Build documentation

```bash
sphinx-build -b html docs/source docs/build/html
```

## Good resources for learning special relativity

* [Susskind, L., & Friedman, A. (2017). _Special relativity and classical Field theory: The theoretical minimum_. Basic Books.  ](https://www.amazon.com/Special-Relativity-Classical-Field-Theory/dp/0465093345)

* [Leonard Susskind's video lectures on special relativity](https://www.youtube.com/watch?v=toGH5BdgRZ4&list=PLD9DDFBDC338226CA)
