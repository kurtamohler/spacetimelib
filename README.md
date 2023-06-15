# SpacetimeLib

SpacetimeLib is a special relativity physics library for Python.

With SpacetimeLib, you can easily calculate Lorentz transformations, time
dilation, length contraction, and more.

## Documentation and getting started

The documentation for the latest `main` branch is found here:
[https://kurtamohler.github.io/spacetimelib/](https://kurtamohler.github.io/spacetimelib/)

The [Start Here](https://kurtamohler.github.io/spacetimelib/start-here.html)
page teaches the basics of using SpacetimeLib.

Be sure to check out the [Worldline
Tutorial](<https://kurtamohler.github.io/spacetimelib/notebooks/Worldline%20Tutorial.html>)
to see a demonstration of one of the main features of SpacetimeLib, boosting
worldlines.

## Install

Install Miniconda: [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Run the following to create and activate an environment with all dependencies.

```bash
conda env create -f environment.yaml -n spacetime && conda activate spacetime
```

Then install SpacetimeLib.

```bash
pip install -e .
```

## Interactive demo

There is also a small interactive demo. Use the arrow keys to
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

## Build documentation

```bash
python setup.py build_sphinx
```

## Good resources for learning special relativity

* [Sabine Hossenfelder - _Special Relativity: This Is Why You Misunderstand It_](https://youtu.be/ZdrZf4lQTSg)
  - Overview of some of the main ideas of Minkowski spacetime

* [Susskind, L., & Friedman, A. (2017). _Special relativity and classical Field theory: The theoretical minimum_. Basic Books.  ](https://www.amazon.com/Special-Relativity-Classical-Field-Theory/dp/0465093345)

* [Leonard Susskind's video lectures on special relativity](https://www.youtube.com/watch?v=toGH5BdgRZ4&list=PLD9DDFBDC338226CA)
