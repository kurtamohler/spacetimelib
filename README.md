# Special relativity simulation

Very early stages of a tool that can simulate motion in relativistic spacetime.

## Install

Install Miniconda: [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Run the following to create and activate an environment with all dependencies.

```bash
conda env create -f environment.yaml -n relativity && conda activate relativity
```

## Run

There is just one small interactive demo at the moment. Use the arrow keys to
move around.

```bash
python sim/sim.py
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
