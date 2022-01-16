# This is a basic demonstration of displaying a set of spacetime events in
# different frames and time slices.
#
# This is meant for exploratory purposes, as one step on the path
# towards creating a special relativity simulation.


from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec4, LineSegs, GeomNode, NodePath, Vec3
from direct.directtools.DirectGeometry import LineNodePath
from direct.task import Task
import numpy as np
import lorentz

base = ShowBase()

#base.disableMouse( )
base.camera.setPos(0, 0, 50)
base.camera.lookAt(0, 0, 0)

# A physical object constructed from a set of events
class PhysObject():
    def __init__(self, positions, times, color=(1, 1, 1, 1)):
        self.positions = np.array(positions)
        self.times = np.array(times)

        assert self.positions.ndim == 2 and self.positions.shape[1] == 2
        assert self.times.ndim == 1
        assert self.positions.shape[0] == self.times.shape[0]

        segs = LineSegs()
        segs.setColor(color)
        segs.moveTo(Vec3(*np.concatenate([positions[0], [times[0]]]).tolist()))
        for position, time in zip(self.positions, self.times):
            segs.drawTo(Vec3(*np.concatenate([position, [time]]).tolist()))
        segs.drawTo(Vec3(*np.concatenate([positions[0], [times[0]]]).tolist()))
        self.obj = NodePath(segs.create())

    def transform(self, frame_velocity, light_speed=1, color=(1, 1, 1, 1), time=None):
        frame_velocity = np.array(frame_velocity)

        assert frame_velocity.shape == (2,)

        new_positions, new_times = lorentz.transform_position(self.positions, self.times, frame_velocity, light_speed)

        # If a time is specified, we need to find out the shape of the object
        # as a slice of spacetime at one specific time, assuming that its
        # velocity does not change.
        # NOTE: This would not work as is if we wanted to include velocity
        # changes in the simulation
        if time is not None:
            time_diffs = time - new_times
            time_diffs = np.expand_dims(time_diffs, axis=-1)
            new_positions -= time_diffs * frame_velocity
            new_times.fill(time)

        return PhysObject(new_positions, new_times, color)


def create_square():
    segs = LineSegs()
    segs.setThickness(1)
    segs.setColor((1, 1, 1, 1))
    segs.moveTo((-1, -1, 0))
    segs.drawTo((1, -1, 0))
    segs.drawTo((1, 1, 0))
    segs.drawTo((-1, 1, 0))
    segs.drawTo((-1, -1, 0))
    obj = NodePath(segs.create())
    return obj

#square_obj = create_square()

# We can change the object's position with this
#square_obj.setPos((10, 0, 0))

#base.render.attachNewNode(square_obj.getNode(0))

square_obj = PhysObject(
    [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]],
    [0, 0, 0, 0, 0, 0, 0, 0]) 
base.render.attachNewNode(square_obj.obj.node())

frame_velocity = [.8, .5]
#frame_velocity = [.5, .5]

square_obj_ = square_obj.transform(frame_velocity, color=(1, .5, 0, 1))
square_obj_0 = square_obj.transform(frame_velocity, color=(1, 1, 0, 1), time=0)
base.render.attachNewNode(square_obj_.obj.node())
base.render.attachNewNode(square_obj_0.obj.node())


base.setBackgroundColor(.1, .1, .1)

axes = [
    [(1, 0, 0), (1, 0, 0, 1)],
    [(0, 1, 0), (0, 1, 0, 1)],
    [(0, 0, 1), (0, 0, 1, 1)]]

for axis, color in axes:
    arrow = LineNodePath(colorVec=color)
    arrow.drawArrow(Vec3(*axis), Vec3(0, 0, 0), 0, 0)
    arrow.create()
    base.render.attachNewNode(arrow.node())

base.run()
