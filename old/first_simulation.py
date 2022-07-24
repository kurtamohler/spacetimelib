# This script demos the basic functionality required to create a visual special
# relativity simulation. It simulates constant velocity motion of a few
# particles moving in two spatial dimensions. The paths of particles over time
# are displayed in three dimensions. Particle paths are updated in a fairly
# efficient way, by directly modifying the Panda3D vertex data.
#
# The paths of particles can be viewed from different frames of reference by
# holding down the arrow keys.
#
# The camera can be rotated by right clicking and dragging the mouse. The camera
# can also be zoomed in and out with the mouse wheel.

from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, Vec3, Quat, AntialiasAttrib, GeomVertexFormat, Geom, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode, NodePath, PandaNode, TransparencyAttrib
import numpy as np
from itertools import product
import lorentz

# A ParticleEvent represents one point along a particle's path in space-time
# from the perspective of a particular frame of reference. It contains the
# time and position of the event, as well as the particle's velocity.
class ParticleEvent():
    def __init__(self, time, position, velocity):
        self.time = time
        self.position = position
        self.velocity = velocity

    def lorentz_transform(self, frame_velocity):
        self.position, self.time, self.velocity = lorentz.transform(
            frame_velocity,
            self.position,
            self.time,
            self.velocity)

# A ParticlePath represents the path of a particle through space-time
# from the perspective of a particular frame of reference. It contains
# a Panda3D line segment. It has efficient methods to either update just
# the last vertex of the line segment (when stepping time forward in the
# simulation) or to update all vertices (when transforming to a different
# frame of reference).
class ParticlePath():
    def __init__(self, events, color=(0, 0, 0, 1)):
        segs = LineSegs()
        segs.setThickness(2)
        segs.setColor(color)

        for idx, event in enumerate(events):
            path_pos = (event.time, event.position[0], event.position[1])
            if idx == 0:
                segs.moveTo(path_pos)
            else:
                segs.drawTo(path_pos)

        self.geom_node = segs.create()


    # Create a memoryview of the geometry data of the path
    def get_path_data(self):
        geom = self.geom_node.modifyGeom(0)
        v_data = geom.modifyVertexData()
        v_array = v_data.modify_array(0)
        view = memoryview(v_array).cast('B').cast('f')
        return np.asarray(view, dtype=np.float32)

    # Update the position and time of the last vertex of the path
    def update_last(self, time, position):
        path_data = self.get_path_data()
        path_data[-4] = time
        path_data[-3] = position[0]
        path_data[-2] = position[1]

    # Update the positions and times of all vertices of the path
    def update_all(self, times, positions):
        path_data = self.get_path_data()

        for idx, (time, position) in enumerate(zip(times, positions)):
            path_data[4*idx] = time
            path_data[4*idx + 1] = position[0]
            path_data[4*idx + 2] = position[1]

# Particle represents a particle moving through space-time. It is comprised
# of a series of ParticleEvents which describes its motion. It also contains
# a ParticlePath that is used to display the particle's path in Panda3D.
class Particle():
    def __init__(self, time, position, velocity, color=(0, 0, 0, 1)):
        time = np.array(time, dtype=np.float64)
        position = np.array(position, dtype=np.float64)
        velocity = np.array(velocity, dtype=np.float64)

        assert time.shape == ()
        assert position.shape == (2,)
        assert velocity.shape == (2,)

        self.events = [
            ParticleEvent(time, position, velocity),
            ParticleEvent(time, position, velocity)]

        self.path = ParticlePath(self.events, color=color)

    def step(self, time_delta):
        cur_event = self.events[-1]
        cur_event.time += time_delta
        cur_event.position += time_delta * cur_event.velocity
        self.path.update_last(cur_event.time, cur_event.position)

    def lorentz_transform(self, frame_velocity):
        for event in self.events:
            event.lorentz_transform(frame_velocity)

        self.path.update_all(
            [event.time for event in self.events],
            [event.position for event in self.events])

# SimulationState represents the state of a region of space-time in
# a particular frame of reference. It includes the paths of particles moving
# within the region of space-time.
class SimulationState():
    def __init__(self):
        self.particles = []
        # TODO: Need to create empty geom node here, so that we can shift the whole thing down along the time axis as the simulation progresses

    def step(self, time_step):
        for particle in self.particles:
            particle.step(time_step)

    def lorentz_transform(self, frame_velocity):
        for particle in self.particles:
            particle.lorentz_transform(frame_velocity)

def draw_line(p0, p1, color):
    segs = LineSegs()
    segs.setThickness(1)
    segs.setColor(color)
    segs.moveTo(p0)
    segs.drawTo(p1)
    return segs.create()

def draw_grid(x0=-1000, x1=1000, y0=-1000, y1=1000, z0=0, z1=0, interval=100, color=(.2, .2, .2, 1), skip_x=None, skip_y=None, skip_z=None):
    if x0 > x1:
        _ = x0
        x0 = x1
        x1 = _

    if y0 > y1:
        _ = y0
        y0 = y1
        y1 = _

    if z0 > z1:
        _ = z0
        z0 = z1
        z1 = _

    node = NodePath('grid')

    if x0 != x1:
        for y in range(y0, y1 + 1, interval):
            if skip_y and (y % skip_y) == 0:
                continue
            for z in range(z0, z1 + 1, interval):
                if skip_z and (z % skip_z) == 0:
                    continue
                node.attachNewNode(draw_line(
                    (x0, y, z),
                    (x1, y, z),
                    color))

    if y0 != y1:
        for x in range(x0, x1 + 1, interval):
            if skip_x and (x % skip_x) == 0:
                continue
            for z in range(z0, z1 + 1, interval):
                if skip_z and (z % skip_z) == 0:
                    continue
                node.attachNewNode(draw_line(
                    (x, y0, z),
                    (x, y1, z),
                    color))

    if z0 != z1:
        for x in range(x0, x1 + 1, interval):
            if skip_x and (x % skip_x) == 0:
                continue
            for y in range(y0, y1 + 1, interval):
                if skip_y and (y % skip_y) == 0:
                    continue
                node.attachNewNode(draw_line(
                    (x, y, z0),
                    (x, y, z1),
                    color))

    return node


# TODO: Add some coordinate axes in the corner of the camera, labelled with
# time, x, and y.

# TODO: There should be a second 2-D timeslice of the current latest simulation
# time in the current frame of reference. This will let us effectively
# understand how things appear to move in just space when changing to different
# frames of reference.

# TODO: It might be a good idea to perform calculations all in the base
# reference frame, then perform lorentz transform for view purposes? That
# would fix the existing problem of float errors coming in from repeatedly
# performing transforms on top of transforms. However, maybe it would
# be too resource intensive to perform so many transforms. Perhaps
# there's a more efficient way to reduce the float errors? Or maybe it's not
# even too intensive after all--with 4 particle paths, it doesn't seem so bad.
# Actually, it is quit intensive, it seems. I will need to optimize the transform
# calculation though, since I want to be able to accelerate in real time

class App(ShowBase):
    def __init__(self):
        super().__init__()

        self.setFrameRateMeter(True)
        self.setBackgroundColor(.95, .95, 1)

        self.disableMouse()

        self.camera.setPos(0, -500, 0)

        self.sim = SimulationState()

        self.sim.particles.append(
            Particle(0, (0, 0), (0, 0), color=(0, 0, 0, 0)))

        self.sim.particles.append(
            Particle(0, (0, 10), (.6, .6), color=(0, 0.8, 0, 0)))

        self.sim.particles.append(
            Particle(0, (0, 10), (.6, -.6), color=(0.8, 0, 0, 0)))

        self.sim.particles.append(
            Particle(0, (20, 0), (-.3, 0), color=(0, 0, 0.8, 0)))

        self.prev_time = 0

        self.time_scale = 10

        for particle in self.sim.particles:
            render.attachNewNode(particle.path.geom_node)

        self.grids = [
            # 10 light-second grid
            draw_grid(
                x0=-1000, x1=1000,
                y0=-1000, y1=1000,
                z0=0, z1=0,
                interval=10,
                color=(0, 0, 0, 1),
                skip_x=100,
                skip_y=100),

            # 100 light-second grid
            draw_grid(
                x0=-10_000, x1=10_000,
                y0=-10_000, y1=10_000,
                z0=0, z1=0,
                interval=100,
                color=(0, 0, 0, 1))
        ]

        self.grids[0].setTransparency(TransparencyAttrib.MAlpha)
        self.grids[0].setAlphaScale(.2)
        self.grids[0].reparentTo(render)

        self.grids[1].setTransparency(TransparencyAttrib.MAlpha)
        self.grids[1].setAlphaScale(.5)
        self.grids[1].reparentTo(render)

        def gen_xy_plane():
            width = 0.2
            x = 100
            y = 100
            fmt = GeomVertexFormat.getV3()
            data = GeomVertexData("Data", fmt, Geom.UHStatic)
            vertices = GeomVertexWriter(data, "vertex")
            vertices.addData3d(-width, -x, -y)
            vertices.addData3d( width, -x, -y)
            vertices.addData3d(-width,  x, -y)
            vertices.addData3d( width,  x, -y)
            vertices.addData3d(-width, -x,  y)
            vertices.addData3d( width, -x,  y)
            vertices.addData3d(-width,  x,  y)
            vertices.addData3d( width,  x,  y)
            triangles = GeomTriangles(Geom.UHStatic)

            def addQuad(v0, v1, v2, v3):
                triangles.addVertices(v0, v1, v2)
                triangles.addVertices(v0, v2, v3)
                triangles.closePrimitive()

            addQuad(4, 5, 7, 6) # Z+
            addQuad(0, 2, 3, 1) # Z-
            addQuad(3, 7, 5, 1) # X+
            addQuad(4, 6, 2, 0) # X-
            addQuad(2, 6, 7, 3) # Y+
            addQuad(0, 1, 5, 4) # Y+

            geom = Geom(data)
            geom.addPrimitive(triangles)
            
            node = GeomNode("xy-plane")
            node.addGeom(geom)

            return NodePath(node)


        xy_plane = gen_xy_plane()
        xy_plane.setColor(.9, .9, .9, 1)
        xy_plane.reparentTo(render)

        self.taskMgr.add(self.step_task)

        self.accept("wheel_up", self.zoom_in)
        self.accept("wheel_down", self.zoom_out)
        self.zoom_out_factor = 1.1
        self.zoom_in_factor = 1 / self.zoom_out_factor

        self.accept('mouse3', self.start_view_rotation)
        self.accept('mouse3-up', self.stop_view_rotation)

        self.view_rotation_on = False
        self.last_mouse_x = None
        self.last_mouse_y = None

        self.camera_rotation_factor = 360
        self.camera_pitch_factor = 180

        self.taskMgr.add(self.transform_task)
        self.transform_direction = None
        self.transform_speed = 0.001

        self.accept('arrow_up', self.set_transform_up)
        self.accept('arrow_down', self.set_transform_down)
        self.accept('arrow_right', self.set_transform_right)
        self.accept('arrow_left', self.set_transform_left)

        self.accept('arrow_up-up', self.unset_transform)
        self.accept('arrow_down-up', self.unset_transform)
        self.accept('arrow_right-up', self.unset_transform)
        self.accept('arrow_left-up', self.unset_transform)

        # TODO: For some reason, the camera seems to zoom out the first time
        # the lorentz_transform is used. So we do one right at the beginning
        # to work around it, but it should be fixed.
        self.sim.lorentz_transform([0, 0.001])
        self.sim.lorentz_transform([0, -0.001])

    def set_transform_up(self):
        self.transform_direction = [0, self.transform_speed]

    def set_transform_down(self):
        self.transform_direction = [0, -self.transform_speed]

    def set_transform_right(self):
        self.transform_direction = [self.transform_speed, 0]

    def set_transform_left(self):
        self.transform_direction = [-self.transform_speed, 0]

    def unset_transform(self):
        self.transform_direction = None


    def transform_task(self, task):
        if self.transform_direction is not None:
            self.sim.lorentz_transform(self.transform_direction)

        return task.cont


    def start_view_rotation(self):
        if self.mouseWatcherNode.hasMouse():
            self.taskMgr.add(self.view_rotation_task)
            self.view_rotation_on = True
            self.last_mouse_x = self.mouseWatcherNode.getMouseX()
            self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def stop_view_rotation(self):
        self.view_rotation_on = False

    def view_rotation_task(self, task):
        if self.view_rotation_on:
            if self.mouseWatcherNode.hasMouse():
                x = self.mouseWatcherNode.getMouseX()
                y = self.mouseWatcherNode.getMouseY()

                x_diff = x - self.last_mouse_x
                y_diff = y - self.last_mouse_y

                self.last_mouse_x = x
                self.last_mouse_y = y

                if x_diff != 0:
                    axis = Vec3(0, 0, 1)
                    angle = -self.camera_rotation_factor * x_diff
                    quat = Quat()
                    quat.setFromAxisAngle(angle, axis)
                    new_cam_pos = quat.xform(self.camera.getPos())
                    self.camera.setPos(*new_cam_pos)
                    self.camera.lookAt((0, 0, 0))

                if y_diff != 0:
                    camera_direction = self.camera.getPos()
                    camera_direction.normalize()
                    axis = camera_direction.cross(Vec3(0, 0, 1))
                    axis.normalize()

                    angle = -self.camera_pitch_factor * y_diff
                    quat = Quat()
                    quat.setFromAxisAngle(angle, axis)

                    # Limit the vertical camera rotation so that it cannot
                    # rotate across the vertical axis
                    # TODO: This could probably be done in a simpler way
                    cur_angle_ratio = camera_direction.dot(Vec3(0, 0, 1))
                    camera_direction_horiz = Vec3(
                        camera_direction[0],
                        camera_direction[1],
                        0)
                    camera_direction_horiz.normalize()
                    rotation_angle_ratio = quat.xform(camera_direction_horiz).dot(Vec3(0, 0, 1))
                    ratio_after_rotation = rotation_angle_ratio + cur_angle_ratio

                    if ratio_after_rotation < 1 and ratio_after_rotation > -1:
                        new_cam_pos = quat.xform(self.camera.getPos())
                        self.camera.setPos(*new_cam_pos)
                        self.camera.lookAt((0, 0, 0))

                    
            self.adjust_grid_transparency()
            return task.cont


    def step_task(self, task):
        time_delta = self.time_scale * (task.time - self.prev_time)
        self.sim.step(time_delta)
        self.prev_time = task.time
        return task.cont

    def zoom_in(self):
        cam_pos = self.camera.getPos()
        self.camera.setPos(*(cam_pos * self.zoom_in_factor))
        self.adjust_grid_transparency()

    def zoom_out(self):
        cam_pos = self.camera.getPos()
        self.camera.setPos(*(cam_pos * self.zoom_out_factor))
        self.adjust_grid_transparency()

    def adjust_grid_transparency(self):
        cam_pos = self.camera.getPos()
        def adjust_by_height():
            z_abs = abs(cam_pos[2])
            opaque_z = 100
            transparent_z = 2000

            if z_abs < opaque_z:
                self.grids[0].setAlphaScale(.5)
            elif z_abs > transparent_z:
                self.grids[0].setAlphaScale(0)
            else:
                self.grids[0].setAlphaScale(
                    .5 * (transparent_z - z_abs) / (transparent_z - opaque_z))

        def adjust_by_dist2():
            dist2 = cam_pos[0] ** 2 + cam_pos[1] ** 2 + cam_pos[2] ** 2
            opaque_dist2 = 10 ** 2
            transparent_dist2 = 2000 ** 2

            if dist2 < opaque_dist2:
                self.grids[0].setAlphaScale(.5)
            elif dist2 > transparent_dist2:
                self.grids[0].setAlphaScale(0)
            else:
                self.grids[0].setAlphaScale(
                    .5 * (transparent_dist2 - dist2) / (transparent_dist2 - opaque_dist2))

        def adjust_by_dist():
            dist = (cam_pos[0] ** 2 + cam_pos[1] ** 2 + cam_pos[2] ** 2) ** 0.5
            opaque_dist = 1
            transparent_dist = 2000

            if dist < opaque_dist:
                self.grids[0].setAlphaScale(.5)
            elif dist > transparent_dist:
                self.grids[0].setAlphaScale(0)
            else:
                self.grids[0].setAlphaScale(
                    .5 * (transparent_dist - dist) / (transparent_dist - opaque_dist))

        adjust_by_dist2()
        adjust_by_dist()

    def mouse_task(self, task):
        if self.mouseWatcherNode.hasMouse():
            x = self.mouseWatcherNode.getMouseX()
            y = self.mouseWatcherNode.getMouseY()

        return task.cont


app = App()
app.run()
