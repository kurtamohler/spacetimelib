from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, Vec3, Quat, AntialiasAttrib, GeomVertexFormat, Geom, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode, NodePath
import numpy as np
from itertools import product
import lorentz

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


    def get_path_data(self):
        geom = self.geom_node.modifyGeom(0)
        v_data = geom.modifyVertexData()
        v_array = v_data.modify_array(0)
        view = memoryview(v_array).cast('B').cast('f')
        return np.asarray(view, dtype=np.float32)

    def update_last(self, time, position):
        path_data = self.get_path_data()
        path_data[-4] = time
        path_data[-3] = position[0]
        path_data[-2] = position[1]

    def update_all(self, times, positions):
        path_data = self.get_path_data()

        for idx, (time, position) in enumerate(zip(times, positions)):
            path_data[4*idx] = time
            path_data[4*idx + 1] = position[0]
            path_data[4*idx + 2] = position[1]

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

        for x, y in product(range(-100, 101, 50), range(-100, 101, 50)):
            if x != 0 or y != 0:
                self.sim.particles.append(
                    Particle(0, (x, y), (0, 0), color=(0.5, 0.5, 0.5, 0)))


        self.prev_time = 0

        self.time_scale = 10

        for particle in self.sim.particles:
            render.attachNewNode(particle.path.geom_node)

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

        #render.setAntialias(AntialiasAttrib.MMultisample)
        #render.setAntialias(AntialiasAttrib.MPolygon)

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
        self.transform_speed = 0.01

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

                    
            return task.cont


    def step_task(self, task):
        time_delta = self.time_scale * (task.time - self.prev_time)
        self.sim.step(time_delta)
        self.prev_time = task.time
        return task.cont

    def zoom_in(self):
        cam_pos = self.camera.getPos()
        self.camera.setPos(*(cam_pos * self.zoom_in_factor))

    def zoom_out(self):
        cam_pos = self.camera.getPos()
        self.camera.setPos(*(cam_pos * self.zoom_out_factor))

    def mouse_task(self, task):
        if self.mouseWatcherNode.hasMouse():
            x = self.mouseWatcherNode.getMouseX()
            y = self.mouseWatcherNode.getMouseY()
            print(f'{x}, {y}')

        return task.cont


app = App()
app.run()
