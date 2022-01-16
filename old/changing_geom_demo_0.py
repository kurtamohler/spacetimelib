# This file is an exploration of modifying line segment geometry in Panda3D.
#
# In the special relativity simulation, we will need to be able to modify the
# geometry of the paths of particles. We will have all the path data for
# a particle in a NumPy NDArray, represented as a list of events within one
# reference frame. There are three different operations we'll want to perform
# on this path:
#
#   * Edit the most recent event to step it forward in time according to the
#     particle's current velocity.
#
#   * Append a new event to the list any time the velocity of the particle
#     changes.
#
#   * Lorentz transform all of the events in a particle's path when we change
#     to a different reference frame.
#
# Any time we make one of these changes to a particle's path, we also need to
# update the Panda3D representation of the path. A naive way to do this would
# be to remove the geometry from the scene graph, generate a new one, and add
# it back to the scene graph. But this would be inefficient. It would be best
# if we could just modify the geometry in place. Thankfully, Panda3D has
# a concept of memoryviews, where we can view the geometry data through an
# NDArray:
# https://docs.panda3d.org/1.10/python/programming/internal-structures/other-manipulation/using-memoryviews
#
# Memoryviews in Panda3D provide an optimal way to both modify existing
# vertices and to add new vertices, so we will use it to solve our problem.

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, GeomVertexReader, GeomVertexWriter, GeomVertexFormat, GeomVertexData, GeomEnums, Geom
from direct.task import Task
import array
import time

class App(ShowBase):
    def __init__(self):
        super().__init__()
        self.setFrameRateMeter(True)
        self.setBackgroundColor(.1, .1, .1)
        
        # Move the camera back so we can see the line segment
        self.cam.setPos(0, -4, 0)

        # Create a line segment with 3 vertices
        segs = LineSegs()
        segs.setThickness(2)
        segs.setColor((1, 1, 0, 1))
        segs.moveTo((-0.5, 0, 0))
        segs.setColor((1, 0, 1, 1))
        segs.drawTo((0, 0, 0))
        segs.setColor((0, 1, 1, 1))
        segs.drawTo((0.5, 0, 0))
        self.path = segs.create()

        self.render.attachNewNode(self.path)
        self.grow_count = 0

        self.taskMgr.add(self.vibrate_task, 'Vibrate')

        self.taskMgr.add(self.grow_task, 'Grow')
        #self.taskMgr.add(self.slide_camera, 'SlideCamera')


        self.grow()

    def slide_camera(self, task):
        self.cam.setPos(task.time / 4, -4, 0)
        return task.cont

    # This method of modifying vertex data is inefficient. It sequentially
    # reads each vertex in the path and completely rewrites each one into the
    # geometry again. Since we're only modifying one vertex, all the extra
    # reads and writes are suboptimal.
    def set_vertex_inefficient(self, vertex_idx, position):
        geom = self.path.modifyGeom(0)
        v_data = geom.modifyVertexData()
        reader = GeomVertexReader(v_data, 'vertex')
        writer = GeomVertexWriter(v_data, 'vertex', reader.getCurrentThread())
        cur_idx = 0
        while not reader.isAtEnd():
            v = reader.getData3()
            if cur_idx == vertex_idx:
                writer.setData3(position[0], position[1], position[2])
            else:
                writer.setData3(v[0], v[1], v[2])
            cur_idx += 1

    # This method of modifying vertex data is much more efficent. We view
    # the geometry data of the path as a NumPy NDArray and modify only the
    # values corresponding to the vertex that needs to be changed.
    def set_vertex_efficient(self, vertex_idx, position):
        def get_path_data(path):
            geom = path.modifyGeom(0)
            v_data = geom.modifyVertexData()
            v_array = v_data.modify_array(0)
            view = memoryview(v_array).cast('B').cast('f')

            # TODO: For some reason, we cannot cache path_data for later use.
            # I tried it, so that I would only have to initialize it once, but
            # writing to it repeatedly had no effect on the displayed geometry.
            # I wonder if there's some kind of redraw process that only gets
            # applied if we recreate the memory view from the geometry each
            # time.

            # TODO: If I try to reshape path_data to 2-D, so that it represents
            # a list of vertices, np.reshape returns a copy of the data, rather
            # than a view of the original. I'm not sure why. I could try
            # PyTorch instead, though I would much prefer to somehow get it
            # working with NumPy.

            path_data = np.asarray(view, dtype=np.float32)
            return path_data

        path_data = get_path_data(self.path)
        path_data[4 * vertex_idx + 0] = position[0]
        path_data[4 * vertex_idx + 1] = position[1]
        path_data[4 * vertex_idx + 2] = position[2]

    def vibrate_task(self, task):
        freq = 2
        offset_z = 0.4 * np.sin(freq * (2 * np.pi) * task.time)
        offset_y = 0.4 * np.cos(freq * (2 * np.pi) * task.time)

        #self.set_vertex_inefficient(1, (0, offset_y, offset_z))
        self.set_vertex_efficient(1, (0, offset_y, offset_z))

        return task.cont

    def append_vertex(self, vertex):
        geom = self.path.modifyGeom(0)
        v_data = geom.modifyVertexData()

        old_count = v_data.get_num_rows()
        v_data.setNumRows(1 + old_count)

        
        v_array = v_data.modify_array(0)
        view = memoryview(v_array).cast('B').cast('f')

        view[-4] = vertex[0]
        view[-3] = vertex[1]
        view[-2] = vertex[2]

        # TODO: Right now, this just reuses the previous vertex's color,
        # but it would be better to allow it to be set to anything
        view[-1] = view[-5]

        prim = geom.modifyPrimitive(0)

        # TODO: I hope there's a way to avoid having to clear
        # and read all the vertex indices to the primitive
        prim.clearVertices()
        for vertex_idx in range(1 + old_count):
            prim.addVertex(vertex_idx)
        prim.closePrimitive()

        #print(prim.modifyVertices())
        #vertices = prim.modifyVertices()
        #vertices.set_
        #print(type(vertices))

    def grow(self):
        self.grow_count += 1

        radius = 0.4 * np.sin(self.grow_count / 50)
        
        vertex = [
            self.grow_count * 0.008,
            radius * np.sin(self.grow_count * np.pi / 2),
            radius * np.cos(self.grow_count * np.pi / 2),
        ]

        vertex[0] += 0.5
        self.append_vertex(vertex)

    def grow_task(self, task):
        if (task.time * 30) > self.grow_count:
            self.grow()
        return task.cont

app = App()
app.run()
