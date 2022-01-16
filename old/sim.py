from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.task.Task import Task
import panda3d
from panda3d.core import TextNode, LineSegs, LVecBase3f, NodePath

import numpy as np

def np2vec3(np_vec):
    return LVecBase3f(np_vec[0], np_vec[1], np_vec[2])

class Particle:
    def __init__(self):
        self.pos = np.array([0., 0., 0.])
        self.vel = np.array([0., 0., 0.])
        #self.obj = LineSegs('particle')

        '''
        self.obj.setColor((1, 1, 1, 1))
        self.obj.setThickness(10)
        self.obj.moveTo(np2vec3(self.pos + (-0.1, -0.1, 0)))
        self.obj.drawTo(np2vec3(self.pos + (-0.1, 0.1, 0)))
        self.obj.drawTo(np2vec3(self.pos + (0.1, 0.1, 0)))
        self.obj.drawTo(np2vec3(self.pos + (0.1, -0.1, 0)))
        self.obj.drawTo(np2vec3(self.pos + (-0.1, -0.1, 0)))
        self.obj = NodePath(self.obj.create())
        '''
        ls = LineSegs()
        ls.setThickness(100)

        # X axis
        ls.setColor(1.0, 0.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(1.0, 0.0, 0.0)

        # Y axis
        ls.setColor(0.0, 1.0, 0.0, 1.0)
        ls.moveTo(0.0,0.0,0.0)
        ls.drawTo(0.0, 1.0, 0.0)

        # Z axis
        ls.setColor(0.0, 0.0, 1.0, 1.0)
        ls.moveTo(0.0,0.0,0.0)
        ls.drawTo(0.0, 0.0, 1.0)
        #self.obj = NodePath(ls.create())
        self.obj = ls.create()

    def draw(self):
        pass
        #self.obj.setPos(np2vec3(self.pos))

class View(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor((0, 0, 0, 1))
        self.gameTask = taskMgr.add(self.gameLoop, "gameLoop")

        self.particle = Particle()
        #self.particle.obj.reparentTo(self.render)

        #self.particle.obj.setScale(0.25, 0.25, 0.25)
        #self.particle.obj.setPos(0, 0, 0)

        #self.geom = Geom(self.particle.obj)

        self.scene = self.loader.loadModel('models/environment')
        #self.scene.reparentTo(self.render)
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        #self.particle.obj.reparentTo(self.render)
        self.render.attachNewNode(self.particle.obj)

        

    def gameLoop(self, task):
        dt = globalClock.getDt()

        self.particle.pos += self.particle.vel * dt
        self.particle.draw()



        return Task.cont

view = View()
view.run()
