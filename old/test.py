from pandac.PandaModules import Vec4
from pandac.PandaModules import LineSegs
import direct.directbase.DirectStart
from panda3d.core import GeomNode, NodePath
import math
cos=math.cos
sin=math.sin
pi=math.pi

base.disableMouse( )
base.camera.setPos( 0, 0, 50)
base.camera.lookAt( 0, 0, 0 )
#base.enableMouse()

segs = LineSegs()
segs.setThickness(1)
segs.setColor((1, 1, 1, 1))
segs.moveTo((-1, -1, 0))
segs.drawTo((1, -1, 0))
segs.drawTo((1, 1, 0))
segs.drawTo((-1, 1, 0))
segs.drawTo((-1, -1, 0))
obj = NodePath(segs.create())

render.attachNewNode(obj.getNode(0))

# We can change the object's position with this
obj.setPos((10, 0, 0))

base.setBackgroundColor(0, 0, 0)
base.run()
