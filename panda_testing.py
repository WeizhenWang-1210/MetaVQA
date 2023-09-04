from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase, Loader
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3,NodePath
import re
from panda3d.core import CollisionTraverser, CollisionNode, CollisionPolygon, BitMask32,CollisionHandlerQueue
from panda3d.bullet import BulletConvexHullShape,BulletBodyNode
from panda3d.core import LPoint3f,PandaNode
from metadrive.engine.asset_loader import AssetLoader





class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Disable the camera trackball controls.
        self.disableMouse()

        # Load the environment model

        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        #self.taskMgr.add(self.findObservableNodes, "check observed")
        


        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model",
                                {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)
        #AssetLoader.init_loader()
        #loadeer = AssetLoader.get_loader()
        self.pandaActor2 = Actor("models/panda-model",
                                {"walk": "models/panda-walk4"})
        #self.taskMgr.add(self.checkFrustum,"PandaVisible")
        self.pandaActor2.setScale(0.01, 0.01, 0.01)
        self.pandaActor2.reparentTo(self.render)
        self.pandaActor2.setPos(self.render, 10,10,0)




        # Loop its animation.
        #self.pandaActor.loop("walk")
        #self.pandaActor2.loop("walk")

        # Create the four lerp intervals needed for the panda to
        # walk back and forth.
        posInterval1 = self.pandaActor.posInterval(13,
                                                   Point3(0, -10, 0),
                                                   startPos=Point3(0, 10, 0))
        posInterval2 = self.pandaActor.posInterval(13,
                                                   Point3(0, 10, 0),
                                                   startPos=Point3(0, -10, 0))
        hprInterval1 = self.pandaActor.hprInterval(3,
                                                   Point3(180, 0, 0),
                                                   startHpr=Point3(0, 0, 0))
        hprInterval2 = self.pandaActor.hprInterval(3,
                                                   Point3(0, 0, 0),
                                                   startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        self.pandaPace = Sequence(posInterval1, hprInterval1,
                                  posInterval2, hprInterval2,
                                  name="pandaPace")
        #self.pandaPace.loop()

    def findObservableNodes(self,camera):
        observable_nodes = []
        for node in self.render.find_all_matches("**/panda*"):
            print(self.cam.getPos(self.render))
            print(node.getPos(self.cam))
            mnode = NodePath("Test")
            mnode.reparentTo(self.render)
            mnode.setPos(0,0,1.5)
            print(self.camNode.isInView(mnode.getPos(self.cam)))
            """ if not self.camNode.isInView(node.getPos(self.cam)):
                observable_nodes.append(node)
            print(self.camNode.isInView(LPoint3f(0,2,0)))"""
        #print(observable_nodes)
        return Task.cont
    
    def pandaVisible(self,camera):
        panda_node = self.pandaActor.getGeomNode()
        for i in range(panda_node.node().getNumGeoms()):
            geom = panda_node.getGeom(i)
            vdata = geom.getVertexData()
            print(vdata)
        return Task.cont

    def checkFrustum(self,camera):
        
        """FOV, CLOSE, FAR = lens.getFov(),lens.getNear(), lens.getFar()
        print(FOV, CLOSE, FAR)"""
        self.traverser.traverse(self.render)
        collisions = self.collision_handler.get_entries()
        for entry in collisions:
            print(entry.get_into_node_path().get_parent())
        

        return Task.cont
    
    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        radius = 40
        self.camera.setPos(radius * sin(angleRadians), -radius * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


app = MyApp()
app.run()






