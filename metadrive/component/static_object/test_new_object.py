import logging
from typing import Tuple

from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.core import Material, Vec3, TransformState

from metadrive.component.static_object.base_static_object import BaseStaticObject
from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.constants import CollisionGroup
from metadrive.constants import MetaDriveType
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import get_engine, engine_initialized
from metadrive.engine.physics_node import BaseRigidBodyNode


LaneIndex = Tuple[str, str, int]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestObject(TrafficObject):
    """A barrier"""

    # HEIGHT = 2.0
    # MASS = 10
    CLASS_NAME = "TestObject"

    def __init__(self, asset_metainfo, position, heading_theta, lane=None, static: bool = False, random_seed=None, name=None):
        super(TestObject, self).__init__(position, heading_theta, lane, random_seed, name)
        self.asset_metainfo = asset_metainfo
        if "general" in asset_metainfo.keys():
            self._length = asset_metainfo["general"]["length"]
            self._width = asset_metainfo["general"]["width"]
        else:
            self._length = asset_metainfo["length"]
            self._width = asset_metainfo["width"]

        # self._length = 2
        # self._width = 2
        self._height = asset_metainfo["height"]
        self.filename = asset_metainfo["filename"]
        self.hshift = asset_metainfo["hshift"]
        self.pos0 = asset_metainfo["pos0"]
        self.pos1 = asset_metainfo["pos1"]
        self.pos2 = asset_metainfo["pos2"]
        self.scale = asset_metainfo["scale"]

        n = self._create_obj_chassis()
        self.add_body(n)

        # self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.height / 2)))
        self.body.addShape(BulletBoxShape((self.LENGTH / 2, self.WIDTH / 2, self.height / 2)))

        # self.set_static(static)
        if self.render:
            # model_file_path1 = AssetLoader.file_path("models", "test", "stop sign-8be31e33b3df4d6db7c75730ff11dfd8.glb")
            model_file_path2 = AssetLoader.file_path("models", "test", self.filename)
            model = self.loader.loadModel(model_file_path2)
            model.setH(self.hshift)
            model.setPos(self.pos0, self.pos1, self.pos2)
            model.setScale(self.scale)
            model.reparentTo(self.origin)

    def _create_obj_chassis(self):

        chassis = BaseRigidBodyNode(self.name, self.asset_metainfo['CLASS_NAME'])
        self._node_path_list.append(chassis)

        chassis_shape = BulletBoxShape(Vec3(self.LENGTH / 2, self.WIDTH / 2, self.HEIGHT / 2))
        ts = TransformState.makePos(Vec3(0, 0, self.HEIGHT / 2))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        # physics_world = get_engine().physics_world
        # vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        # vehicle_chassis.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(chassis)
        return chassis
    def get_asset_metainfo(self):
        return self.asset_metainfo
    @property
    def LENGTH(self):
        return self._length

    @property
    def WIDTH(self):
        return self._width

    @property
    def HEIGHT(self):
        return self._height

    @property
    def top_down_length(self):
        # reverse the direction
        return self.WIDTH * 2

    @property
    def top_down_width(self):
        # reverse the direction
        return self.LENGTH




class TestGLTFObject(TrafficObject):
    """A barrier"""

    # HEIGHT = 2.0
    # MASS = 10
    CLASS_NAME = "TestObject"

    def __init__(self, asset_metainfo, position, heading_theta, lane=None, static: bool = False, random_seed=None, name=None):
        super(TestGLTFObject, self).__init__(position, heading_theta, lane, random_seed, name)
        self.asset_metainfo = asset_metainfo
        n = BaseRigidBodyNode(self.name, self.asset_metainfo['CLASS_NAME'])
        self.add_body(n)
        self._length = asset_metainfo["length"]
        self._width = asset_metainfo["width"]
        self._height = asset_metainfo["height"]
        self.foldername = asset_metainfo["foldername"]
        self.filename = asset_metainfo["filename"]
        self.hshift = asset_metainfo["hshift"]
        self.pos0 = asset_metainfo["pos0"]
        self.pos1 = asset_metainfo["pos1"]
        self.pos2 = asset_metainfo["pos2"]
        self.scale = asset_metainfo["scale"]

        # self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.height / 2)))
        self.body.addShape(BulletBoxShape((self.LENGTH / 2, self.WIDTH / 2, self.height / 2)))
        self.set_static(static)
        if self.render:
            # model_file_path1 = AssetLoader.file_path("models", "test", "stop sign-8be31e33b3df4d6db7c75730ff11dfd8.glb")
            model_file_path2 = AssetLoader.file_path("models", "test", self.foldername, self.filename)
            model = self.loader.loadModel(model_file_path2)
            model.setH(self.hshift)
            model.setPos(self.pos0, self.pos1, self.pos2)
            model.setScale(self.scale)
            model.reparentTo(self.origin)
    @property
    def LENGTH(self):
        return self._length

    @property
    def WIDTH(self):
        return self._width

    @property
    def HEIGHT(self):
        return self._height

    @property
    def top_down_length(self):
        # reverse the direction
        return self.WIDTH * 2

    @property
    def top_down_width(self):
        # reverse the direction
        return self.LENGTH