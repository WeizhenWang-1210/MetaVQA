# Test sidewalk manager that can spawn all types of object on, near, or outside sidewalk
# Please refer to  metadrive.manager.sidewalk_manager for implementation detail
#
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.static_object.test_new_object import TestObject
from metadrive.envs.test_pede_metadrive_env import TestPedeMetaDriveEnv
from metadrive.component.vehicle.vehicle_type import CustomizedCar
def try_pedestrian(render=False):
    env = TestPedeMetaDriveEnv(
        {
            "num_scenarios": 5,
            "traffic_density": 0.5,
            "traffic_mode": "hybrid",
            "start_seed": 1,
            "debug": True,
            # "cull_scene": True,
            "manual_control": True,
            "use_render": render,
            "decision_repeat": 5,
            "need_inverse_traffic": True,
            # "rgb_clip": True,
            "map": 4,
            # "agent_policy": IDMPolicy,
            "random_traffic": True,
            "random_lane_width": True,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "force_destroy": True,
            "window_size": (2400, 1600),
            "vehicle_config": {
                "enable_reverse": True,
            },
        }
    )
    asset_metainfo = {
        "length": 2,
        "width": 2,
        "height": 2,
        "filename": "car-3f699c7ce86c4ba1bad62a350766556f.glb",
        "CLASS_NAME": "06e459171a264e999b3 763335403b719",
        "hshift": 0,
        "pos0": 0,
        "pos1": 0,
        "pos2": 0,
        "scale": 1
    }
    env.reset()
    drawer = env.engine.make_point_drawer(scale=1)
    try:
        # obj_1 = env.engine.spawn_object(TestObject, position=[30, -5], heading_theta=0, random_seed=1, force_spawn=True, asset_metainfo = asset_metainfo)
        for s in range(1, 100000000):
            o, r, tm, tc, info = env.step([0, 0])
            # for obj_id,obj in env.engine.get_objects().items():
            #     if isinstance(obj,CustomizedCar) or isinstance(obj, TestObject):
            #         print(obj.get_asset_metainfo())
            #     else:
            #         print(type(obj))
            ego = env.vehicle
            for key, object in env.engine.get_objects().items():

                close, far =object.origin.getTightBounds()
                width, height, length = object.WIDTH, object.HEIGHT, object.LENGTH
                height = far[2]
                box = [
                    (length / 2, width / 2),
                    (length / 2, -width / 2),
                    (-length / 2, width / 2),
                    (-length / 2, -width / 2),
                ]
                transformed_box = [object.convert_to_world_coordinates(vec, object.position) for vec in box]
                height_extended_box = [(x, y, height) for x, y in transformed_box]
                drawer.draw_points(height_extended_box, [
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                    (1, 1, 0, 1),
                    (1, 1, 0, 1)
                ])

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True

    finally:
        env.close()


if __name__ == "__main__":
    try_pedestrian(True)
