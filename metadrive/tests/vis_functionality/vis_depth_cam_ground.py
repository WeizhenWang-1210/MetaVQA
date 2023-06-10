from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_module.depth_camera import DepthCamera
from metadrive.component.vehicle_module.vehicle_panel import VehiclePanel
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

if __name__ == "__main__":

    def get_image(env):
        depth_cam = env.vehicle.get_camera(env.vehicle.config["image_source"])
        rgb_cam = env.vehicle.get_camera("rgb_camera")
        for h in range(-180, 180, 20):
            env.engine.graphicsEngine.renderFrame()
            depth_cam.get_cam().setH(h)
            rgb_cam.get_cam().setH(h)
            depth_cam.save_image(env.vehicle, "depth_{}.jpg".format(h))
            rgb_cam.save_image(env.vehicle, "rgb_{}.jpg".format(h))
        # env.engine.screenshot()

    env = SafeMetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "accident_prob": 1.,
            "start_seed": 4,
            "map": "SSSSS",
            "manual_control": True,
            "use_render": True,
            "image_observation": True,
            "rgb_clip": True,
            "interface_panel": [DepthCamera, VehiclePanel],
            "vehicle_config": dict(depth_camera=(800, 600, True), rgb_camera=(800, 600), image_source="depth_camera"),
            # "map_config": {
            #     BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            #     BaseMap.GENERATE_CONFIG: 12,
            #     BaseMap.LANE_WIDTH: 3.5,
            #     BaseMap.LANE_NUM: 3,
            # }
        }
    )
    env.reset()
    env.engine.accept("m", get_image, extraArgs=[env])

    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        if env.config["use_render"]:
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render()
        # if d:
        #     # print("Reset")
        #     env.reset()
    env.close()
