# Test Pedestrain manager that can generate static pedestrian on sidewalk
# Please refer to  metadrive.manager.sidewalk_manager for implementation detail
# !!!!!!!!!!!!You need to change asset used in  metadrive.manager.sidewalk_manager
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.static_object.test_new_object import TestObject
from metadrive.envs.test_pede_metadrive_env import TestPedeMetaDriveEnv
from metadrive.component.vehicle.vehicle_type import CustomizedCar
from metadrive.utils.utils import get_object_from_node
from utils_testing import sample_bbox
from panda3d.core import LPoint3f, NodePath
import yaml
import random
import os
import pygame
import cv2
import json

def try_pedestrian(render=False):
    try:
        with open('scene_generation_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        print("Couldn't load config for dataset generation! Exiting")
        exit(1)
    scene_config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=config["map_setting"]["traffic_density"],
        num_scenarios=config["map_setting"]["num_scenarios"],
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        window_size = (1920,1080),
        need_inverse_traffic = config["map_setting"]["inverse_traffic"],
        accident_prob = config["map_setting"]["accident_prob"],
        map=config["map_setting"]["map_size"] if config["map_setting"]["PG"] else config["map_setting"]["map_sequence"],  # seven block
        start_seed=random.randint(0, 1000),
        vehicle_config = {"image_source":"rgb_camera", 
                          "rgb_camera":(config["rgb_setting"]["shape"][0], config["rgb_setting"]["shape"][1]), 
                          "show_lidar":True, 
                          "show_navi_mark":False},
    )
    env = TestPedeMetaDriveEnv(
        scene_config
    )
    asset_metainfo = {
        "length": 2,
        "width": 2,
        "height": 2,
        "filename": "car-3f699c7ce86c4ba1bad62a350766556f.glb",
        "CLASS_NAME": "06e459171a264e999b3763335403b719",
        "hshift": 0,
        "pos0": 0,
        "pos1": 0,
        "pos2": 0,
        "scale": 1
    }
    env.reset()
    path = '{}'.format(env.current_seed)
    folder = os.path.join(config["storage_path"], path)
    os.mkdir(folder)
    print(folder)
    try:
        count = 1
        # obj_1 = env.engine.spawn_object(TestObject, position=[30, -5], heading_theta=0, random_seed=1, force_spawn=True, asset_metainfo = asset_metainfo)
        for s in range(1, 100000000):
            o, r, tm, tc, info = env.step([0, 0])
            lidar_reading,observable = env.vehicle.lidar.perceive(env.vehicle)
            if s % config["sample_frequency"] == 0:
                agents = env.engine.agents
                agent = list(agents.values())[0] #if single-agent setting
                agent_id = list(agents.values())[0].id
                amin_point, amax_point = agent.origin.getTightBounds()
                p1 = amax_point[0],amax_point[1]
                p2 = amax_point[0],amin_point[1]
                p3 = amin_point[0],amin_point[1]
                p4 = amin_point[0],amax_point[1]
                atight_box = [p1,p2,p3,p4]
                aheight = amax_point[2]





                agent_description = dict(
                        id = agent.id,
                        color = 'red',    
                        heading = agent.heading ,      
                        lane = agent.lane_index,                          
                        speed =  agent.speed,
                        pos = agent.position,
                        bbox = [point for point in atight_box],
                        type = "Sedan",
                        height = aheight,
                        road_type = "NA",
                        class_name = str(type(agent))
                    )




                rgb_cam = env.vehicle.get_camera(env.vehicle.config["image_source"])
                hfov, vfov = rgb_cam.get_lens().fov



                observable = [get_object_from_node(car.getNode()) for car in list(observable)]
                if len(observable)==0:
                     continue
                
                observable_objects = []
                unique_id = set()

                Lidar_RGB_Observable_objects = []
                Lidar_RGB_Observable_boxs = []
                Lidar_RGB_Observable_heights = []




                for node in observable:
                    if node.id not in unique_id:
                        unique_id.add(node.id)
                        observable_objects.append(node)
                for object in observable_objects:
                    min_point, max_point = object.origin.getTightBounds(object.origin)
                    g_min_point,g_max_point = object.origin.getTightBounds()
                    Height = max_point[2]
                    p4 = LPoint3f(min_point[0],max_point[1],0)
                    p1 = LPoint3f(max_point[0],max_point[1],0)
                    p2 = LPoint3f(max_point[0],min_point[1],0)
                    p3 = LPoint3f(min_point[0],min_point[1],0)
                    tight_box = [p1,p2,p3,p4]
                    height = max_point[2]
                    origin_x, origin_y, _ = object.origin.getPos()
                    z_augmented = [p1,p2,p3,p4]
                    object_sample_box = sample_bbox(z_augmented,height,8,8,4)
                    observable_count = 0
                    total_count = len(object_sample_box)
                    for sample in object_sample_box:
                            sample_np = NodePath("tmp_node")
                            sample_np.reparentTo(object.origin)
                            sample_np.setPos(sample[0],sample[1],sample[2])
                            if rgb_cam.get_cam().node().isInView(sample_np.getPos(rgb_cam.get_cam())):
                                observable_count += 1
                            sample_np.detach_node()
                    if observable_count/total_count>=0.2:
                            Lidar_RGB_Observable_objects.append(object)
                            p4 = g_min_point[0],g_max_point[1]
                            p1 = g_max_point[0],g_max_point[1]
                            p2 = g_max_point[0],g_min_point[1]
                            p3 = g_min_point[0],g_min_point[1]
                            tight_box = [p1,p2,p3,p4]
                            Lidar_RGB_Observable_boxs.append(tight_box)
                            Lidar_RGB_Observable_heights.append(height)
                if len(Lidar_RGB_Observable_objects) == 0:
                        continue
                identifier = "{}_{}".format(env.current_seed,env.episode_step)
                instance_folder = os.path.join(folder,identifier)
                try:
                    os.mkdir(instance_folder)
                except:
                    print("Error in making instance folder")
                object_descriptions = [
                        dict(
                            id = Lidar_RGB_Observable_objects[i].id,
                            color = Lidar_RGB_Observable_objects[i].get_asset_metainfo()["general"]["color"] \
                                if (isinstance(Lidar_RGB_Observable_objects[i], CustomizedCar) \
                                    or  isinstance(Lidar_RGB_Observable_objects[i], TestObject))\
                                else "NA",
                            heading =  Lidar_RGB_Observable_objects[i].heading ,      
                            lane =  Lidar_RGB_Observable_objects[i].lane_index,                          
                            speed =   Lidar_RGB_Observable_objects[i].speed,
                            pos =  Lidar_RGB_Observable_objects[i].position,
                            bbox = [point for point in Lidar_RGB_Observable_boxs[i]],
                            type =Lidar_RGB_Observable_objects[i].get_asset_metainfo()["general"]["detail_type"] \
                                if (isinstance(Lidar_RGB_Observable_objects[i], CustomizedCar) \
                                    or  isinstance(Lidar_RGB_Observable_objects[i], TestObject))\
                                else "NA",
                            height =  Lidar_RGB_Observable_heights[i],
                            road_type = "NA",
                            class_name = str(type(Lidar_RGB_Observable_objects[i]))
                        )
                        for i in range(len(Lidar_RGB_Observable_objects))
                    ]
                scene_dict = dict()
                scene_dict["agent"] = agent_description
                scene_dict["vehicles"] = object_descriptions
                rgb_cam.save_image(env.vehicle, name= path + "/" +identifier + "/"+ "rgb_{}.png".format(identifier))
                ret1 = env.render(mode = 'top_down', film_size=(6000, 6000), target_vehicle_heading_up=False, screen_size=(3000,3000),show_agent_name=True)
                ret1 = pygame.transform.flip(ret1,flip_x = True, flip_y = False)
                file_path = os.path.join(instance_folder, "top_down_{}.png".format(identifier))
                pygame.image.save(ret1, file_path)
                ret2 = env.render(mode = 'rgb_array',screen_size=(1600, 900), film_size=(6000, 6000), target_vehicle_heading_up=True)
                file_path = os.path.join(instance_folder, "front_{}.png".format(identifier))
                cv2.imwrite(file_path,cv2.cvtColor(ret2, cv2. COLOR_BGR2RGB))
                try:
                    file_path = os.path.join(instance_folder,"world_{}.json".format(identifier))
                    with open(file_path, 'w') as f:
                        # write the dictionary to the file in JSON format 
                        json.dump(scene_dict,f)
                    file_path = os.path.join(instance_folder,"lidar_{}.json".format(identifier))
                    with open(file_path, 'w') as f:
                        observation = {}
                        observation['lidar'] = lidar_reading
                        # write the dictionary to the file in JSON format
                        json.dump(observation,f)
                except:
                    print("Error in storing JSON file")
                print("Generated the %d th data point" %(count))
                count += 1
            """for obj_id,obj in env.engine.get_objects().items():
                if isinstance(obj,CustomizedCar) or isinstance(obj, TestObject):
                    print(obj.get_asset_metainfo())
                else:
                    print(type(obj))"""

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True

    finally:
        env.close()


if __name__ == "__main__":
    try_pedestrian(True)
