"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import random

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        num_scenarios=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        # debug=True,
        # debug_static_world=True,
        map=4,  # seven block
        start_seed=random.randint(0, 1000)
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(image_observation=True))
    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if i % 100== 0: #note: the "1st" object in objects is the agent
                objects = env.engine.get_objects()
                agents = env.engine.agents
                agent = list(agents.values())[0] #if single-agent setting
                
                agent_id = list(agents.values())[0].id
                print("Agent: ",agent.position)
                print("Agent lane: ", agent.lane_index)
                for id, object in objects.items():
                    if id != agent_id:
                        relative_displacement = object.convert_to_local_coordinates(object.position,agent.position)
                        relative_distance = np.sqrt(relative_displacement[0]**2 + relative_displacement[1]**2)
                        if relative_distance <= 30:
                            #print("old color:", object.origin.getMaterial())
                            #object.panda_color = [1,1,1]
                            #print("New color:", object.panda_color)
                            print("Object_relative: ",relative_distance)
                            #object._panda_color = old_color

                        
                        
            

               
                    
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True
    except Exception as e:
        raise e
    finally:
        env.close()
