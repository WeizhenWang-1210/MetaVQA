"""
This script defines a custom environment with single block: inramp.
"""
import random

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.envs.marl_envs.marl_parking_lot import MultiAgentParkingLotEnv

if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        start_seed=random.randint(0, 1000),
        traffic_density=0.05,

        # Solution 1: use easy config to customize the map
        # map="r",  # seven block

        # Solution 2: you can define more complex map config
        #map_config=dict(lane_num=1, type="block_sequence", config="SP")
    )

    env = MultiAgentParkingLotEnv(config)
    try:
        o, _ = env.reset()
        env.vehicle.expert_takeover = True
        assert isinstance(o, np.ndarray)
        print("The observation is an numpy array with shape: ", o.shape)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            if (tm or tc) and info["arrive_dest"]:
                print("s")
                env.reset()
                env.current_track_vehicle.expert_takeover = True
    except:
        pass
    finally:
        env.close()