
from metadrive import MetaDriveEnv
from metadrive.examples import expert

# Set the envrionment config
env_config={
    "manual_control": True,
    "use_render": True,
    # "controller": "keyboard",  # or joystick
    "window_size": (1600, 1100),
    "start_seed": 1000,
    # "map": "COT",
    # "environment_num": 1,
}


# ===== Setup the training environment =====
env = MetaDriveEnv(config=env_config)

o, _ = env.reset()
env.vehicle.expert_takeover = True
count = 1
for i in range(1, 9000000000):
    o, r, tm, tc, info = env.step([0, 0])