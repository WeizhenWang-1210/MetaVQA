#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}

import numpy as np


def wrap_to_pi(radians_array):
    """
    Wrap all input radians to range [-pi, pi]
    """
    if isinstance(radians_array, np.ndarray):
        wrapped_radians_array = np.mod(radians_array, 2 * np.pi)
        wrapped_radians_array[wrapped_radians_array > np.pi] -= 2 * np.pi
    # elif isinstance(radians_array, torch.Tensor):
    #     wrapped_radians_array = radians_array % (2 * torch.tensor(np.pi))
    #     wrapped_radians_array[wrapped_radians_array > torch.tensor(np.pi)] -= 2 * np.pi
    elif isinstance(radians_array, (float, np.float32)):
        wrapped_radians_array = radians_array % (2 * np.pi)
        if wrapped_radians_array > np.pi:
            wrapped_radians_array -= 2 * np.pi
    else:
        raise ValueError("Input must be a NumPy array or PyTorch tensor")

    return wrapped_radians_array


def masked_average_numpy(tensor, mask, dim):
    """
    Compute the average of tensor along the specified dimension, ignoring masked elements.
    """
    assert tensor.shape == mask.shape
    count = mask.sum(axis=dim)
    count = np.maximum(count, np.ones_like(count))
    return (tensor * mask).sum(axis=dim) / count


class TurnAction:
    STOP = 0
    KEEP_STRAIGHT = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    U_TURN = 4

    num_actions = 5

    @classmethod
    def get_str(cls, action):
        if action == TurnAction.STOP:
            return "stop"  #"STOP"
        elif action == TurnAction.KEEP_STRAIGHT:
            return "forward"  #"KEEP_STRAIGHT"
        elif action == TurnAction.TURN_LEFT:
            return "go left"  #“TURN_LEFT"
        elif action == TurnAction.TURN_RIGHT:
            return "go right"  #"TURN_RIGHT"
        elif action == TurnAction.U_TURN:
            return "u turn"  #"U_TURN"
        else:
            raise ValueError("Unknown action: {}".format(action))


def get_direction_action_from_trajectory_batch(
        traj,
        mask,
        dt,
        U_TURN_DEG=115,
        LEFT_TURN_DEG=25,
        RIGHT_TURN_DEG=-25,
        STOP_SPEED=0.06,
):
    assert traj.ndim == 3
    traj_diff = traj[1:] - traj[:-1]
    mask_diff = mask[1:] & mask[:-1]

    displacement = np.linalg.norm(traj_diff, axis=-1)

    mask_diff_stop = mask_diff & (displacement > 0.1)

    pred_angles = np.arctan2(traj_diff[..., 1], traj_diff[..., 0])
    pred_angles_diff = wrap_to_pi(pred_angles[1:] - pred_angles[:-1])

    # It's meaning less to compute heading for a stopped vehicle. So mask them out!
    mask_diff_diff = mask_diff_stop[1:] & mask_diff_stop[:-1]
    # Note that we should not wrap to pi here because the sign is important.
    accumulated_heading_change_rad = (pred_angles_diff * mask_diff_diff).sum(axis=0)
    accumulated_heading_change_deg = np.degrees(accumulated_heading_change_rad)

    # print("accumulated_heading_change_deg: ", list(zip(ooi, accumulated_heading_change_deg)))

    speed = displacement / dt
    avg_speed = masked_average_numpy(speed, mask_diff, dim=0)

    actions = np.zeros(accumulated_heading_change_deg.shape, dtype=int)
    actions.fill(TurnAction.KEEP_STRAIGHT)
    actions[accumulated_heading_change_deg > LEFT_TURN_DEG] = TurnAction.TURN_LEFT
    actions[accumulated_heading_change_deg < RIGHT_TURN_DEG] = TurnAction.TURN_RIGHT
    actions[accumulated_heading_change_deg > U_TURN_DEG] = TurnAction.U_TURN
    actions[accumulated_heading_change_deg < -U_TURN_DEG] = TurnAction.U_TURN
    actions[avg_speed < STOP_SPEED] = TurnAction.STOP
    return actions


def get_navigation_signal(scenario, timestamp):
    U_TURN_DEG = 140
    TURN_DEG = 25
    STOP_SPEED = 0.06
    chunk_size = 50  # 10 frames = 1s
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    ego_valid_mask = ego_track["state"]["valid"].astype(bool)
    T = ego_traj.shape[0]
    if timestamp + chunk_size > T:
        timestamp = T - chunk_size
    traj = ego_traj[timestamp:timestamp + chunk_size]
    mask = ego_valid_mask[timestamp:timestamp + chunk_size]
    assert traj.shape[0] == chunk_size
    actions = get_direction_action_from_trajectory_batch(
        traj.reshape(chunk_size, 1, -1),
        mask.reshape(chunk_size, 1),
        dt=0.1,
        U_TURN_DEG=U_TURN_DEG,
        LEFT_TURN_DEG=TURN_DEG,
        RIGHT_TURN_DEG=-TURN_DEG,
        STOP_SPEED=STOP_SPEED,
    )
    actions = actions[0]
    print("Action at step {}, t={}s: {}".format(timestamp, timestamp / 10, TurnAction.get_str(actions)))
    #print(111)
    return actions


def get_trajectory(env):
    """
    n,2 array
    """
    scenario = env.engine.data_manager.current_scenario
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    return ego_traj


# Function to check if a point P can be projected onto a segment AB


def dynamic_get_navigation_signal(scenario, timestamp, env):
    def is_point_on_segment(A, B, P):
        AB = B - A
        AP = P - A
        # Calculate the projection scalar t
        t = np.dot(AP, AB) / np.dot(AB, AB)
        # Check if the projection falls within the segment
        if 0 <= t <= 1:
            # Calculate the projected point
            P_proj = A + t * AB
            return True, P_proj
        else:
            return False, None
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    #Sparsity to every 0.5 seconds while making sure the end is always there
    END = ego_traj[-1]
    ego_traj = ego_traj[:-1]
    ego_traj = np.vstack([ego_traj[::5,:], END])
    T = ego_traj.shape[0]
    adjustment_duration = 8 # looks into 4 seconds of future
    ego_pos, ego_heading = np.array(env.agent.position), np.array(env.agent.heading)
    segments_with_projection = []
    for i in range(len(ego_traj) - 1):
        A = ego_traj[i]
        B = ego_traj[i + 1]
        on_segment, projected_point = is_point_on_segment(A, B, ego_pos)
        if on_segment:
            segments_with_projection.append((i, projected_point))
    if len(segments_with_projection) == 0:
        # We are so wrong, and we have to go to the ending point.
        future_pos = ego_traj[-1]
    else:
        projected_segment_index, projected_point = \
        sorted(segments_with_projection, key=lambda x: np.linalg.norm(x[1]), reverse=False)[0]  #choose closest waypoint
        #make sure the destination is sufficiently 2 meters away form current positions.
        if adjustment_duration + projected_segment_index < T:
            dest_index = projected_segment_index + adjustment_duration
            while dest_index < T and np.linalg.norm(ego_traj[dest_index] - ego_pos) < 2:
                dest_index += 1
            if dest_index >= T:
                dest_index = T-1
            assert dest_index < T
            future_pos = ego_traj[dest_index]
        else:
            future_pos = ego_traj[-1]
    pos_diff = future_pos - ego_pos
    angle = np.arccos(np.dot(pos_diff, ego_heading) / (np.linalg.norm(pos_diff) * np.linalg.norm(ego_heading)))
    #same_dir = np.dot(pos_diff, ego_heading) > 0
    wrapped_angle = angle * -1 if np.cross(pos_diff, ego_heading) > 0 else angle * 1
    wrapped_angle = np.degrees(wrapped_angle)
    if wrapped_angle > 5:
        action = TurnAction.TURN_LEFT
        #if not same_dir:
        #    print("U")
    elif wrapped_angle < -5:
        action = TurnAction.TURN_RIGHT
        #if not same_dir:
        #    print("U")
    else:
        action = TurnAction.KEEP_STRAIGHT
    print("Dynamic Navigation at step {}, t={}s: {}".format(timestamp, timestamp / 10, TurnAction.get_str(action)))
    return TurnAction.get_str(action)


def dest_navigation_signal(scenario, timestamp, env):
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    #Sparsity to every 0.5 seconds while making sure the end is always there
    END = ego_traj[-1,:]
    ego_pos, ego_heading = np.array(env.agent.position), np.array(env.agent.heading)
    future_pos = END
    distance = np.linalg.norm(future_pos-ego_pos)
    pos_diff = future_pos - ego_pos
    angle = np.arccos(np.dot(pos_diff, ego_heading) / (np.linalg.norm(pos_diff) * np.linalg.norm(ego_heading)))
    same_dir = np.dot(pos_diff, ego_heading) > 0
    wrapped_angle = angle * -1 if np.cross(pos_diff, ego_heading) > 0 else angle * 1
    wrapped_angle = np.degrees(wrapped_angle)
    if wrapped_angle > 5:
        if same_dir:
            dir = "lf"
        else:
            dir = "lb"
    elif wrapped_angle < -5:
        if same_dir:
            dir = "rf"
        else:
            dir = "rb"
    else:
        if same_dir:
            dir = "f"
        else:
            dir = "b"
    print("Dynamic Navigation at step {}, t={}s: {}-{}m".format(timestamp, timestamp / 10, dir, distance))
    return dir, distance



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", "--topdown", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)
    try:
        env = ScenarioEnv(
            {
                #"manual_control": True,
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": AssetLoader.file_path(
                    asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False
                ),
                "num_scenarios": 3 if use_waymo else 10,
                "agent_policy": ReplayEgoCarPolicy
            }
        )
        o, _ = env.reset()

        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])

            get_navigation_signal(env.engine.data_manager.current_scenario, timestamp=env.episode_step)

            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            if tm or tc:
                env.reset()
    finally:
        env.close()
