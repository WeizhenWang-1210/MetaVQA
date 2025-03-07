from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
import copy
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion
from vqa.configs.NAMESPACE import MIN_OBSERVABLE_PIXEL
from masking import find_areas
import json
import os
from nuscenes.eval.common.utils import quaternion_yaw
from PIL import Image

#TODO  FURTHER FILTER BY OBSERVABLE PIXELS.

ALL_TYPE = {
    "noise": 'noise',
    "human.pedestrian.adult": 'Pedestrian',
    "human.pedestrian.child": 'Pedestrian',
    "human.pedestrian.wheelchair": 'Wheelchair',
    "human.pedestrian.stroller": 'Stroller',
    "human.pedestrian.personal_mobility": 'p.mobility',
    "human.pedestrian.police_officer": 'Police_officer',
    "human.pedestrian.construction_worker": 'Construction_worker',
    "animal": 'Animal',
    "vehicle.car": 'Car',
    "vehicle.motorcycle": 'Motorcycle',
    "vehicle.bicycle": 'Bike',
    "vehicle.bus.bendy": 'Bus',
    "vehicle.bus.rigid": 'Bus',
    "vehicle.truck": 'Truck',
    "vehicle.construction": 'Construction_vehicle',
    "vehicle.emergency.ambulance": 'Ambulance',
    "vehicle.emergency.police": 'Policecar',
    "vehicle.trailer": 'Trailer',
    "movable_object.barrier": 'Barrier',
    "movable_object.trafficcone": 'Cone',
    "movable_object.pushable_pullable": 'push/pullable',
    "movable_object.debris": 'debris',
    "static_object.bicycle_rack": 'bicycle racks',
    "flat.driveable_surface": 'driveable',
    "flat.sidewalk": 'sidewalk',
    "flat.terrain": 'terrain',
    "flat.other": 'flat.other',
    "static.manmade": 'manmade',
    "static.vegetation": 'vegetation',
    "static.other": 'static.other',
    "vehicle.ego": "ego"
}

IGNORED_NUSC_TYPE = (
    "noise", "human.pedestrian.personal_mobility", "movable_object.pushable_pullable", "movable_object.debris",
    "static_object.bicycle_rack", "flat.driveable_surface", "flat.sidewalk", "flat.terrain", "flat.other",
    "static.manmade",
    "static.vegetation", "static.other")


def normalize_point(shape, point):
    new_point = list(point)
    if point[0] < 0:
        new_point[0] = 0
    elif point[0] > shape[0]:
        new_point[0] = shape[0]
    if point[1] < 0:
        new_point[1] = 0
    elif point[1] > shape[1]:
        new_point[1] = shape[1]
    return tuple(new_point)


def find_extremeties(points):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    max_area = abs((ymin - ymax) * (xmin - xmax))
    return (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), max_area


def create_mask(shape, points):
    # Create a path using the given points
    points = [normalize_point(shape, point) for point in points]
    result = find_extremeties(np.array(points))
    normalized_area = result[-1]
    path = Path(points)
    # Get all coordinates in the grid
    y, x = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = np.vstack((x.flatten(), y.flatten())).T
    # Find points inside the path
    mask = path.contains_points(coords).reshape(shape)
    return mask, normalized_area


import tqdm


def func(nusc, ego_pose_token, boxes, visibilities, camera_intrinsic):
    """
    boxes need to be in ego-sensor-centric view
    """
    # order boxes from furthest to the closest
    visibilities_copy = copy.deepcopy(visibilities)

    ordered_boxes = sorted(
        list(enumerate(boxes)),
        key=lambda pair: np.linalg.norm(np.array(nusc.get("sample_annotation", pair[1].token)["translation"][:2]) \
                                        - np.array(nusc.get("ego_pose", ego_pose_token)["translation"][:2])),
        reverse=True
    )
    individual_masks = []
    color_mapping = {}
    corners_mapping = {}
    for idx, pair in enumerate(ordered_boxes):
        old_idx, box = pair
        if not visibilities[old_idx]:
            continue
        color = plt.get_cmap('viridis')(old_idx / len(boxes))
        color = [round(c * 255) for c in color[:3]]
        # print("Color", color)
        corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
        # Define the eight points
        points = corners.T
        # Switch (x,y) dimension order
        flipped_points = np.flip(points, axis=1)
        # Get the enclosing 2D-Bounding boxes
        flipped_points = find_extremeties(flipped_points)
        max_area = flipped_points[-1]
        flipped_points = flipped_points[:4]
        # Separate the x and y coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        # Create a binary mask of the region
        mask_shape = (900, 1600)  # Size of the mask
        mask, normalized_area = create_mask(mask_shape, flipped_points)
        # print(max_area, normalized_area, round(normalized_area/max_area,2))
        if normalized_area / max_area < 0.5 or normalized_area < MIN_OBSERVABLE_PIXEL:
            visibilities_copy[old_idx] = False
            continue
        color_mapping[old_idx] = color
        corners_mapping[old_idx] = [normalize_point(mask_shape, point) for point in flipped_points]
        image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        image[mask] = color  # [:3]
        # image = image.reshape(height, width, 3)
        individual_masks.append(image)
    # now we've created individual maps, superimpose them from far to close(similar to z-buffering)
    final_result = np.zeros((900, 1600, 3))
    for individual_mask in individual_masks:
        bmask = ~np.all(individual_mask == [0, 0, 0], axis=-1)
        bmask_3d = np.stack([bmask] * 3, axis=-1)
        final_result = np.where(bmask_3d, individual_mask, final_result)
    combined_image_pil = Image.fromarray(final_result.astype('uint8'))

    #further processing to account for observable pixels after occlusion.
    areas, masks = find_areas(img=final_result.astype('uint8'), colors=[tuple(val) for val in color_mapping.values()])
    new_color_mapping = {}
    new_corners_mapping = {}
    for idx, (key, value) in enumerate(color_mapping.items()):
        if areas[idx] >= MIN_OBSERVABLE_PIXEL:
            new_color_mapping[key] = value
            new_corners_mapping[key] = corners_mapping[key]
        else:
            visibilities_copy[key] = False  #key here is the old indices consistent in visibilities.

    # fig, axes = plt.subplots(figsize=(1600 / 100, 900 / 100), dpi=100)
    # axes.imshow(combined_image_pil)
    # axes.axis('off')
    return combined_image_pil, visibilities_copy, new_color_mapping, new_corners_mapping


def traverse(nusc, start, steps):
    cur_sample = start
    while steps > 0 and cur_sample["next"] != "":
        cur_sample = nusc.get("sample", cur_sample["next"])
        steps -= 1
    return cur_sample


def job(job_range=[1, 2], root="./", nusc=None, proc_id=0):
    if nusc is None:
        nusc = NuScenes(version='v1.0-trainval', dataroot='/bigdata/datasets/nuscenes', verbose=True)
    nusc_scenes = nusc.scene
    EGO_SHAPE = (1.730, 4.084, 1.562)
    for scene_idx, nusc_scene in tqdm.tqdm(enumerate(nusc_scenes[job_range[0]:job_range[1]]),
                                           desc=f"Process-{proc_id}, {job_range[1] - job_range[0]} scenes in total",
                                           unit="scene"):
        frame_idx = 0  # zero-based index
        sample = nusc.get("sample", nusc_scene["first_sample_token"])
        assert sample is not None
        nbr_samples = nusc_scene["nbr_samples"]
        print(f"Total of {nbr_samples} frames in scene-{scene_idx}")
        frame_annotations = {}
        scene_name = nusc_scene["name"]
        scene_length = nusc_scene["nbr_samples"]
        #TODO create scene-consistent color mapping.
        for frame_idx in tqdm.tqdm(range(scene_length),
                                   desc=f"Process-{proc_id}, annotating {scene_name} with {scene_length} frames",
                                   unit="frame"):  #while frame_idx < nusc_scene["nbr_samples"]:
            # Get CAM_FRONT data
            cam_front_sample_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            # Get the path to img, bboxes, and camera_instrinsics.
            data_path, visible_boxes, camera_intrinsic = nusc.get_sample_data(cam_front_sample_data['token'],
                                                                              box_vis_level=BoxVisibility.ANY)
            if len(visible_boxes) > 0:
                original_boxes = nusc.get_boxes(cam_front_sample_data['token'])
                # Get the sample_annotaion token associated with each box inside the frustum.
                visible_boxes_token = [box.token for box in visible_boxes]
                # find the corresponding sample_annotations for each box.
                sample_annotations = [nusc.get("sample_annotation", token) for token in visible_boxes_token]
                # find the attributes for each sample_annotation. One object have a list of attributes at particuar frame.
                attribute_annotations = [
                    [nusc.get("attribute", token) for token in
                     sample_annotation["attribute_tokens"]
                     ]
                    for sample_annotation in sample_annotations]
                # Get the instance token. One instance token can map to muliple annotation tokens(depending on the timestamp)
                instance_tokens = [sample_annotation["instance_token"] for sample_annotation in sample_annotations]
                ego_pose = nusc.get("ego_pose", cam_front_sample_data["ego_pose_token"])
                # get ego pos in world coordinates
                ego_translation = ego_pose["translation"]
                # get ego heading in Quarterion(world?)
                ego_rotation = ego_pose["rotation"]
                # get obj pos in world coordinates
                instance_translations = [sample_annotation["translation"] for sample_annotation in sample_annotations]
                # get obj heading in Quarterion(world?)
                instance_rotations = [sample_annotation["rotation"] for sample_annotation in sample_annotations]
                # get obj type
                instance_types = [nusc.get("instance", sample_annotation["instance_token"])["category_token"] for
                                  sample_annotation in sample_annotations]
                instance_types = [nusc.get("category", instance_type)["name"] for instance_type in instance_types]
                # Get visible objects(no occlusion considered, only in frustum)'s 3d bounding boxes in world coordinates.
                visible_boxes_world = [box for box in original_boxes if box.token in visible_boxes_token]
                # Get visible objects(no occlusion considered, only in frustum)'s top-down bounding box's corners in world coordinates. (n,4,2)
                visible_boxes_world_bottom = [box.bottom_corners()[:2].T for box in visible_boxes_world]
                # Get number of lidar shot on the object
                num_lidar_points = [sample_annotation["num_lidar_pts"] for sample_annotation in sample_annotations]
                # Get the visible level for each object
                visibility_levels = [nusc.get("visibility", sample_annotation["visibility_token"]) for sample_annotation
                                     in sample_annotations]
                # Get ego's top-down bounding box's corners in world coordinates. (4,2)
                ego_box_world_bottom = Box(
                    center=ego_translation,
                    size=EGO_SHAPE,
                    orientation=Quaternion(ego_rotation),
                    name="EGO"
                ).bottom_corners()[:2, :].T
                medium_visible_indices = []
                #first, filter out some clearly none-visible objects.
                for idx, box in enumerate(visible_boxes):
                    if num_lidar_points[idx] < 5 or visibility_levels[idx]["token"] not in ["3", "4"]:
                        medium_visible_indices.append(False)
                    else:
                        medium_visible_indices.append(True)
                # second, create instance_segmentation mask and filter out objects with less than 0.5 area visible in the image.
                instance_seg, final_visible_indices, color_mapping, corners_mapping = func(
                    nusc=nusc,
                    ego_pose_token=cam_front_sample_data["ego_pose_token"],
                    boxes=visible_boxes,
                    visibilities=medium_visible_indices,
                    camera_intrinsic=camera_intrinsic
                )
                # The sample_annotation token for the visible(occlusion, observable pixels considered). objects
                frame_data = {
                    "ego": None, "objects": [], "world": scene_name, "data_summary": scene_name
                }
                # print(instance_tokens)
                # print(medium_visible_indices)
                # print(final_visible_indices)
                # print(f"{frame_idx}:{len(instance_rotations)}")
                for idx, visible_flag in enumerate(final_visible_indices):
                    if not visible_flag or instance_types[idx] in IGNORED_NUSC_TYPE:
                        continue
                    yaw = quaternion_yaw(Quaternion(instance_rotations[idx]))
                    data_point = dict(
                        id=instance_tokens[idx],
                        color="",
                        heading=[np.cos(yaw), np.sin(yaw)],
                        pos=instance_translations[idx][:2],
                        speed=np.linalg.norm(nusc.box_velocity(visible_boxes_token[idx])),
                        bbox=visible_boxes_world_bottom[idx].tolist(),
                        type=ALL_TYPE[instance_types[idx]],
                        height=sample_annotations[idx]["size"][2],
                        class_name=instance_types[idx],
                        visible=True,
                        observing_camera=["front"],
                        collisions=[],
                    )
                    frame_data["objects"].append(data_point)
                yaw = quaternion_yaw(Quaternion(ego_rotation))
                frame_data["ego"] = dict(
                    id="EGO",
                    color="",
                    heading=[np.cos(yaw), np.sin(yaw)],
                    pos=ego_translation[:2],
                    speed=0,
                    bbox=ego_box_world_bottom.tolist(),
                    type=ALL_TYPE["vehicle.ego"],
                    height=EGO_SHAPE[2],
                    class_name="vehicle.ego",
                    visible=False,
                    observing_camera=[],
                    collisions=[]
                )
                # R,G,B, int -> R,G,B, float
                id2c = {
                    instance_tokens[key]: [round(c / 255, 5) for c in value] for key, value in color_mapping.items()
                }
                id2corners = {
                    instance_tokens[key]: value for key, value in corners_mapping.items()
                }
                frame_annotations[frame_idx] = dict(
                    annotation=frame_data,
                    id2c=id2c,
                    instance_seg=instance_seg,
                    rgb=Image.open(data_path),
                    id2corners=id2corners
                )
            #print(f"Done {frame_idx} for {scene_name}")
            sample = traverse(nusc, sample, 1)
            #frame_idx += 1
        episode_path = os.path.join(root, f"{scene_name}_0_{scene_length - 1}")
        for frame_idx, records in tqdm.tqdm(frame_annotations.items(),
                                            desc=f"Process-{proc_id}, storing {len(frame_annotations)} annotations for {scene_name}",
                                            unit="point"):
            identifier = f"{scene_idx}_{frame_idx}"
            #print(f"Saving for {identifier}")
            frame_path = os.path.join(episode_path, identifier)
            os.makedirs(frame_path, exist_ok=True)
            id2c_path = os.path.join(frame_path, f"id2c_{identifier}.json")
            id2corners_path = os.path.join(frame_path, f"id2corners_{identifier}.json")
            rgbf_path = os.path.join(frame_path, f"rgb_front_{identifier}.png")
            mask_path = os.path.join(frame_path, f"mask_front_{identifier}.png")
            world_path = os.path.join(frame_path, f"world_{identifier}.json")
            try:
                json.dump(records["annotation"], open(world_path, "w"), indent=2)
                json.dump(records["id2c"], open(id2c_path, "w"), indent=2)
                json.dump(records["id2corners"], open(id2corners_path, "w"), indent=2)
                records["rgb"].save(rgbf_path)
                records["instance_seg"].save(mask_path)
            except Exception as e:
                raise e
        assert sample["token"] == nusc_scene["last_sample_token"]


def split_ranges(start, end, chunk_size):
    result = []
    while start + chunk_size < end:
        result.append([start, start + chunk_size])
        start += chunk_size
    result.append([start, end])
    return result


def main():
    import argparse
    import multiprocessing as multp
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Inclusive starting index")
    parser.add_argument("--end", type=int, default=800, help="Exclusive starting index")
    parser.add_argument("--num_proc", type=int, default=2, help="Processes used to extract scenarios")
    parser.add_argument("--store_dir", type=str, default="./", help="Root directory of stored episodes")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    job_range = [args.start, args.end]
    num_scenes = args.end - args.start
    num_proc = args.num_proc
    chunk_size = math.ceil(num_scenes / num_proc)
    print(f"Working on {num_scenes} frames with {num_proc} process(es), {chunk_size} MAX scenes per process")
    job_ranges = split_ranges(job_range[0], job_range[1], chunk_size)
    assert len(job_ranges) == num_proc and job_ranges[0][0] == job_range[0] and job_ranges[-1][1] == job_range[1]
    root = args.store_dir
    nusc = NuScenes(version='v1.0-trainval', dataroot='/bigdata/datasets/nuscenes', verbose=True)
    total_scenes = len(nusc.scene)
    assert args.start >=0 and args.end <= num_scenes and num_scenes <= total_scenes, "Invalid range!"
    processes = []
    for proc_id in range(num_proc):
        print(f"Sending job {proc_id}")
        p = multp.Process(
            target=job,
            args=(
                job_ranges[proc_id],
                root,
                None,
                proc_id
            )
        )
        print(f"Successfully sent {proc_id}")
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


if __name__ == "__main__":
    main()
