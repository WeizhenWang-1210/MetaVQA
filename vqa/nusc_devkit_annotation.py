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
import json
import os
from nuscenes.eval.common.utils import quaternion_yaw
from PIL import Image

ALL_TYPE = {
    "noise": 'noise',
    "human.pedestrian.adult": 'adult',
    "human.pedestrian.child": 'child',
    "human.pedestrian.wheelchair": 'wheelchair',
    "human.pedestrian.stroller": 'stroller',
    "human.pedestrian.personal_mobility": 'p.mobility',
    "human.pedestrian.police_officer": 'police',
    "human.pedestrian.construction_worker": 'worker',
    "animal": 'animal',
    "vehicle.car": 'car',
    "vehicle.motorcycle": 'motorcycle',
    "vehicle.bicycle": 'bicycle',
    "vehicle.bus.bendy": 'bus.bendy',
    "vehicle.bus.rigid": 'bus.rigid',
    "vehicle.truck": 'truck',
    "vehicle.construction": 'constr. veh',
    "vehicle.emergency.ambulance": 'ambulance',
    "vehicle.emergency.police": 'police car',
    "vehicle.trailer": 'trailer',
    "movable_object.barrier": 'barrier',
    "movable_object.trafficcone": 'trafficcone',
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




def find_extremeties(points):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    max_area = abs((ymin - ymax) * (xmin - xmax))
    return (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), max_area
def create_mask(shape, points):
    def normalize_point(point):
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
    # Create a path using the given points
    points = [normalize_point(point) for point in points]
    result = find_extremeties(np.array(points))
    normalized_area = result[-1]
    path = Path(points)
    # Get all coordinates in the grid
    y, x = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = np.vstack((x.flatten(), y.flatten())).T
    # Find points inside the path
    mask = path.contains_points(coords).reshape(shape)
    return mask, normalized_area
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
        if normalized_area / max_area < 0.5:
            visibilities_copy[old_idx] = False
            continue
        color_mapping[old_idx] = color
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
    # fig, axes = plt.subplots(figsize=(1600 / 100, 900 / 100), dpi=100)
    # axes.imshow(combined_image_pil)
    # axes.axis('off')

    return combined_image_pil, visibilities_copy, color_mapping
    # combined_image_pil.save('combined_image.png')
def traverse(nusc, start,steps):
    cur_sample = start
    while steps > 0 and cur_sample["next"] != "":
        cur_sample = nusc.get("sample", cur_sample["next"])
        steps -= 1
    return cur_sample

def main():
    ROOT = "/bigdata/weizhen/metavqa_iclr/scenarios/nusc_real"
    nusc = NuScenes(version='v1.0-trainval', dataroot='/bigdata/datasets/nuscenes', verbose=True)
    nusc_scenes = nusc.scene
    EGO_SHAPE = (1.730, 4.084, 1.562)
    for scene_idx, nusc_scene in enumerate(nusc_scenes[1:2]):
        frame_idx = 0  # zero-based index
        sample = nusc.get("sample", nusc_scene["first_sample_token"])
        nbr_samples = nusc_scene["nbr_samples"]
        print(f"Total of {nbr_samples} frames in scene-{scene_idx}")
        frame_annotations = {}
        while frame_idx < nusc_scene["nbr_samples"]:
            # Get CAM_FRONT data
            cam_front_sample_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            # Get the path to img, bboxes, and camera_instrinsics.
            data_path, visible_boxes, camera_intrinsic = nusc.get_sample_data(cam_front_sample_data['token'],box_vis_level=BoxVisibility.ANY)
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
                instance_seg, final_visible_indices, color_mapping = func(
                    nusc=nusc,
                    ego_pose_token=cam_front_sample_data["ego_pose_token"],
                    boxes=visible_boxes,
                    visibilities=medium_visible_indices,
                    camera_intrinsic=camera_intrinsic
                )
                # The sample_annotation token for the visible(occlusion, observable pixels considered). objects
                frame_data = {
                    "ego": None, "objects": []
                }
                # print(instance_tokens)
                # print(medium_visible_indices)
                # print(final_visible_indices)
                print(f"{frame_idx}:{len(instance_rotations)}")
                for idx, visible_flag in enumerate(final_visible_indices):
                    if not visible_flag:
                        continue
                    print(idx)
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
                        class_name=ALL_TYPE[instance_types[idx]],
                        visible=True,
                        observing_camera=["front"],
                        collisions=[]
                    )
                    frame_data["objects"].append(data_point)
                yaw = quaternion_yaw(Quaternion(ego_rotation))
                frame_data["ego"] = dict(
                    id="EGO",
                    color="",
                    heading=[np.cos(yaw), np.sin(yaw)],
                    pos=ego_translation[:2],
                    speed=0,
                    bbox=visible_boxes_world_bottom[idx].tolist(),
                    type=ALL_TYPE["vehicle.ego"],
                    height=EGO_SHAPE[2],
                    class_name=ALL_TYPE["vehicle.ego"],
                    visible=False,
                    observing_camera=[],
                    collisions=[]
                )
                # R,G,B, int -> R,G,B, float
                id2c = {
                    instance_tokens[key]: [round(c / 255, 5) for c in value] for key, value in color_mapping.items()
                }
                frame_annotations[frame_idx] = dict(
                    annotation=frame_data,
                    id2c=id2c,
                    instance_seg=instance_seg,
                    rgb=Image.open(data_path)
                )
            sample = traverse(nusc, sample, 1)
            frame_idx += 1
        scene_name = nusc_scene["name"]
        episode_path = os.path.join(ROOT, f"{scene_name}_0_{frame_idx - 1}")
        for frame_idx, records in frame_annotations.items():
            identifier = f"{scene_idx}_{frame_idx}"
            print(f"Saving for {identifier}")
            frame_path = os.path.join(episode_path, identifier)
            os.makedirs(frame_path, exist_ok=True)
            id2c_path = os.path.join(frame_path, f"id2c_{identifier}.json")
            rgbf_path = os.path.join(frame_path, f"rgb_front_{identifier}.png")
            mask_path = os.path.join(frame_path, f"mask_front_{identifier}.png")
            world_path = os.path.join(frame_path, f"world_{identifier}.json")
            try:
                json.dump(records["annotation"], open(world_path, "w"), indent=2)
                json.dump(records["id2c"], open(id2c_path, "w"), indent=2)
                records["rgb"].save(rgbf_path)
                records["instance_seg"].save(mask_path)
            except Exception as e:
                raise e
        assert sample["token"] == nusc_scene["last_sample_token"]
if __name__ == "__main__":
    main()