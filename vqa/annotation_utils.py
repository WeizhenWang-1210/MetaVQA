from typing import Iterable, Callable, List
import numpy as np
from metadrive.envs.base_env import BaseEnv
from metadrive.base_class.base_object import BaseObject
from metadrive.component.vehicle.vehicle_type import BaseVehicle, SVehicle, MVehicle, LVehicle, XLVehicle, \
    DefaultVehicle, StaticDefaultVehicle, VaryingDynamicsVehicle, CustomizedCar
from metadrive.component.static_object.traffic_object import TrafficBarrier, TrafficCone, TrafficWarning
from metadrive.component.traffic_light.scenario_traffic_light import ScenarioTrafficLight
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.static_object.test_new_object import TestObject


def annotate_type(object):
    """
    Return the predefined type annotation of an object. This only applies to metadrive-native classes.
    """

    if isinstance(object, TestObject) or isinstance(object, CustomizedCar):
        # print(object.get_asset_metainfo())
        return object.get_asset_metainfo()['general']["detail_type"]
    # TODO Cautious Here
    vehicle_type = {
        SVehicle: "Hatchback",
        MVehicle: "Sedan",
        LVehicle: "Pickup",
        XLVehicle: "Truck",
        DefaultVehicle: "Sedan",
        StaticDefaultVehicle: "Sedan",
        VaryingDynamicsVehicle: "Sedan",
        TrafficBarrier: "Barrier",
        TrafficWarning: "Warning",
        TrafficCone: "Cone",
        Cyclist: "Bike",
        Pedestrian: "Pedestrian",
        ScenarioTrafficLight: "TrafficLight"
    }
    for c, name in vehicle_type.items():
        if isinstance(object, c):
            return name
    return "f"


def annotate_lane(object):
    raise DeprecationWarning("This function is not useful")

    if isinstance(object, BaseVehicle) and object.navigation:
        return object.lane_index
    else:
        return ""


def annotate_color(object):
    """
    Return the predefined type annotation of an object. This only applies to metadrive-native classes.
    """
    if isinstance(object, TestObject) or isinstance(object, CustomizedCar):
        return object.get_asset_metainfo()['general']["color"]
    vehicle_type = {
        SVehicle: "Blue",
        MVehicle: "White",
        LVehicle: "Gray",
        XLVehicle: "White",
        DefaultVehicle: "Red",
        StaticDefaultVehicle: "Red",
        VaryingDynamicsVehicle: "Red",
        TrafficCone: "Orange",
        TrafficBarrier: "White",
        TrafficWarning: "Red",
        Cyclist: "White",
        Pedestrian: "White",
        ScenarioTrafficLight: "Traffic Light"
    }
    for c, color in vehicle_type.items():
        if isinstance(object, c):
            return color
    return "f"


def annotate_states(object):
    raise DeprecationWarning("This function is not useful")
    states = dict()
    if isinstance(object, CustomizedCar) or isinstance(object, BaseVehicle):
        states["acceleration"] = object.throttle_brake
        states["steering"] = object.steering
    return states


def annotate_collision(obj):
    result = []
    if isinstance(obj, BaseVehicle):
        result = list(obj.crashed_objects)
    return result


def get_visible_object_ids(imgs: np.array, mapping: dict(), filter: Callable) -> Iterable[str]:
    '''
    imgs: np.array(H,W,C), the observation by the instance segmentation camera, clipped between 0,1
    mapping: dictionary mapping (r,g,b) to object id. Note that each float is rounded to 5 points
    filter: boolean function to filter out certain colors. Takes a tuple (r,g,b) and an int c 
    
    return an iterable containings the ids of all visible items, mapping of all objects to their respective colors
    note that objects can be considered "invisible" by having few pixels observed.
    '''
    flattened = imgs.reshape(-1, imgs.shape[2])
    unique_colors, counts = np.unique(flattened, axis=0, return_counts=True)
    unique_colors, counts = unique_colors.tolist(), counts.tolist()
    unique_colors_filtered = [(r, g, b) for (b, g, r), c in zip(unique_colors, counts) if filter(r, g, b, c)]
    unique_colors_processed = [(round(r, 5), round(g, 5), round(b, 5)) for r, g, b in unique_colors_filtered]
    return [mapping[unique_color] for unique_color in unique_colors_processed], \
        {mapping[unique_colors_processed[i]]: unique_colors_filtered[i] for i in range(len(unique_colors_processed))}


def generate_annotations(objects: Iterable[BaseObject], env: BaseEnv, visible_mask: List[bool] = [False],
                         observing_camera=[[]]) -> Iterable[dict]:
    result = []
    for idx, obj in enumerate(objects):
        g_min_point, g_max_point = obj.origin.getTightBounds()
        height = g_max_point[2]
        box = obj.bounding_box
        box = [list(b) for b in box]
        speed = -1
        position = obj.position.tolist()

        if isinstance(obj, BaseObject):
            speed = obj.speed
        annotation = dict(
            id=obj.id,
            color=annotate_color(obj),
            heading=(obj.heading[0], obj.heading[1]),
            speed=speed,
            pos=position,
            bbox=box,
            type=annotate_type(obj),
            height=height,
            class_name=str(type(obj)),
            visible=visible_mask[idx],
            observing_camera=observing_camera[idx],
            collisions=annotate_collision(obj)
        )
        result.append(annotation)
    return result


def genearte_annotation(object: BaseObject, env: BaseEnv, visible_mask=[False], observing_camera=[[]]) -> dict:
    return generate_annotations([object], env, visible_mask, observing_camera)[0]
