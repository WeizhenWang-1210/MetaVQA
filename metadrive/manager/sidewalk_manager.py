"""
This script defines the SidewalkManager class, which is responsible for adding items
on sidewalks.
This class extends the functionality of the BaseManager, leveraging its capabilities to manage static objects on the map.

Class Methods:
- __init__: Initializes the SidewalkManager with default settings.
- reset: Main function, Populates the map with static objects for each block.
- init_static_adj_list: Initializes the list of static objects' metadata.
- get_num_and_pos: Retrieves the number and positions of each detail object type to be spawned.
- load_json_file: Loads a JSON file and returns its content.
- quickSpawn: Quickly spawns an object on a specified lane at given coordinates.
- does_overlap: Checks if a new object would overlap with existing objects on a lane.
- create_grid: Creates a grid layout for object placement on a lane.
- create_grids_for_lanes: Generates grid layouts for all lanes of a block.
- calculate_sidewalk_lateral_range: Calculates the lateral range for the sidewalk area of a lane.
- calculate_outsidewalk_lateral_range: Calculates the lateral range for the area outside the sidewalk of a lane.
- calculate_nearsidewalk_lateral_range: Calculates the lateral range for the area near the sidewalk of a lane.
- retrieve_and_sort_objects_for_block: Retrieves and sorts objects based on their size for a block.
- fit_objects_to_grids: Fits objects into the generated grids on the lanes.
- place_object_in_grid_if_fits: Attempts to place an object in the grid if it fits without overlapping.
- check_fit_and_place: Checks if an object fits in a specific grid location and places it if it does.
- set_state: Restores the state of spawned objects after environment reset.
"""
import math
import os
from collections import defaultdict
from asset.read_config import configReader
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
from metadrive.component.pgblock.straight import Straight
from metadrive.component.road_network import Road
from metadrive.component.static_object.test_new_object import TestObject, TestGLTFObject
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficWarning, TrafficBarrier
from metadrive.engine.engine_utils import get_engine
from metadrive.manager.base_manager import BaseManager
from metadrive.constants import PGDrivableAreaProperty as DrivableAreaProperty

import json
import random


class GridCell:
    def __init__(self, occupied=False, position=None):
        self.occupied = occupied
        self.position = position

class SidewalkManager(BaseManager):
    """
    This class is used to spawn static objects (currently pedestrian) on the sidewalk
    """
    PRIORITY = 9


    def __init__(self):
        """
        Initializes the SidewalkManager with default settings.
        Sets up configuration readers and initializes static object metadata lists.
        """
        super(SidewalkManager, self).__init__()
        self.debug = True
        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.init_static_adj_list()
        self.get_num_and_pos()
        self.spawned_obj_positions = defaultdict(list)


    def init_static_adj_list(self):
        """
        Initializes the list of static objects' metadata from the adjusted parameter folder.
        It filters out car-related objects, focusing only on static objects.
        """
        self.type_metainfo_dict = defaultdict(list)
        for root, dirs, files in os.walk(self.path_config["adj_parameter_folder"]):
            for file in files:
                if not file.lower().startswith("car"):
                    # print(file)
                    with open(os.path.join(root, file), 'r') as f:
                        loaded_metainfo = json.load(f)
                        self.type_metainfo_dict[loaded_metainfo['general']['detail_type']].append(loaded_metainfo)
    def get_num_and_pos(self):
        """
        Retrieves the number and positions for spawning each type of static object.
        It uses configurations defined in YAML files to determine these values.
        """
        self.num_dict = dict()
        self.pos_dict = dict()
        self.heading_dict = dict()
        for detail_type in self.type_metainfo_dict.keys():
            self.num_dict[detail_type] = self.config.getSpawnNum(detail_type)
            self.pos_dict[detail_type]  = self.config.getSpawnPos(detail_type)
            if self.config.getSpawnHeading(detail_type):
                self.heading_dict[detail_type] = [heading * math.pi for heading in self.config.getSpawnHeading(detail_type)]
            else:
                self.heading_dict[detail_type] = [0, math.pi]
    @staticmethod
    def load_json_file(filepath):
        """
        Loads and returns the content of a JSON file.

        Parameters:
        - filepath (str): The path to the JSON file to be loaded.

        Returns:
        - dict: Content of the JSON file.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    def quickSpawn(self, obj, lane, longitude, lateral):
        """
        Quickly spawns an object at a specified position on a lane.

        Parameters:
        - obj (Object): The object to be spawned.
        - lane (AbstractLane): The lane on which to spawn the object.
        - longitude (float): Longitudinal position on the lane.
        - lateral (float): Lateral position on the lane.
        """
        self.spawn_object(
            obj,
            lane=lane,
            position=lane.position(longitude, lateral),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(longitude)
        )

    def does_overlap(self, lane, new_long, new_lat, new_width, new_length):
        """
        Checks if a new object would overlap with existing objects on a specified lane.

        Parameters:
        - lane (AbstractLane): The lane to check for overlaps.
        - new_long (float): Longitudinal position of the new object.
        - new_lat (float): Lateral position of the new object.
        - new_width (float): Width of the new object.
        - new_length (float): Length of the new object.

        Returns:
        - bool: True if there is an overlap, False otherwise.
        """
        for obj in self.spawned_obj_positions[lane]:
            obj_long, obj_lat, obj_width, obj_length = obj
            if (abs(new_long - obj_long) < (new_width + obj_width) / 2) and \
                    (abs(new_lat - obj_lat) < (new_length + obj_length) / 2):
                return True
        return False
    def create_grid(self, lane_length, lateral_range, cell_size):
        """
        Creates a grid layout for object placement along a lane.

        Parameters:
        - lane_length (float): The length of the lane.
        - lateral_range (tuple): A tuple indicating the lateral range (start, end) for object placement.
        - cell_size (dict): A dictionary specifying the cell size {'length': float, 'width': float}.

        Returns:
        - list[list[GridCell]]: A 2D grid of GridCell objects for object placement.
        """
        # Calculate number of cells along the length and width
        num_cells_long = int(lane_length / cell_size['length'])
        num_cells_lat = int((lateral_range[1] - lateral_range[0]) / cell_size['width'])

        # Create a 2D array of grid cells
        grid = [[GridCell(position=(long_idx * cell_size['length'],
                                    lat_idx * cell_size['width'] + lateral_range[0]))
                 for lat_idx in range(num_cells_lat)] for long_idx in range(num_cells_long)]
        return grid
    def create_grids_for_lanes(self, block):
        """
        Generates grid layouts for each lane of a block for object placement.

        Parameters:
        - block (Block): The block for which to create grids.

        Returns:
        - dict: A dictionary of grids for each lane and each region within the block.
        """
        # Define cell size based on the smallest object or a fixed dimension
        self.cell_size = {'length': 1, 'width': 1}  # Example cell size

        # Create grids for each lane
        grids = {}
        for lane_key in ['positive_basic_lane', 'negative_basic_lane']:
            lane = getattr(block, lane_key)
            grids[lane_key] = {
                'sidewalk': self.create_grid(lane.length,
                                             self.calculate_sidewalk_lateral_range(lane),
                                             self.cell_size),
                'outsidewalk': self.create_grid(lane.length,
                                                self.calculate_outsidewalk_lateral_range(lane),
                                                self.cell_size),
                'nearsidewalk': self.create_grid(lane.length,
                                                 self.calculate_nearsidewalk_lateral_range(lane),
                                                 self.cell_size),
            }
        return grids

    def calculate_sidewalk_lateral_range(self, lane):
        # Calculate the lateral range for the sidewalk
        start_lat = lane.width_at(0) / 2 + 0.2
        return (start_lat,start_lat + DrivableAreaProperty.SIDEWALK_WIDTH)

    def calculate_outsidewalk_lateral_range(self, lane):
        # Calculate the lateral range for the outsidewalk
        return (lane.width_at(0) / 2 + 0.2,
                lane.width_at(0) / 2 + 0.2 + 5 * lane.width)

    def calculate_nearsidewalk_lateral_range(self, lane):
        # Calculate the lateral range for the nearsidewalk
        return (lane.width_at(0) / 2 + 0.2 - lane.width,lane.width_at(0) / 2 + 0.2)
    def retrieve_and_sort_objects_for_block(self, block):
        """
        Retrieves and sorts objects for a block based on size and configuration settings.

        Parameters:
        - block (Block): The block for which to retrieve and sort objects.

        Returns:
        - list[dict]: A list of sorted objects (metadata) for placement.
        """
        # Retrieve objects based on num_dict and sort them
        objects = []
        for detail_type, meta_info_list in self.type_metainfo_dict.items():
            num = self.num_dict[detail_type]
            objects.extend(random.choices(meta_info_list,  k=num))
        # Sort objects by size (length * width)
        objects.sort(key=lambda obj: obj['general']['length'] * obj['general']['width'], reverse=True)
        return objects

    def fit_objects_to_grids(self, block):
        """
        Attempts to fit and place objects into the grids created for each lane of a block.

        Parameters:
        - block (Block): The block for which to fit objects into grids.

        Returns:
        - None
        """
        # Retrieve and sort objects for the block
        all_objects = self.retrieve_and_sort_objects_for_block(block)

        # Create grids for each lane and each region
        grids = self.create_grids_for_lanes(block)
        print("Starting object placement process for new block.")
        # Try fitting objects into grids
        for obj_attribute in all_objects:
            detail_type = obj_attribute['general']['detail_type']
            position = self.pos_dict[detail_type]
            placed = False

            for lane_key, regions_grids in grids.items():
                lane = getattr(block, lane_key)
                if position in regions_grids:
                    grid = regions_grids[position]
                    if self.place_object_in_grid_if_fits(obj_attribute, grid, lane):
                        placed = True
                        break  # Object placed, move to the next object

            if not placed:
                print(f"Could not place object of type {detail_type} in any grid.")

    def place_object_in_grid_if_fits(self, obj_attribute, grid, lane):
        """
        Places an object in a grid if it fits without overlapping other objects.

        Parameters:
        - obj_attribute (dict): Metadata attributes of the object to place.
        - grid (list[list[GridCell]]): The grid in which to attempt placement.
        - lane (AbstractLane): The lane associated with the grid.

        Returns:
        - bool: True if the object was successfully placed, False otherwise.
        """
        obj_length = obj_attribute['general']['length']
        obj_width = obj_attribute['general']['width']
        # Calculate the number of cells the object spans
        cells_span_length = int(obj_length / self.cell_size['length'])
        cells_span_width = int(obj_width / self.cell_size['width'])

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if self.check_fit_and_place(i, j, cells_span_length, cells_span_width, grid, obj_attribute, lane):
                    return True
        return False

    def check_fit_and_place(self, start_i, start_j, span_length, span_width, grid, obj, lane):
        """
        Checks if an object fits in a specific grid location and places it if it does.

        Parameters:
        - start_i (int): Starting index in the grid (longitude).
        - start_j (int): Starting index in the grid (latitude).
        - span_length (int): Number of cells the object spans in length.
        - span_width (int): Number of cells the object spans in width.
        - grid (list[list[GridCell]]): The grid for placement.
        - obj (dict): The object to place.
        - lane (AbstractLane): The lane associated with the grid.

        Returns:
        - bool: True if the object fits and is placed, False otherwise.
        """
        # Check if the object fits in the grid without overlapping
        if start_i + span_length > len(grid) or start_j + span_width > len(grid[0]):
            return False

        # Check for overlap
        for i in range(start_i, start_i + span_length):
            for j in range(start_j, start_j + span_width):
                if grid[i][j].occupied:
                    return False
        print("start i {} start j {}".format(start_i, start_j))
        obj_length = obj['general']['length']
        obj_width = obj['general']['width']
        obj_type = obj['general']['detail_type']
        # Calculate the position for spawning
        spawn_long = start_i * self.cell_size['length']
        spawn_lat = start_j * self.cell_size['width']
        with open('spawned_objects_log.csv', 'a') as file:
            file.write("{}, {},{},{},{},{}\n".format(lane.id,obj_type, spawn_long, spawn_lat, obj_length, obj_width))
        # Place object in the simulation
        self.spawn_object(
            TestObject,
            lane=lane,
            position=lane.position(spawn_long, spawn_lat),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(spawn_long) + obj['general'].get('heading', 0),
            asset_metainfo=obj,
        )

        # Mark cells as occupied
        for i in range(start_i, start_i + span_length):
            for j in range(start_j, start_j + span_width):
                grid[i][j].occupied = True
                grid[i][j].object = obj  # Optionally store the object reference

        return True

    def reset(self):
        """
        Fill the map with static objects for each block. Specifically:
        1. Skips the `FirstPGBlock` type blocks.
        2. Places `TrafficWarning` objects at the start and end of each lane. (serve as boundary checker)
        3. Spawns random pedestrians (or other objects from the obj_json_list) at random positions along the lane.

        Returns:
        - None
        """
        if self.debug:
            with open('spawned_objects_log.csv', 'a') as file:
                file.write("id, obj,long,lat,len,width\n")
        engine = get_engine()
        for block in engine.current_map.blocks:
            if type(block) is FirstPGBlock:
                continue
            # positive lane refers to the current lane, negative refer to the reverse lane on your left
            # for lane in [block.positive_basic_lane, block.negative_basic_lane]:
            for lane in [block.positive_basic_lane]:
                # min range of longtitude
                longitude = 0
                # min range for lateral for the sidewalk
                start_lat = lane.width_at(0) / 2 + 0.2 - lane.width
                end_lat = lane.width_at(0) / 2 + 0.2
                lateral = 0 + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
                # place a trafficwarning to tell the boundary
                self.quickSpawn(TrafficWarning, lane, longitude, start_lat)
                # max range for long
                longitude = lane.length
                # max range for lateral for the sidewalk
                lateral = lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
                self.quickSpawn(TrafficWarning, lane, longitude, end_lat)
                self.fit_objects_to_grids(block)
            break


    def set_state(self, state: dict, old_name_to_current=None):
        """
        Restores the state of spawned objects after an environment reset.

        Parameters:
        - state (dict): A dictionary containing the state information to restore.
        - old_name_to_current (dict, optional): A mapping from old object names to current ones.

        Returns:
        - None
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        if old_name_to_current is None:
            old_name_to_current = {key: key for key in state.keys()}
        spawned_objects = state["spawned_objects"]
        ret = {}
        for name, class_name in spawned_objects.items():
            current_name = old_name_to_current[name]
            name_obj = self.engine.get_objects([current_name])
            assert current_name in name_obj and name_obj[current_name
                                                         ].class_name == class_name, "Can not restore mappings!"
            # Restore some internal states
            name_obj[current_name].lane = self.engine.current_map.road_network.get_lane(
                name_obj[current_name].lane.index
            )

            ret[current_name] = name_obj[current_name]
        self.spawned_objects = ret