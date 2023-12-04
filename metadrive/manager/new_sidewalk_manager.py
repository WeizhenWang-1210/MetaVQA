# manager that adds items (currently pedestrian) on the sidewalk.
# Note: currently you need to change path in the init function.
import math
import os
from collections import defaultdict
from random import sample
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
    def __init__(self, position, occupied=False):
        self.position = position
        self.occupied = occupied
        self.object = None  # Optional, to reference the object occupying the cell

    def is_occupied(self):
        return self.occupied

    def occupy(self, obj):
        self.occupied = True
        self.object = obj

    def release(self):
        self.occupied = False
        self.object = None
class ObjectPlacer:
    def __init__(self, grid):
        self.grid = grid
        self.placed_objects = {}

    def place_object(self, obj):
        """
        Attempts to place a single object on the grid.

        Args:
            obj (dict): The object to be placed, with properties like length and width.

        Returns:
            bool: True if the object was successfully placed, False otherwise.
        """
        position = self.find_placement_position(obj)
        if position:
            self.mark_occupied_cells(position, obj)
            obj_id = (obj['CLASS_NAME'], position)  # Assuming each object has a unique 'id'
            self.placed_objects[obj_id] = (position, obj)
            return True
        return False

    def find_placement_position(self, obj):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.can_place(i, j, obj):
                    return (i+1, j+1)  # Top-left position where the object can be placed
        return None

    def can_place(self, start_i, start_j, obj):
        cell_length = 1  # Length of each cell in meters
        cell_width = 1  # Width of each cell in meters

        # Calculate the number of cells the object spans, rounding up
        span_length = math.ceil(obj['general']['length'] / cell_length) + 2
        span_width = math.ceil(obj['general']['width'] / cell_width) + 2
        # span_length = math.ceil(obj['general']['width'] / cell_width)
        # span_width = math.ceil(obj['general']['length'] / cell_length)

        # Check if the object fits within the grid bounds
        if start_i + span_length > len(self.grid) or start_j + span_width > len(self.grid[0]):
            return False

        # Check for any overlaps with existing objects
        for i in range(start_i, start_i + span_length):
            for j in range(start_j, start_j + span_width):
                if self.grid[i][j].is_occupied():
                    return False

        return True

    def mark_occupied_cells(self, start_position, obj):
        start_i, start_j = start_position
        cell_length = 1  # Length of each cell in meters
        cell_width = 1  # Width of each cell in meters

        # Calculate the number of cells the object spans, rounding up
        span_length = math.ceil(obj['general']['length'] / cell_length)
        span_width = math.ceil(obj['general']['width'] / cell_length)

        # Mark the occupied cells
        for i in range(start_i, start_i + span_length):
            for j in range(start_j, start_j + span_width):
                self.grid[i][j].occupy(obj)

    def is_placement_possible(self):
        """
        Optional: Checks if there is any space left on the grid to place any object.

        Returns:
            bool: True if there is space available, False otherwise.
        """
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if not self.grid[i][j].is_occupied():
                    return True
        return False

class SidewalkManager(BaseManager):
    """
    This class is used to spawn static objects (currently pedestrian) on the sidewalk
    """
    PRIORITY = 9


    def __init__(self):
        super(SidewalkManager, self).__init__()
        self.debug = True
        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.init_static_adj_list()
        self.get_num_and_pos()
        self.spawned_obj_positions = defaultdict(list)


    def init_static_adj_list(self):
        self.type_metainfo_dict = defaultdict(list)
        for root, dirs, files in os.walk(self.path_config["adj_parameter_folder"]):
            for file in files:
                if not file.lower().startswith("car"):
                    # print(file)
                    with open(os.path.join(root, file), 'r') as f:
                        loaded_metainfo = json.load(f)
                        self.type_metainfo_dict[loaded_metainfo['general']['detail_type']].append(loaded_metainfo)
    def get_num_and_pos(self):
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
        with open(filepath, 'r') as f:
            return json.load(f)

    def reset(self):
        engine = get_engine()

        for block in engine.current_map.blocks:
            if isinstance(block, FirstPGBlock):
                continue

            for lane in [block.positive_basic_lane, block.negative_basic_lane]:
                # Create grids for each region
                sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("onsidewalk", lane))
                outsidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("outsidewalk", lane))
                nearsidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearsidewalk", lane))

                # Retrieve and place objects for each region
                for region, grid in [('onsidewalk', sidewalk_grid), ('outsidewalk', outsidewalk_grid),
                                     ('nearsidewalk', nearsidewalk_grid)]:
                    object_placer = ObjectPlacer(grid)
                    self.retrieve_objects_for_region(region, object_placer)
                    print("======For region:{}=======".format(region))
                    self.visualize_grid(grid)
                    for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                        lane_position = self.convert_grid_to_lane_position(grid_position, lane,
                                                                           self.calculate_lateral_range(region, lane))
                        self.spawn_object(
                            TestObject,
                            lane=lane,
                            position=lane_position,
                            static=self.engine.global_config["static_traffic_object"],
                            heading_theta=lane.heading_theta_at(lane_position[0]) + obj['general'].get(
                                'heading', 0),
                            asset_metainfo=obj
                        )


    def create_grid(self, lane, lateral_range):
        # Define the size of each cell (in meters, for example)
        cell_length = 1  # Length of a cell along the lane
        cell_width = 1  # Width of a cell across the lane

        # Calculate the number of cells along the lane and across its width
        num_cells_long = int(lane.length / cell_length)
        num_cells_lat = int((lateral_range[1] - lateral_range[0]) / cell_width)

        # Create the grid as a 2D array of GridCell objects
        grid = [[GridCell(position=(i * cell_length, j * cell_width + lateral_range[0]))
                 for j in range(num_cells_lat)] for i in range(num_cells_long)]

        return grid


    def retrieve_objects_for_region(self, region, object_placer):
        # Group objects by detail type and initialize counters for each type
        detail_type_groups = defaultdict(list)
        object_counts = defaultdict(int)  # Track how many objects of each type have been tried
        for detail_type, objects in self.type_metainfo_dict.items():
            if self.pos_dict[detail_type] == region:
                for idx, obj in enumerate(objects):
                    unique_id = (detail_type, idx)
                    detail_type_groups[detail_type].append(unique_id)

        # Set to keep track of tried objects
        any_object_placed = True
        # Continue round-robin placement until all objects are tried or count limit is reached
        while any_object_placed:
            any_object_placed = False
            for detail_type, object_ids in detail_type_groups.items():
                if object_counts[detail_type] < self.num_dict[detail_type]:
                    obj_id = random.sample(object_ids, 1)[0]
                    obj = self.type_metainfo_dict[detail_type][obj_id[1]]  # Retrieve the actual object
                    if object_placer.place_object(obj):
                        object_counts[detail_type] += 1
                        any_object_placed = True
            if not any_object_placed and all(object_counts[dt] >= self.num_dict[dt] for dt in detail_type_groups):
                break

    def convert_grid_to_lane_position(self, grid_position, lane, lateral_range):
        grid_i, grid_j = grid_position
        cell_length = 1  # Length of a cell along the lane, should be consistent with create_grid method
        cell_width = 1   # Width of a cell across the lane, should be consistent with create_grid method

        # Convert grid position to longitudinal and lateral position relative to the lane
        longitude = grid_i * cell_length
        lateral = lateral_range[0] + grid_j * cell_width

        return lane.position(longitude, lateral)
    def visualize_grid(self, grid):
        for row in grid:
            for cell in row:
                # Assuming each cell has a method 'is_occupied' to check if it's occupied
                char = 'X' if cell.is_occupied() else '.'
                print(char, end=' ')
            print()  # Newline after each row
    def calculate_lateral_range(self, region, lane):
        """
        Calculate the lateral range for a given region of a lane.

        Args:
            region (str): The region (e.g., 'sidewalk', 'nearsidewalk', 'outsidewalk').
            lane (Lane): The lane object.

        Returns:
            tuple: A tuple representing the start and end of the lateral range.
        """
        if region == 'onsidewalk':
            start_lat = lane.width_at(0) / 2 + 0.2
            return (start_lat, start_lat + DrivableAreaProperty.SIDEWALK_WIDTH)

        elif region == 'outsidewalk':
            return (lane.width_at(0) / 2 + 0.2 + DrivableAreaProperty.SIDEWALK_WIDTH,
                    lane.width_at(0) / 2 + 0.2 + DrivableAreaProperty.SIDEWALK_WIDTH + 5 * lane.width)

        elif region == 'nearsidewalk':
            return (lane.width_at(0) / 2 + 0.2 - lane.width,lane.width_at(0) / 2 + 0.2)

        else:
            raise ValueError("Unknown region type")
