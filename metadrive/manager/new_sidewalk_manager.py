# manager that adds items (currently pedestrian) on the sidewalk.
# Note: currently you need to change path in the init function.
import math
import os
from collections import defaultdict

import vqa.vqagen.utils.qa_utils
from asset.read_config import configReader
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.static_object.test_new_object import TestObject
from metadrive.engine.engine_utils import get_engine
from metadrive.manager.base_manager import BaseManager
from metadrive.constants import PGDrivableAreaProperty as DrivableAreaProperty

import json
import random


class GridCell:
    """
    Represents a single cell in a grid, which can be occupied by an object.
    """
    def __init__(self, position, occupied=False):
        """
        Args:
            position (tuple): The position of the cell in the grid, as a tuple (i, j).
            occupied (bool): Whether the cell is occupied by an object.
        """
        self.position = position
        self.occupied = occupied
        self.object = None  # Optional, to reference the object occupying the cell

    def is_occupied(self):
        """
        Check if the cell is occupied by an object.
        returns:
            bool: True if the cell is occupied, False otherwise.
        """
        return self.occupied

    def occupy(self, obj):
        """
        Mark the cell as occupied by an object.
        returns:
            bool: True if the cell is occupied, False otherwise.
        """
        self.occupied = True
        self.object = obj

    def release(self):
        """
        Mark the cell as unoccupied.
        """
        self.occupied = False
        self.object = None


class ObjectPlacer:
    """
    This class is used to place objects on a grid, ensuring that they do not overlap.
    """
    def __init__(self, grid):
        """
        Args:
            grid (list): A 2D list of GridCell objects representing the placement grid.
            
            Example:
            grid = [
                [GridCell(), GridCell(), GridCell()],
                [GridCell(), GridCell(), GridCell()],
                [GridCell(), GridCell(), GridCell()]
            ]
        """
        self.grid = grid
        # Dictionary to store the objects that have been placed, with their positions
        # Keys are tuples (object_id, position), values are tuples (position, object)
        self.placed_objects = {}

    def place_object(self, obj):
        """
        Attempts to place a single object on the grid.

        Args:
            obj (dict): The object to be placed, with properties like length and width.

        Returns:
            bool: True if the object was successfully placed, False otherwise.
        """
        # Find a position where the object can be placed
        position = self.find_placement_position(obj)
        if position:
            # Mark the cells as occupied
            self.mark_occupied_cells(position, obj)
            obj_id = (obj['CLASS_NAME'], position)  # each object has a unique 'id'
            self.placed_objects[obj_id] = (position, obj)
            return True
        return False

    def find_placement_position(self, obj):
        """
        Find a position on the grid where the object can be placed.
        For now, we just return the top-left position where the object can be placed.
        Args:
            obj (dict): The object to be placed, with properties like length and width.
        Returns:
            tuple: The top-left position where the object can be placed, or None if no position is found.
        """
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.can_place(i + 1, j + 1, obj):
                    return (i + 1, j + 1)  # Top-left position where the object can be placed
        return None

    def can_place(self, start_i, start_j, obj):
        """
        Check if the object can be placed starting from the given position.
        The object is placed with additional 2-cell buffer around it.
        Args:
            start_i (int): The starting row index of the grid.
            start_j (int): The starting column index of the grid.
            obj (dict): The object to be placed, with properties like length and width.
        Returns:
            bool: True if the object can be placed, False otherwise.
        """
        # Define the size of each cell (in meters, for example)
        cell_length = 1  # Length of each cell in meters
        cell_width = 1  # Width of each cell in meters

        # Calculate the number of cells the object spans, rounding up
        # Note we add 2 to the span to create a buffer around the object
        span_length = math.ceil(obj['general']['length'] / cell_length) + 2
        span_width = math.ceil(obj['general']['width'] / cell_width) + 2
        # span_length = math.ceil(obj['general']['width'] / cell_width) + 2
        # span_width = math.ceil(obj['general']['length'] / cell_length) + 2

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
        """
        Mark the cells occupied by the object, with a buffer of 2 cells around it.
        Args:
            start_position (tuple): The top-left position where the object is placed.
            obj (dict): The object to be placed, with properties like length and width.
        Returns:
            Nothing, directly marks the cells as occupied.
        """
        start_i, start_j = start_position
        cell_length = 1  # Length of each cell in meters
        cell_width = 1  # Width of each cell in meters

        # Calculate the number of cells the object spans, rounding up
        span_length = math.ceil(obj['general']['length'] / cell_length) + 2
        span_width = math.ceil(obj['general']['width'] / cell_width) + 2

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
    This class is used to spawn static objects on the sidewalk
    The main idea is to create a grid for each region of the sidewalk (e.g., onsidewalk, outsidewalk, nearsidewalk)
    The main entry point is the reset method, which is called at the beginning of each episode.
    """
    PRIORITY = 9

    def __init__(self):
        super(SidewalkManager, self).__init__()
        self.debug = True
        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.init_static_adj_list()  # Load the metainfo for all static objects
        self.get_num_and_pos()  # Get the number and position of objects to spawn

    def init_static_adj_list(self):
        """
        Load the metainfo for all static objects
        """
        # The dictionary to store the metainfo for each object type
        # The key is the detail type, the value is a list of metainfo dictionaries
        # For example, key is bicycle, value is a list of metainfo dictionaries for all bicycle objects
        self.type_metainfo_dict = defaultdict(list)
        for root, dirs, files in os.walk(self.path_config["adj_parameter_folder"]):
            for file in files:
                # We only load the metainfo for static objects, skip cars
                if not file.lower().startswith("car"):
                    # print(file)
                    with open(os.path.join(root, file), 'r') as f:
                        loaded_metainfo = json.load(f)
                        self.type_metainfo_dict[loaded_metainfo['general']['detail_type']].append(loaded_metainfo)

    def get_num_and_pos(self):
        """
        Get the number and position of objects to spawn
        """
        # The dictionary to store the number of objects to spawn for each type
        self.num_dict = dict()
        # The dictionary to store the position of objects to spawn for each type
        self.pos_dict = dict()
        # The dictionary to store the heading (rotation) of objects to spawn for each type
        self.heading_dict = dict()
        for detail_type in self.type_metainfo_dict.keys():
            self.num_dict[detail_type] = self.config.getSpawnNum(detail_type)
            self.pos_dict[detail_type] = self.config.getSpawnPos(detail_type)
            if self.config.getSpawnHeading(detail_type):
                self.heading_dict[detail_type] = [
                    heading * math.pi for heading in self.config.getSpawnHeading(detail_type)
                ]
            else:
                self.heading_dict[detail_type] = [0, math.pi]

    @staticmethod
    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def before_reset(self):
        """
        Update episode level config to this manager and clean element or detach element
        """
        items = self.clear_objects([object_id for object_id in self.spawned_objects.keys()])
        self.spawned_objects = {}

    def reset(self):
        """
        Reset the manager and spawn objects on the sidewalk.
        Main entry point for the manager.
        """
        super(SidewalkManager, self).reset()
        self.count = 0
        engine = get_engine()
        assert len(self.spawned_objects.keys()) == 0
        # Iterate over all blocks in the current map (The blocks are the straight road segments in the map)
        for block in engine.current_map.blocks:
            if isinstance(block, FirstPGBlock):
                continue
            # Iterate over both lanes in the block (Each block has a positive and negative lane, representing the two directions of traffic)
            for lane in [block.positive_basic_lane, block.negative_basic_lane]:
                # for lane in [block.positive_basic_lane]:
                # Create grids for each region
                sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("onsidewalk", lane))
                outsidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("outsidewalk", lane))
                nearsidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearsidewalk", lane))
                print(self.calculate_lateral_range("outsidewalk", lane))
                # Retrieve and place objects for each region
                for region, grid in [('onsidewalk', sidewalk_grid), ('outsidewalk', outsidewalk_grid),
                                     ('nearsidewalk', nearsidewalk_grid)]:
                    # Create an ObjectPlacer object to place objects on the grid
                    object_placer = ObjectPlacer(grid)
                    # Place objects on the virtual grid, but not actually spawned yet
                    self.retrieve_objects_for_region(region, object_placer)
                    print("======For region:{}=======".format(region))
                    # self.visualize_grid(grid)
                    # Iterate over the placed objects and spawn them in the simulation
                    for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                        # Convert the grid position to a lane position
                        lane_position = self.convert_grid_to_lane_position(
                            grid_position, lane, self.calculate_lateral_range(region, lane)
                        )
                        self.count += 1
                        self.spawn_object(
                            TestObject,
                            force_spawn=True,
                            lane=lane,
                            position=lane_position,
                            static=self.engine.global_config["static_traffic_object"],
                            heading_theta=lane.heading_theta_at(lane_position[0]) + vqa.vqagen.utils.qa_utils.get('heading', 0),
                            asset_metainfo=obj
                        )

        print("Spawned {} objects".format(self.count))

    def create_grid(self, lane, lateral_range):
        """
        Create a grid for a given lane and lateral range.
        Args:
            lane (Lane): The lane object.
            lateral_range (tuple): The start and end of the lateral range for the grid.
        Returns:
            list: A 2D list of GridCell objects representing the grid."""
        # Define the size of each cell (in meters, for example)
        cell_length = 1  # Length of a cell along the lane
        cell_width = 1  # Width of a cell across the lane

        # Calculate the number of cells along the lane and across its width
        num_cells_long = int(lane.length / cell_length)
        num_cells_lat = int((lateral_range[1] - lateral_range[0]) / cell_width)

        # Create the grid as a 2D array of GridCell objects
        grid = [
            [GridCell(position=(i * cell_length, j * cell_width + lateral_range[0])) for j in range(num_cells_lat)]
            for i in range(num_cells_long)
        ]

        return grid

    def retrieve_objects_for_region(self, region, object_placer):
        """
        Retrieve objects for a given region and place them on the grid.
        Args:
            region (str): The region (e.g., 'onsidewalk', 'nearsidewalk', 'outsidewalk').
            object_placer (ObjectPlacer): The ObjectPlacer object to place objects on the grid.
        Returns:
            Nothing, directly places objects on the grid.
        """
        # Group objects by detail type and initialize counters for each type
        detail_type_groups = defaultdict(list)
        object_counts = defaultdict(int)  # Track how many objects of each type have been tried
        # iterate over all objects and group them by detail type
        for detail_type, objects in self.type_metainfo_dict.items():
            # Note, we only place objects in the specified region
            if self.pos_dict[detail_type] == region:
                for idx, obj in enumerate(objects):
                    unique_id = (detail_type, idx)
                    detail_type_groups[detail_type].append(unique_id)

        # Set to keep track of tried objects
        any_object_placed = True
        # Continue round-robin placement until all objects are tried or count limit is reached
        while any_object_placed:
            any_object_placed = False
            # Iterate over all detail types and try to place objects
            for detail_type, object_ids in detail_type_groups.items():
                # If we have already placed the required number of objects, skip
                if object_counts[detail_type] < self.num_dict[detail_type]:
                    # Randomly select an object to place
                    obj_id = random.sample(object_ids, 1)[0]
                    obj = self.type_metainfo_dict[detail_type][obj_id[1]]  # Retrieve the actual object
                    if object_placer.place_object(obj):
                        object_counts[detail_type] += 1
                        any_object_placed = True
            if not any_object_placed and all(object_counts[dt] >= self.num_dict[dt] for dt in detail_type_groups):
                break

    def convert_grid_to_lane_position(self, grid_position, lane, lateral_range):
        """
        Convert a grid position to a lane position.
        Args:
            grid_position (tuple): The grid position as a tuple (i, j).
            lane (Lane): The lane object.
            lateral_range (tuple): The start and end of the lateral range for the grid.
        Returns:
            tuple: The lane position as a tuple (longitude, lateral).
        """
        grid_i, grid_j = grid_position
        cell_length = 1  # Length of a cell along the lane, should be consistent with create_grid method
        cell_width = 1  # Width of a cell across the lane, should be consistent with create_grid method

        # Convert grid position to longitudinal and lateral position relative to the lane
        longitude = grid_i * cell_length
        lateral = lateral_range[0] + grid_j * cell_width

        return lane.position(longitude, lateral)

    def visualize_grid(self, grid):
        """
        Visualize the grid by printing it to the console.
        Args:
            grid (list): A 2D list of GridCell objects representing the grid.
        Returns:
            Nothing, directly prints the grid to the console.
        """
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
            return (
                lane.width_at(0) / 2 + 0.2 + DrivableAreaProperty.SIDEWALK_WIDTH + 1,
                lane.width_at(0) / 2 + 0.2 + DrivableAreaProperty.SIDEWALK_WIDTH + 1 + 5 * lane.width
            )

        elif region == 'nearsidewalk':
            return (lane.width_at(0) / 2 + 0.2 - lane.width, lane.width_at(0) / 2 + 0.2)

        else:
            raise ValueError("Unknown region type")
