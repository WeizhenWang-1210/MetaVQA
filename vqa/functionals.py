from collections import defaultdict
from typing import Iterable, Callable
from vqa.object_node import ObjectNode, transform, TemporalNode


def color_wrapper(colors: Iterable[str]) -> Callable:
    '''
    Constructor for a function that return all nodes with color in colors
    '''

    def color(candidates: Iterable[ObjectNode | TemporalNode]):
        results = []
        for candidate in candidates:
            # for TemporalNode, will invoke the property decorator.
            if candidate.visible and candidate.color in colors:
                results.append(candidate)
        return results

    return color


def type_wrapper(types: Iterable[str]) -> Callable:
    '''
    Constructor for a function that return all nodes with type in types or is a subtype of type in types
    '''

    def type(candidates: Iterable[ObjectNode | TemporalNode]):
        # print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            # print(candidate)
            for t in types:
                if candidate.type == t or subclass(candidate.type, t):
                    results.append(candidate)
                    break
        return results

    return type


def subclass(class1: str, class2: str) -> bool:
    """
    determine if class1 is the subclass of class2
    """

    def get_inheritance() -> defaultdict:
        """
        Return a lineage tree as a dictionary
        """
        import yaml
        with open("./asset_config.yaml", "r") as stream:
            tree = yaml.safe_load(stream)["type"]

        inheritance = defaultdict(lambda: [])

        def get_non_leaf_nodes(d, lineage, parent_key='', ):
            non_leaf_nodes = []
            for key, value in d.items():
                # Construct a full key path if you are in a nested dictionary
                full_key = parent_key + '.' + key if parent_key else key
                if isinstance(value, dict):
                    lineage[parent_key].append(key)
                    non_leaf_nodes.append(full_key)
                    # Recursively search for non-leaf nodes
                    non_leaf_nodes.extend(get_non_leaf_nodes(value, lineage, key))
            return non_leaf_nodes

        get_non_leaf_nodes(tree, inheritance)
        return inheritance

    inheritance = get_inheritance()  # inheritance is not a tree. But, it's a DAG from supertype to subtype(like your typing system in C++)
    if class1 == class2:
        return True
    result = False
    for child in inheritance[class2]:
        result = result or subclass(class1, child)
    return result


def state_wrapper(states: Iterable[str]) -> Callable:
    """
    Constructor for a function that return all nodes with one state in states
    """

    def state(candidates: Iterable[TemporalNode]):
        results = []
        for candidate in candidates:
            for s in states:
                if s in candidate.actions:
                    results.append(candidate)
        return results

    return state


def action_wrapper(egos: Iterable[TemporalNode], actions: Iterable[str]) -> Callable:
    """
    Return a function that takes an iterable of ObjectNode and return an iterable of ObjectNode satisfying the
    given actions, or an empty Iterable if no actions are satisfied.
    """

    def act(candidates: Iterable[TemporalNode]):
        results = []
        for candidate in candidates:
            # print(candidate)
            for ego in egos:
                for action in actions:
                    # if candidate performed action against one ego
                    if ego in candidate.interactions[action]:
                        results.append(candidate)
        return results

    return act


def pos_wrapper(egos: Iterable[ObjectNode], spatial_relationships: Iterable[str],
                ref_heading: tuple = None) -> Callable:
    """
    A constructor for selecting all nodes that exhibit spatial_relationship with any ego in egos for
    spatial_relationship in spatial_relationships. ref_heading is provided to define what's left v.s. right
    """

    def pos(candidates: Iterable[ObjectNode]):

        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            for ego in egos:
                if ego.id != candidate.id and ego.compute_relation_string(candidate,
                                                                          ref_heading) in spatial_relationships:
                    results.append(candidate)
        return results

    return pos


def greater(A, B) -> bool:
    '''
    checker
    '''
    return A > B


def count(stuff: Iterable) -> int:
    '''
    checker
    '''
    return [len(s) for s in stuff]


def CountGreater(search_spaces) -> bool:
    """
    Return True if the first set in the search_spaces has greater length than the second set.
    """
    assert len(search_spaces) == 2, "CountGreater should have only two sets to work with"
    nums = count(search_spaces)  # [count(search_space) for search_space in search_spaces]
    return greater(nums[0], nums[1])


def CountEqual(search_spaces) -> bool:
    """
    Return True if all sets in search_spaces have the same length.
    """
    nums = count(search_spaces)  # [count(search_space) for search_space in search_spaces]
    first = nums[0]
    for num in nums:
        if num != first:
            return False
    return True


def CountLess(search_spaces) -> bool:
    """
    Return True if the first set in the search_spaces has greater smaller than the second set.
    """
    assert len(search_spaces) == 2, "CountGreater should have only two sets to work with"
    nums = count(search_spaces)  # [count(search_space) for search_space in search_spaces]
    return greater(nums[1], nums[0])


def Describe(search_spaces) -> str:
    """
    List all items in the search_space
    """
    search_spaces = search_spaces[0]
    if len(search_spaces) == 0:
        return "No, there is not any item with specified action"
    result = "Yes, there is "
    result += search_spaces[0].color
    result += " "
    result += search_spaces[0].type
    if len(search_spaces) == 1:
        return result
    else:
        for node in search_spaces[1:]:
            result += " and "
            result += node.color
            result += " "
            result += node.type
        result += '.'
    return result


def Identity(search_spaces):
    '''
    Return the singleton answer in search spaces
    '''
    return search_spaces[0]


def locate_wrapper(origin: ObjectNode) -> Callable:
    def locate(stuff: Iterable[ObjectNode]) -> Iterable:
        """
        Return the bbox of all AgentNodes in stuff.
        """
        result = []
        for s in stuff:
            for more_stuff in s:
                transformed = transform(origin, more_stuff.bbox)
                result.append(transformed)
        return result

    return locate


def extract_color(search_spaces):
    result = set()
    for search_space in search_spaces:
        result.update([obj.color for obj in search_space])
    return list(result)


def extract_color_unique(search_spaces):
    result = {}
    for search_space in search_spaces:
        for obj in search_space:
            result[obj.id] = obj.color
    return result


def extract_type(search_spaces):
    result = set()
    for search_space in search_spaces:
        result.update([obj.type for obj in search_space])
    return list(result)


def extract_type_unique(search_spaces):
    result = {}
    for search_space in search_spaces:
        for obj in search_space:
            result[obj.id] = obj.type
    return result


def motion_Pred(search_spaces):
    future_dict = {}
    for search_space in search_spaces:
        for obj in search_space:
            future_dict[obj.id] = obj.get


def is_stationary(search_spaces):
    result = {}
    for search_space in search_spaces:
        for object in search_space:
            result[object.id] = "parked" in object.actions
    return result


def is_turning(search_spaces):
    result = {}
    for search_space in search_spaces:
        for object in search_space:
            result[object.id] = ("turn_left" in object.actions) or ("turn_right" in object.actions)
    return result


def accelerated(search_spaces):
    result = {}
    for search_space in search_spaces:
        for object in search_space:
            result[object.id] = "accelerated" in object.actions
    return result


def identify_speed(search_spaces):
    result = {}
    for search_space in search_spaces:
        for object in search_space:
            result[object.id] = object.speed
    return result


def identify_heading(search_spaces):
    result = {}
    for search_space in search_spaces:
        for object in search_space:
            result[object.id] = object.heading
    return result


def identify_head_toward(ego):
    def helper(search_spaces):
        result = {}
        for search_space in search_spaces:
            for object in search_space:
                result[object.id] = ego.id in object.interactions["head_toward"]
        return result

    return helper


def predict_trajectory(now_frame):
    def helper(search_spaces):
        result = {}
        for search_space in search_spaces:
            for object in search_space:
                result[object.id] = object.future_bboxes(now_frame)
        return result

    return helper
