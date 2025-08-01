import numpy as np

from metadrive.base_class.base_object import BaseObject


def l2_distance(o1: BaseObject, o2: BaseObject) -> float:
    """
    Return the l2 distance between two MetaDrive BaseObject
    """
    pos_1, pos_2 = o1.position, o2.position
    pos_1, pos_2 = pos_1.tolist(), pos_2.tolist()
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)


