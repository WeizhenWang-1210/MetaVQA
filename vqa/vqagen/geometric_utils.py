import numpy as np


def get_distance(pos_1: list, pos_2: list) -> float:
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)
