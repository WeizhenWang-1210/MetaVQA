from typing import Iterable
import numpy as np



def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def find_extremities(ref_heading, bboxes, center):
    recentered_bboxes = [[bbox[0] - center[0], bbox[1]-center[1]] for bbox in bboxes]
    max_dot, min_dot = float("-inf"), float("inf")
    left_bbox,right_bbox = bboxes[0], bboxes[1]
    for bbox in recentered_bboxes:
        dotp = dot(bbox, ref_heading)
        if  dotp > max_dot:
            left_bbox = bbox
            max_dot = dotp
        if dotp < min_dot:
            right_bbox = bbox
            min_dot = dotp
    left_bbox[0] += center[0]
    left_bbox[1] += center[1]
    right_bbox[0] += center[0]
    right_bbox[1] += center[1]
    
    return left_bbox, right_bbox
    