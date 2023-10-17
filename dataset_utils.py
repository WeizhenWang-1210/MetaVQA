from typing import Iterable



def dot(v1: Iterable[float], v2:Iterable[float])->float:
    """
    Dot Product of vector v1 and v2.
    """
    assert len(v1) == len(v2), "Mismatched Dimension of v1 and v2!"
    ans = 0
    for l,r in zip(v1,v2):
        ans += l*r
    return ans

def find_extremities(ref_heading: Iterable[float], 
                     bboxes: Iterable[Iterable], center: Iterable[float])->tuple(Iterable[float]):
    """
    Find the two vertice of bbox that are the extremeties along the provided positive axis.
    """
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
    