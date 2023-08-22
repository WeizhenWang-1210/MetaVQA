import numpy as np
def sample_bbox(bbox,height,x_times, y_times, z_times):
    #BBOX: top left -> top right ->bottom right ->top left
    #divide the bbox to points with x_times along width, y_times along length, and z times along height.
    #(x,y,z)
    array_bbox = [np.asarray(point) for point in bbox]
    seg_x = array_bbox[1]-array_bbox[0]
    seg_y = array_bbox[3]-array_bbox[0]
    seg_z = np.array([0,0,height])
    base = array_bbox[0]
    step_x = seg_x/(x_times-1)
    step_y = seg_y/(y_times-1)
    step_z = seg_z/(z_times-1)
    result = []
    for idz in range(0,z_times):
        for idy in range(0,y_times):
            for idx in range(0,x_times):
                new_point = base + idx * step_x + idy * step_y + idz*step_z
                #print(idz,idy,idx,new_point)
                result.append(list(new_point))
    return result
"""def visible(cam, sampled):
    visible_points = 0
    num_points = len(sampled)
    for point in sampled:
        if cam.isInView(np.asarray(point)-cam.pos)





    return (visible_points/num_points) >= 0.1"""

