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



"""
suppose you have a frame h,w,3
for each pixel, you can get a (c,c,c)
this means the frag captured in this pixel has distance 2^(c*4)*8 units to the camera
in addition, there exist transform T(pos, heading,fov, near, far): R^2->R^3 that restore the 3d position of the pixel back to 3d space. 
And these points consist of the point-cloud representations of the lidar's observation.
We also need to findout the actual objects that got hit and generated such point cloud.
void reverse(){
  code = fragColor[0]
  distance_to_camera = 2^(c * log(b))*base
}

"""
