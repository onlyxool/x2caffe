dim_map_nhwc2nchw = [0, 2, 3, 1]

def shape_map_nhwc2nchw(shape:list):
    if len(shape) == 4:
        return [shape[0], shape[3], shape[1], shape[2]]
    else:
        return shape
