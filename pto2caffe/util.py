import numpy as np


dim_map_nhwc2nchw = [0, 2, 3, 1]


trans_dtype = {
    'u8': np.uint8,
    's8': np.int8,
    's16': np.int16,
    'f32': np.float32
}


dtype_map = {
    'u8': 0,
    's16': 1,
    'f32': 2
}


def shape_map_nhwc2nchw(shape):
    if isinstance(shape, np.ndarray):
        if len(shape) == 4:
            return [shape[0], shape[3], shape[1], shape[2]]
        elif len(shape) == 3:
            return [shape[0], shape[1], shape[2]]
        elif len(shape) == 2:
            return [shape[0], shape[1]]
        elif len(shape) == 1:
            return [shape[0]]
        else:
            print(shape, shape.size, len(shape))
            raise NotImplementedError(shape)
    else:
        return [shape]
