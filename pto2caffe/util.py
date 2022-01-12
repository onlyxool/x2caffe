import math
import numpy as np
import tensorflow as tf


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


def transpose(tensor):
    if len(tensor.shape) == 4:
        return tensor.transpose(3, 0, 1, 2)
    else:
        return tensor


def np_nhwc2nchw(array):
    if len(array.shape) == 4:
        return array.transpose(3, 0, 1, 2)
    else:
        return array


def shape_map_nhwc2nchw(shape):
    if isinstance(shape, np.ndarray) or isinstance(shape, list): #TFLite
        if len(shape) == 4:
            return [shape[0], shape[3], shape[1], shape[2]]
        elif len(shape) == 3:
            return [shape[0], shape[1], shape[2]]
        elif len(shape) == 2:
            return [shape[0], shape[1]]
        elif len(shape) == 1:
            return [shape[0]]
        elif len(shape) == 0:
            return []
        else:
            print(shape, shape.size, len(shape))
            raise NotImplementedError(shape)
    elif isinstance(shape, tf.TensorShape): # Tensorflow Frozen Graph
        if shape.rank == None:
            return None
        elif shape.rank == 4:
            return [shape.as_list()[0], shape.as_list()[3], shape.as_list()[1], shape.as_list()[2]]
        elif shape.rank >= 0 and shape.rank <= 3:
            return shape.as_list()
        else:
            print('Shape Error:', shape)
            raise NotImplementedError
    else:
        return [shape]

# Padding
def computePaddingSize(input, output, stride, kernel, dilation, proto_param, layer_type):
    pad = 0
    if layer_type == 'Convolution' or layer_type == 'ConvolutionDepthwise':
        pad = (output - 1) * stride - input + (dilation * (kernel - 1) + 1)
    elif layer_type == 'Deconvolution':
        pad = (input - 1) * stride + (kernel - 1) * dilation + 1 - output
    elif layer_type == 'Pooling':
        if proto_param['ceil_mode']:
            if stride <= kernel:
                pad = (output - 1) * stride - input + kernel
            else:
                pad = (output - 0) * stride - input
        else:
            if stride <= kernel:
                pad = (output - 1) * stride - input + kernel
            else:
                pad = output * stride - input

    return pad if pad >= 0 else 0


def asymmetric_pad(pad):
    if math.modf(pad / 2)[0] != 0:
        pad_b = math.ceil(pad / 2)
        pad_f = pad - pad_b
    else:
        pad_f = pad / 2
        pad_b = pad / 2

    return pad_f, pad_b


def handleLegacyPad(padding_mode, input_size, output_size, proto_param:dict, legacy_pad, layer_type):
    if padding_mode == 'VALID':
        if legacy_pad['left'] == legacy_pad['right'] and legacy_pad['top'] == legacy_pad['bottom']:
            return (int(legacy_pad['left']), int(legacy_pad['top']))
        else:
            return (int(legacy_pad['left']), int(legacy_pad['right']), int(legacy_pad['top']), int(legacy_pad['bottom']))

    # Horizontal
    input_h = input_size[2]
    output_h = output_size[2]
    kernel_h = proto_param['kernel_h']
    stride_h = proto_param['stride_h']
    dilation_h = proto_param.get('dilation', [1,1])[0]

    pad_h = computePaddingSize(input_h, output_h, stride_h, kernel_h, dilation_h, proto_param, layer_type)
    pad_t, pad_b = asymmetric_pad(pad_h)

    # Vertical
    input_w = input_size[3]
    output_w = output_size[3]
    kernel_w = proto_param['kernel_w']
    stride_w = proto_param['stride_w']
    dilation_w = proto_param.get('dilation', [1,1])[1]

    pad_w = computePaddingSize(input_w, output_w, stride_w, kernel_w, dilation_w, proto_param, layer_type)
    pad_l, pad_r = asymmetric_pad(pad_w)

    # Asymmetric Pad
    pad_l = int(pad_l + legacy_pad['left'])
    pad_r = int(pad_r + legacy_pad['right'])
    pad_t = int(pad_t + legacy_pad['top'])
    pad_b = int(pad_b + legacy_pad['bottom'])

    if pad_l == pad_r and pad_t == pad_b:
        return (pad_l, pad_t)
    else:
        return (pad_l, pad_r, pad_t, pad_b)


# Scale Axis
def trim_one(scale_shape):
    if scale_shape == [] or scale_shape is None:
        return scale_shape

    # Remove 1 from head
    while True:
        if scale_shape[0] == 1:
            scale_shape.remove(1)
        else:
            break

    # Remove 1 from tail
    while True:
        if scale_shape[-1] == 1:
            scale_shape.pop()
        else:
            break

    return scale_shape


def compute_scale_axis(bottom_shape, scale_shape):
    '''
    The first axis of bottom[0] (the first input Blob) along which to apply
    bottom[1] (the second input Blob).  May be negative to index from the end
    (e.g., -1 for the last axis).

    For example, if bottom[0] is 4D with shape 100x3x40x60, the output
    top[0] will have the same shape, and bottom[1] may have any of the
    following shapes (for the given value of axis):
       (axis == 0 == -4) 100; 100x3; 100x3x40; 100x3x40x60
       (axis == 1 == -3)          3;     3x40;     3x40x60
       (axis == 2 == -2)                   40;       40x60
       (axis == 3 == -1)                                60
    Furthermore, bottom[1] may have the empty shape (regardless of the value of
    "axis") -- a scalar multiplier.
    '''
    if scale_shape == []:
        return 0

    shapeA = np.array(bottom_shape)
    shapeB = np.array(scale_shape)

    for i in range(len(shapeA)):
        shape_map = (shapeA[i:(len(shapeB)+i)] == shapeB)
        shape_map = list(shape_map)
        if isinstance(shape_map, type(np.array)) and shape_map.count(True) == len(shapeB):
            return i

    return None
