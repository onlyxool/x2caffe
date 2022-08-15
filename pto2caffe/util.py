import math
import numpy as np


dim_map_nhwc2nchw = [0, 2, 3, 1]


def get_layout(shape):
    if not isinstance(shape, list) and not isinstance(shape, tuple):
        raise NotImplementedError

    if not all(shape):
        return None

    diff = np.abs(np.ediff1d(np.array(shape)))
    idx = diff.argmin()

    if len(shape) == 4:
        if idx == 1:
            return 'NHWC'
        elif idx == 2:
            return 'NCHW'
        else:
            return None
    elif len(shape) == 3:
        return 'HWX' if idx == 0 and diff.min() < np.array(shape).min() else 'XHW'
    elif len(shape) == 5:
        return None
    else:
        return None


def shape_map_nhwc2nchw(shape):
    if not isinstance(shape, list) and not isinstance(shape, tuple):
        raise NotImplementedError

    if len(shape) == 4:
        return [shape[0], shape[3], shape[1], shape[2]]
    elif len(shape) == 3:
        return [shape[2], shape[0], shape[1]] if get_layout(shape) == 'HWX' else shape
    else:
        return shape


def shape_map_nchw2nhwc(shape):
    if not isinstance(shape, list) and not isinstance(shape, tuple):
        raise NotImplementedError

    if len(shape) == 4:
        return [shape[0], shape[2], shape[3], shape[1]]
    elif len(shape) == 3:
        return [shape[1], shape[2], shape[0]] if get_layout(shape) == 'XHW' else shape
    else:
        return shape


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
            return {'pad_w': int(legacy_pad['left']), 'pad_h': int(legacy_pad['top'])}
        else:
            return {'pad_l': int(legacy_pad['left']), 'pad_r': int(legacy_pad['right']), 'pad_t': int(legacy_pad['top']), 'pad_b': int(legacy_pad['bottom'])}

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
        return {'pad_w': pad_l, 'pad_h': pad_t}
    else:
        return {'pad_l': pad_l, 'pad_r': pad_r, 'pad_t': pad_t, 'pad_b': pad_b}


# Caffe Scale Operand Shape Compatible
def isShapeCompatible(data_shape:list, weight_shape:list) -> bool:
    if len(data_shape) == 4 and len(weight_shape) <= 4:
        compatible_shape = [
            data_shape[0:1], data_shape[0:2], data_shape[0:3], data_shape[0:4],
            data_shape[1:2], data_shape[1:3], data_shape[1:4],
            data_shape[2:3], data_shape[2:4],
            data_shape[3:4]]

        return weight_shape in compatible_shape
    elif len(data_shape) >= 4 and len(weight_shape) >= 4: # TODO
        return True
    elif len(data_shape) == 3 and len(weight_shape) <= 3: # TODO
        return True
    elif weight_shape == [] or weight_shape == ():
        return True
    else:
        print(data_shape, weight_shape)
        raise NotImplementedError


# Scale Axis
def trim_one(scale_shape):
    if len(scale_shape) <= 1 or scale_shape is None:
        return scale_shape

    # Remove 1 from head
    while True:
        if len(scale_shape) > 1 and scale_shape[0] == 1:
            scale_shape.remove(1)
        else:
            break

    # Remove 1 from tail
    while True:
        if len(scale_shape) > 1 and scale_shape[-1] == 1:
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
        shape_map = list(shapeA[i:(len(shapeB)+i)] == shapeB)

        if isinstance(shape_map, list) and shape_map.count(True) == len(shapeB):
            return i
    return None
