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
#    elif len(shape) == 2:
#        return [shape[1], shape[0]]
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
    if len(weight_shape) == weight_shape.count(1) and len(weight_shape) > 0:
        return False

    if data_shape == weight_shape:
        return True

    if len(data_shape) == 5 and len(weight_shape) <= 5:
        compatible_shape = [
            data_shape[0:1], data_shape[0:2], data_shape[0:3], data_shape[0:4], data_shape[0:5],
            data_shape[1:2], data_shape[1:3], data_shape[1:4], data_shape[1:5],
            data_shape[2:3], data_shape[2:4], data_shape[2:5],
            data_shape[3:4], data_shape[3:5],
            data_shape[4:5], []]
    elif len(data_shape) == 4 and len(weight_shape) <= 4:
        compatible_shape = [
            data_shape[0:1], data_shape[0:2], data_shape[0:3], data_shape[0:4],
            data_shape[1:2], data_shape[1:3], data_shape[1:4],
            data_shape[2:3], data_shape[2:4],
            data_shape[3:4], []]
    elif len(data_shape) == 3 and len(weight_shape) <= 3:
        compatible_shape = [
            data_shape[0:1], data_shape[0:2], data_shape[0:3],
            data_shape[1:2], data_shape[1:3],
            data_shape[2:3], []]
    elif len(data_shape) == 2 and len(weight_shape) <= 2:
        compatible_shape = [
            data_shape[0:1], data_shape[0:2],
            data_shape[1:2], []]
    elif weight_shape == [] or weight_shape == ():
        return True
    elif data_shape == [] or data_shape == ():
        return False
    else:
        print(data_shape, weight_shape)
        raise NotImplementedError

    return weight_shape in compatible_shape


def isShapeFullyDefined(shape:list):
    if not isinstance(shape, list) or 0 in shape:
        return False
    else:
        return True
