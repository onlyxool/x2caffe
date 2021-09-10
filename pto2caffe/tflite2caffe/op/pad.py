import math
import tflite
import logging
import numpy as np

from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

PaddingMapping = {
    tflite.Padding.SAME: 'SAME_UPPER',
    tflite.Padding.VALID: 'VALID',
}


class Pad(Operator):

    TypeMapping = {
        tflite.BuiltinOperator.PAD: 'Pad',
        tflite.BuiltinOperator.MIRROR_PAD: 'Pad',
    }


    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.pad = dict()
        self.attrs = self.pad
        self.setInited()


    @property
    def type(self):
        return 'Pad'


    def parse(self):
        logger.debug("Parsing %s...", self.shorty)

        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        pad_tensor = self.inputs_buf[1]
        self.pad['left'] = pad_tensor[2][0]
        self.pad['right'] = pad_tensor[2][1]
        self.pad['top'] = pad_tensor[1][0]
        self.pad['bottom'] = pad_tensor[1][1]


    def convert(self):
        pass

def computePaddingSize(input, output, stride, kernel, dilation, layer_type):
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

    return pad


def asymmetric_pad(pad):
    if math.modf(pad / 2)[0] != 0:
        pad_b = math.ceil(pad / 2)
        pad_f = pad - pad_b
    else:
        pad_f = pad / 2
        pad_b = pad / 2

    return pad_f, pad_b


def handleLegacyPad(padding_mode, input_size, output_size, proto_param:dict, legacy_pad, layer_type):
    if padding_mode == tflite.Padding.VALID:
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

    pad_h = computePaddingSize(input_h, output_h, stride_h, kernel_h, dilation_h, layer_type)
    pad_t, pad_b = asymmetric_pad(pad_h)

    # Vertical
    input_w = input_size[3]
    output_w = output_size[3]
    kernel_w = proto_param['kernel_w']
    stride_w = proto_param['stride_w']
    dilation_w = proto_param.get('dilation', [1,1])[1]

    pad_w = computePaddingSize(input_w, output_w, stride_w, kernel_w, dilation_w, layer_type)
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
