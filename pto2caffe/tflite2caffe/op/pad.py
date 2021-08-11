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

    def __init__(self, model, tf_op, tf_op_code, index,):
        super().__init__(model, tf_op, tf_op_code, index,)
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


    def propagatableTensors(self):
        return [self.inputs[0], self.outputs[0]]


    def transform(self):
        # Padding.transform() handls TFLite/ONNX semantic gap in addition to layout gap
        # TensorFlow (Lite) pads is `[n, 2]` where `[i, 0]` is _begin_ and `[i, 1]` is _end_
        # ONNX pads is `[n * 2]` sequenced as `[x1_begin, x2_begin,...,x1_end, x2_end,...]`
        layout = self.inputs[0].layout
        pt = self.inputs[1]
        pads = pt.data
        pads = np.reshape(pads, pt.shape)
        if layout is None:
            pads = np.transpose(pads)
        else:
            pads_begin = pads[:, 0]
            pads_end = pads[:, 1]
            pads_begin = layout.transform(pads_begin)
            pads_end = layout.transform(pads_end)
            pads = np.array([pads_begin, pads_end])
        pt.data = pads.flatten()
        pt.shape = [np.prod(pt.shape), ]


    def convert(self):
        pass

def computePaddingSize(padding_mode, input_size, output_size, proto_param:dict, legacy_pad):
    if padding_mode == 1: #tflite.Padding.VALID
        if legacy_pad['left'] == legacy_pad['right'] and legacy_pad['top'] == legacy_pad['bottom']:
            return (int(legacy_pad['left']), int(legacy_pad['top']))
        else:
            return (int(legacy_pad['left']), int(legacy_pad['right']), int(legacy_pad['top']), int(legacy_pad['bottom']))

    stride_h = proto_param['stride_h']
    input_h = input_size[2]
    kernel_h = proto_param['kernel_h']
    output_h = output_size[2]

    if 'dilation' in proto_param: # Convolution
        dilation_h = proto_param['dilation'][0]
        pad_h = (output_h - 1) * stride_h - input_h + (dilation_h * (kernel_h - 1) + 1)
    else: # Pooling
        pad_h = (output_h - 1) * stride_h - input_h + kernel_h

    if math.modf(pad_h/2)[0] != 0:
        pad_b = math.ceil(pad_h)
        pad_t = pad_h - pad_b
    else:
        pad_t = pad_h/2
        pad_b = pad_h/2

    stride_w = proto_param['stride_w']
    input_w = input_size[3]
    kernel_w = proto_param['kernel_w']
    output_w = output_size[3]

    if 'dilation' in proto_param: # Convolution
        dilation_w = proto_param['dilation'][1]
        pad_w = (output_w - 1) * stride_w - input_w + (dilation_w * (kernel_w - 1) + 1)
    else: # Pooling
        pad_w = (output_w - 1) * stride_w - input_w + kernel_w

    if math.modf(pad_w/2)[0] != 0:
        pad_r = math.ceil(pad_w)
        pad_l = pad_w - pad_r
    else:
        pad_r = pad_w/2
        pad_l = pad_w/2

    pad_l = int(pad_l + legacy_pad['left'])
    pad_r = int(pad_r + legacy_pad['right'])
    pad_t = int(pad_t + legacy_pad['top'])
    pad_b = int(pad_b + legacy_pad['bottom'])
    if pad_l == pad_r and pad_t == pad_b:
        return (pad_l, pad_t)
    else:
        return (pad_l, pad_r, pad_t, pad_b)
