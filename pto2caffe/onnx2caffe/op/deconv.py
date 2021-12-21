import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Deconvolution(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.convolution_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Deconvolution'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Weight
        self.weight = self.inputs_buf[1]
        print(self.name, self.inputs_shape, self.outputs_shape)
        # Bias
        self.bias = self.inputs_buf[2] if len(self.inputs_buf) == 3 else None

        # Option
        self.parseAttributes()
        self.convolution_param['num_output'] = self.outputs_shape[0][1]
        self.convolution_param['stride'] = self.attrs.get('strides', [1, 1]) 
        self.convolution_param['dilation'] = self.attrs.get('dilations', [1, 1]) 
        self.convolution_param['group'] = self.attrs.get('group', 1)
        self.convolution_param['kernel_size'] = self.attrs['kernel_shape']
        self.convolution_param['bias_term'] = True if self.bias is not None else False

        # Padding
        attr_padding = self.attrs.get('pads', [0,0,0,0])
        for legacy in self.model.legacys:
            if legacy.outputs[0] == self.inputs[0] and legacy.op_code == 'Pad':
                legacy_pad = legacy.pad
                pad_l = attr_padding[1] + legacy.pad['left']
                pad_r = attr_padding[3] + legacy.pad['right']
                pad_t = attr_padding[0] + legacy.pad['top']
                pad_b = attr_padding[2] + legacy.pad['bottom']
                self.inputs[0] = legacy.inputs[0]
                self.inputs_shape[0] = legacy.inputs_shape[0]
        else:
            pad_l = attr_padding[1]
            pad_r = attr_padding[3]
            pad_t = attr_padding[0]
            pad_b = attr_padding[2]

        if pad_l == pad_r and pad_t == pad_b:
            self.convolution_param['pad_w'] = pad_l
            self.convolution_param['pad_h'] = pad_t
        else:
            self.convolution_param['pad_l'] = pad_l
            self.convolution_param['pad_r'] = pad_r
            self.convolution_param['pad_t'] = pad_t
            self.convolution_param['pad_b'] = pad_b

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
