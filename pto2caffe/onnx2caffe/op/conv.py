import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator
#from onnx2caffe.op.pad import computePaddingSize

logger = logging.getLogger('onnx2caffe')

class Convolution(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.convolution_param = dict()
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2] if len(self.inputs_buf) == 3 else None

        # Option
        self.parseAttributes()
        self.convolution_param['num_output'] = self.outputs_shape[0][1]#self.graph.Tensors(self.outputs[0]).Shape(3)
        self.convolution_param['stride'] = self.attrs.get('strides', [1, 1])
        self.convolution_param['dilation'] = self.attrs.get('dilations', [1, 1])
        self.convolution_param['group'] = self.attrs.get('group', 1)
        self.convolution_param['kernel_size'] = self.attrs['kernel_shape']
        self.convolution_param['bias_term'] = True if self.bias is not None else False
        self.convolution_param['pad_l'] = self.attrs['pads'][1]
        self.convolution_param['pad_r'] = self.attrs['pads'][3]
        self.convolution_param['pad_t'] = self.attrs['pads'][0]
        self.convolution_param['pad_b'] = self.attrs['pads'][2]

#        legacy_pad = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
#        for legacy in self.model.legacys:
#            if legacy.outputs[0] == self.inputs[0]:
#                legacy_pad = legacy.pad
#                self.inputs[0] = legacy.inputs[0]
#        padding = computePaddingSize(opt.Padding(), self.inputs_shape[0], self.outputs_shape[0], self.convolution_param, legacy_pad)
#        if len(padding) == 2:
#            self.convolution_param['pad_w'] = padding[0]
#            self.convolution_param['pad_h'] = padding[1]
#        elif len(padding) == 4:
#            self.convolution_param['pad_l'] = padding[0]
#            self.convolution_param['pad_r'] = padding[1]
#            self.convolution_param['pad_t'] = padding[2]
#            self.convolution_param['pad_b'] = padding[3]
#            print(self.name, padding)
#            if self.isDepthwise is True:
#                raise NotImplementedError("Depthwise Convolution not support asymmetric padding")

        self.attrs = self.convolution_param
        self.setParsed()
        
    @property
    def type(self):
        return 'Convolution'

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)
        self.setConverted()
        return [layer]
