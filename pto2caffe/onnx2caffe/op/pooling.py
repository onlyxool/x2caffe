import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Pooling(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.pooling_param = dict()
        self.setInited()

    @property
    def type(self):
        return 'Pooling'
        
    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Options
        self.parseAttributes()
        if self.op_code == 'MaxPool':
            self.pooling_param['pool'] = 0
            self.pooling_param['kernel_h'] = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = self.attrs['kernel_shape'][1]
        elif self.op_code == 'AveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = self.attrs['kernel_shape'][1]
        elif self.op_code == 'GlobalAveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
        else:
            raise NotImplementedError 

        strides = self.attrs.get('strides', [1, 1])
        self.pooling_param['stride_h'] = strides[0]
        self.pooling_param['stride_w'] = strides[1]
        self.pooling_param['ceil_mode'] = self.attrs.get('ceil_mode', False)

        # Padding
        attr_padding = self.attrs.get('pads', [0,0,0,0])
        for legacy in self.model.legacys:
            if legacy.outputs[0] == self.inputs[0]:
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
            self.pooling_param['pad_w'] = pad_l
            self.pooling_param['pad_h'] = pad_t
        else:
            self.pooling_param['pad_l'] = pad_l
            self.pooling_param['pad_r'] = pad_r
            self.pooling_param['pad_t'] = pad_t
            self.pooling_param['pad_b'] = pad_b

        self.attrs = self.pooling_param

        self.setParsed()

    def propagatableTensors(self):
        pass

    def transform(self):
        pass

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()
        return [layer]
