import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator
#from onnx2caffe.op.pad import computePaddingSize

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
        if self.op_code.endswith('MaxPool'):
            self.pooling_param['pool'] = 0
            self.pooling_param['kernel_h'] = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = self.attrs['kernel_shape'][1]
        elif self.op_code.endswith('AveragePool'):
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
        else:
            raise NotImplementedError 

        strides = self.attrs.get('strides', [1, 1])
        self.pooling_param['stride_h'] = strides[0]
        self.pooling_param['stride_w'] = strides[1]
        self.pooling_param['ceil_mode'] = self.attrs.get('ceil_mode', True)

        padding = self.attrs.get('pads', [0,0,0,0])
        self.pooling_param['pad_l'] = padding[1]
        self.pooling_param['pad_r'] = padding[3]
        self.pooling_param['pad_t'] = padding[0]
        self.pooling_param['pad_b'] = padding[2]
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
