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

        # options
        self.parseAttributes()
        if self.op_code == 'MaxPool':
            self.pooling_param['pool'] = 0
            self.pooling_param['kernel_h'] = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = self.attrs['kernel_shape'][1]
            self.pooling_param['stride_h'] = self.attrs['strides'][0]
            self.pooling_param['stride_w'] = self.attrs['strides'][1]
            self.pooling_param['ceil_mode'] = self.attrs.get('ceil_mode', True)
            self.pooling_param['pad_l'] = self.attrs['pads'][1]
            self.pooling_param['pad_r'] = self.attrs['pads'][3]
            self.pooling_param['pad_t'] = self.attrs['pads'][0]
            self.pooling_param['pad_b'] = self.attrs['pads'][2]
            self.attrs = self.pooling_param
        else:
            raise NotImplementedError 

        self.setParsed()

    def propagatableTensors(self):
        pass

    def transform(self):
        pass

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()
        return [layer]
