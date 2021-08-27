import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Reduce(Operator):

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
        assert(self.attrs['keepdims'] == 0), 'Do Not Support keepdims == 1'
        assert(self.attrs['axes'] == [2, 3]), 'Do Not Support reduce on axis 0 or 1'

        if self.op_code == 'ReduceMean':
            self.pooling_param['pool'] = 1 # Pooling.AVE
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
            self.pooling_param['stride'] = 1
            self.pooling_param['ceil_mode'] = False
        else:
            raise NotImplementedError(self.op_code)

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
