import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Debug(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.debug_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Debug'
        logger.debug('Parsing %s...', self.type)

        self.parseInput()
        self.parseOutput()
        print(self.name, self.operator)
        print(self.inputs)
        print(self.inputs_shape)
        print(self.inputs_buf)
        print(self.outputs, self.outputs_shape)
        # Attributes
        self.parseAttributes()
        print(self.attrs)

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, debug_param=self.debug_param)

        self.setConverted()

        return [layer]
