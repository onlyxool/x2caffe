import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Swish(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.swish_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Swish'
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        if self.operator == 'nn.Hardswish':
            self.swish_param['beta'] = 1.0

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, swish_param=self.swish_param)

        self.setConverted()

        return [layer]
