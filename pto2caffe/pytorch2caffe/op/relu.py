import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class ReLU(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.relu_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'ReLU'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        self.relu_param['negative_slope'] = 0

        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]