import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Sqrt(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.power_param = dict()
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.layer_type = 'Power'

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()

        self.power_param['power'] = 0.5
        self.power_param['scale'] = 1
        self.power_param['shift'] = 0
        self.attrs = self.power_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, power_param=self.power_param)

        self.setConverted()

        return [layer]
