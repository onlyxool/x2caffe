import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class ReLU(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.layer_type = 'ReLU'

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        self.relu_param = dict()
        self.relu_param['negative_slope'] = self.attrs.get('alpha', 0)
        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
