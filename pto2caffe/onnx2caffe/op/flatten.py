import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Flatten(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.flatten_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Flatten'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        self.flatten_param['axis'] = self.attrs.get('axis', 1)
        self.attrs = self.flatten_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, flatten_param=self.flatten_param)
        self.setConverted()
        return [layer]

