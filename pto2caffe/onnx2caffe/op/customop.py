import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Mish(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.mish_param = dict()
        self.attrs = self.mish_param
        self.setInited()


    @property
    def type(self):
        return 'Mish'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, mish_param=self.mish_param)

        self.setConverted()

        return [layer]
