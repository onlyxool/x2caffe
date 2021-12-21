import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Constant(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        return 'Constant'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        self.model.input_tensor[self.node.output[0]] = self.attrs['value']

        self.isLegacy = True


    def convert(self):
        pass
