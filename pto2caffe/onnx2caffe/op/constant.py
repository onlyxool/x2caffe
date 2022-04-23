import logging

from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Constant(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.layer_type = 'Constant'

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        self.model.input_tensor[self.node.output[0]] = self.attrs['value']


    def convert(self):
        pass
