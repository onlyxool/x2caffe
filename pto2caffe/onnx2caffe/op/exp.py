import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Exp(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.exp_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Exp'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        # Leave all arguments in ExpParameter as default
        # base = -1.0 (base = e)
        # scale = 1.0
        # shift = 0.0

        self.attrs = self.exp_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, exp_param=self.exp_param)

        self.setConverted()

        return [layer]
