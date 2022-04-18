import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Elu(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        return 'ELU'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()
        self.elu_param = dict()
        self.elu_param['alpha'] = self.attrs.get('alpha', 1.0)
        self.attrs = self.elu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, elu_param=self.elu_param)

        self.setConverted()

        return [layer]
