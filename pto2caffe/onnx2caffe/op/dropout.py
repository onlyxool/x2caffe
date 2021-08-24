import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Dropout(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.dropout_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Dropout'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()
        if len(self.outputs) == 2: # Remove output mask
            self.outputs.pop()
            self.outputs_shape.pop()

        # Option
        self.parseAttributes()
        self.dropout_param['dropout_ratio'] = self.attrs['ratio']
        self.attrs = self.dropout_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, dropout_param=self.dropout_param)

        self.setConverted()

        return [layer]

