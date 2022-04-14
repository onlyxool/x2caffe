import logging
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Div(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        return 'Scale'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.parseAttributes()

        if self.inputs_buf[1] is not None:
            self.scale_param = dict()
            self.scale_param['bias_term'] = False
            for i in range(len(self.inputs_shape[0])):
                if self.inputs_shape[1] == [] or self.inputs_shape[1] == ():
                    self.scale_param['axis'] = 0
                    break
                if self.inputs_shape[0][i] == self.inputs_shape[1][0]:
                    self.scale_param['axis'] = i
                    break
            self.attrs = self.scale_param
            self.weight = 1/self.inputs_buf[1]
            self.bias = None
        else:
            raise NotImplementedError(self.op_code)

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
