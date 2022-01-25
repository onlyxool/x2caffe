import logging
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Binary(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        if hasattr(self, 'eltwise_param'):
            return 'Eltwise'
        elif hasattr(self, 'scale_param'):
            return 'Scale'
        else:
            return self.op_code


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        if self.op_code == 'Sum':
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1 #Caffe Eltwise SUM
            self.attrs = self.eltwise_param
        elif self.op_code == 'Div':
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
        else:
            raise NotImplementedError(self.op_code)

        self.setParsed()


    def convert(self):
        if hasattr(self, 'eltwise_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif hasattr(self, 'scale_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()
        return [layer]
