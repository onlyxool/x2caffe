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

#        if self.op_code == 'Add':
#            return 'Eltwise'
#        elif self.op_code == 'Sum':
#            return 'Eltwise'
#        elif self.op_code == 'Sub':
#            return 'Eltwise'
#        elif self.op_code == 'Mul':
#            if hasattr(self, 'eltwise_param'):
#                return 'Eltwise'
#            elif hasattr(self, 'scale_param'):
#                return 'Scale'
#            else:
#                return 'Mul'
#        elif self.op_code == 'Div':
#            return 'Eltwise'
#        elif self.op_code == 'Pow':
#            return 'TODO'
#        elif self.op_code == 'MatMul':
#            return 'Eltwise'
#        else:
#            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        if self.op_code == 'Add': # TODO: boradcast concern
            if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 1 #Caffe Eltwise SUM
                self.attrs = self.eltwise_param
            else:
                self.weight = np.ones(self.inputs_shape[0][1], dtype=None, order='C')
                self.bias = self.inputs_buf[1]
                self.scale_param = dict()
                self.scale_param['bias_term'] = True
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
                self.scale_param['num_axes'] = len(self.bias.shape)
                self.attrs = self.scale_param
        elif self.op_code == 'Sum':
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1 #Caffe Eltwise SUM
            self.attrs = self.eltwise_param
        elif self.op_code == 'Sub':
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 #Caffe Eltwise SUB
            self.attrs = self.eltwise_param
        elif self.op_code == 'Mul':
            if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 0 #Caffe Eltwise PROD
                self.attrs = self.eltwise_param
            else:
                self.scale_param = dict()
                self.scale_param['bias_term'] = False
                for i in range(len(self.inputs_shape[0])):
                    if self.inputs_shape[0][i] == self.inputs_shape[1][0]:
                        self.scale_param['axis'] = i
                        break
                self.attrs = self.scale_param
                self.weight = self.inputs_buf[1]
                self.bias = None
        elif self.op_code == 'Div':
            if self.inputs_buf[1] is not None:
                self.scale_param = dict()
                self.scale_param['bias_term'] = False
                for i in range(len(self.inputs_shape[0])):
                    if self.inputs_shape[0][i] == self.inputs_shape[1][0]:
                        self.scale_param['axis'] = i
                        break
                self.attrs = self.scale_param
                self.weight = 1/self.inputs_buf[1]
                self.bias = None
            else:
                raise NotImplementedError(self.op_code)
        elif self.op_code == 'MatMul':
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0 #Caffe Eltwise PROD
            self.attrs = self.eltwise_param
        else:
            raise NotImplementedError(self.op_code)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass

    def convert(self):
        if hasattr(self, 'eltwise_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif hasattr(self, 'scale_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

#        if self.op_code == 'Add' or self.op_code == 'Sum' or self.op_code == 'Sub':
#            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
#        elif self.op_code == 'Mul' or self.op_code == 'MatMul':
#            if hasattr(self, 'eltwise_param'):
#                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
#            elif hasattr(self, 'scale_param'):
#                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, scale_param=self.scale_param)
#        elif self.op_code == 'Div':
#                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, scale_param=self.scale_param)
#        else:
#            raise NotImplementedError

        self.setConverted()
        return [layer]
