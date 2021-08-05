import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Binary(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()

    @property
    def type(self):
        if self.op_code == 'Add':
            return 'Eltwise'
        elif self.op_code == 'Sub':
            return 'TODO'
        elif self.op_code == 'Mul':
            if hasattr(self, 'eltwise_param'):
                return 'Eltwise'
            elif hasattr(self, 'scale_param'):
                return 'Scale'
            else:
                return 'Mul'
        elif self.op_code == 'Pow':
            return 'TODO'
        else:
            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        if self.op_code == 'Add':
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1 #Caffe Eltwise SUM
            self.attrs = self.eltwise_param
        elif self.op_code == 'Mul':
            if len(self.inputs_shape[0]) == len(self.inputs_shape[1]):
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
        else:
            raise NotImplementedError

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass

    def convert(self):
        if self.op_code == 'Add':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.op_code == 'Mul':
            if hasattr(self, 'eltwise_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
            elif hasattr(self, 'scale_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, scale_param=self.scale_param)
        else:
            raise NotImplementedError

        self.setConverted()
        return [layer]
