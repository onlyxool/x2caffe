import copy
import logging

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import trim_one
from util import compute_scale_axis

logger = logging.getLogger('TensorFlow2Caffe')

class Scale(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.scale_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Scale'


    def parse(self):
        logger.debug('Parsing %s...', self.type)
        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        if self.op_code == 'BiasAdd':
            self.bias = self.inputs_buf[1]
            if self.inputs_shape[1] != []:
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
            self.scale_param['bias_term'] = True
            self.attrs = self.scale_param
        elif self.op_code == 'Mul':
            if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
                self.model.constant[self.outputs[0]] = self.inputs_buf[0] * self.inputs_buf[1]
            elif self.inputs_shape[0] != self.inputs_shape[1] or self.inputs_buf[1] is not None:
                self.scale_param = dict()

                org_shape = copy.deepcopy(self.inputs_shape[1])
                trim = trim_one(org_shape)
                if trim != self.inputs_shape[1]:
                    self.pre.append('Reshape')
                    self.inputs_shape[1] = trim
                    if self.inputs_buf[1] is not None:
                        self.inputs_buf[1] = self.inputs_buf[1].reshape(tuple(trim))

                axis = compute_scale_axis(self.inputs_shape[0], trim)
                if axis is not None:
                    self.scale_param['axis'] = axis

                self.weight = self.inputs_buf[1]
                self.scale_param['bias_term'] = False
                self.bias = None
                self.attrs = self.scale_param
                self.setParsed()
            else:
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 0
                self.attrs = self.eltwise_param
                self.setParsed()
        else:
            print(self.op_code)
            raise NotImplementedError


    def convert(self):
        if hasattr(self, 'scale_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, None, self.bias, scale_param=self.scale_param)
        elif hasattr(self, 'eltwise_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)

        self.setConverted()

        return [layer]
