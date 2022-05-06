import numpy as np
import copy

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import trim_one
from util import compute_scale_axis

class Scale(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'Mul' or self.operator == 'BiasAdd')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.scale_param = dict()
        if self.operator == 'BiasAdd':
            self.layer_type = 'Scale'
            self.weight = np.ones(self.inputs_shape[1], dtype=float, order='C')
            self.bias = self.inputs_buf[1]
            if self.inputs_shape[1] != []:
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
            self.scale_param['bias_term'] = True
            self.attrs = self.scale_param
            self.setParsed()
        elif self.operator == 'Mul':
            self.layer_type = 'Scale'
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
                self.layer_type = 'Eltwise'
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 0
                self.attrs = self.eltwise_param
                self.setParsed()
        else:
            print(self.operator)
            raise NotImplementedError


    def convert(self):
        if self.layer_type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
        elif self.layer_type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)

        self.setConverted()

        return [layer]
