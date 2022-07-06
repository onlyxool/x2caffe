import numpy as np
from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import compute_scale_axis


class Sub(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Sub')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.model.constant[self.outputs[0]] = self.inputs_buf[0] - self.inputs_buf[1]
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is not None:
            self.layer_type = 'Scale'

            # Weight
            self.weight = np.ones(self.inputs_shape[0]).astype(np.float32)

            # Bias
            if len(self.inputs_shape[1]) == 4:
                self.bias = -self.inputs_buf[1].transpose(0,3,1,2)
            elif len(self.inputs_shape[1]) == 3:
                self.bias = -self.inputs_buf[1].transpose(2,0,1)
            else:
                self.bias = -self.inputs_buf[1]

            # Attribute
            self.scale_param = dict()
            self.scale_param['axis'] = 0
            self.scale_param['num_axes'] = len(self.weight.shape)
            self.scale_param['bias_term'] = True

            self.attrs = self.scale_param
            self.setParsed()
        else:
            raise NotImplementedError


    def convert(self):
        if self.layer_type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        if self.layer_type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
