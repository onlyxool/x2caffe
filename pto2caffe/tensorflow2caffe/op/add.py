import numpy as np
from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Add(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('Add', 'AddV2', 'AddN'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1

            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.model.constant[self.outputs[0]] = self.inputs_buf[0] + self.inputs_buf[1]
        else:
            self.layer_type = 'Scale'

            if self.inputs_buf[0] is not None:
                bias_index = 0
                input_index = 1
            else:
                bias_index = 1
                input_index = 0

            # Weight
            self.weight = np.ones(self.inputs_shape[bias_index], dtype=float, order='C')

            # Bias
            self.bias = self.inputs_buf[bias_index]

            # Attributes
            self.scale_param = dict()
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
            self.scale_param['num_axes'] = len(self.weight.shape)

            self.scale_param['bias_term'] = True

            self.attrs = self.scale_param

            self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
