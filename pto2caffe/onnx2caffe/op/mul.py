import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Mul(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Mul')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            # Eltwise Layer
            self.layer_type = 'Eltwise'

            # Attributes
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0 # Caffe Eltwise PROD
            self.attrs = self.eltwise_param
        else:
            # Scale Layer
            self.layer_type = 'Scale'

            # Attributes
            self.scale_param = dict()
            self.scale_param['bias_term'] = False

            # Axis
            if self.model.opset[0] >= 7:
                for i in range(len(self.inputs_shape[0])):
                    if self.inputs_buf[1].shape == () or self.inputs_buf[1].shape == []:
                        self.inputs_buf[1] = np.ones(self.inputs_shape[0]) * self.inputs_buf[1]
                        self.inputs_shape[1] = self.inputs_shape[0]

                    if self.inputs_shape[0][i] == self.inputs_shape[1][0]:
                        self.scale_param['axis'] = i
                        break
            else:
                self.scale_param['axis'] = self.attrs['axis']

            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.attrs = self.scale_param

            # Weight
            self.weight = self.inputs_buf[1]

            # Bias
            self.bias = None

        self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
