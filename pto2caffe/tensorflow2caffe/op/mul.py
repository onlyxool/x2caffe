import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Mul(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Mul')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] * self.inputs_buf[1])

        elif (self.inputs_buf[0] is not None or self.inputs_buf[1] is not None) or (self.inputs_shape[0] != self.inputs_shape[1]):
            self.type = 'Scale'

            if self.inputs_buf[0] is not None or \
                (self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0].count(1) > self.inputs_shape[1].count(1) and \
                len(self.inputs_shape[0]) <= len(self.inputs_shape[1])):
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            # Weight & Bias
            self.weight = self.inputs_buf[1]
            self.bias = None

            # Weight Shape
            self.inputs_shape[1] = list(np.squeeze(np.random.random(self.inputs_shape[1])).shape)
            if self.weight is None:
                self.type = 'Reshape+Scale'
                self.inter_blob = 'reshape_scale' + str(self.index)
            else:
                self.weight = self.weight.reshape(self.inputs_shape[1])

            # Attribute
            self.scale_param = dict()
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False

            self.attrs = self.scale_param
            self.setParsed()
        else:
            self.type = 'Eltwise'

            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0

            self.attrs = self.eltwise_param
            self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Scale':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        elif self.type == 'Reshape+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [None], [self.inter_blob], reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.inter_blob], self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        elif self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))

        self.setConverted()

        return layers
