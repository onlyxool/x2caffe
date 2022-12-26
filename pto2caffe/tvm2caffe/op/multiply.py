import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Multiply(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'multiply')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0 # Caffe Eltwise PROD
            self.attrs = self.eltwise_param
            self.setParsed()
        elif (self.inputs_buf[0] is not None or self.inputs_buf[1] is not None) or (self.inputs_shape[0] != self.inputs_shape[1]):
            self.type = 'Scale'

            inputs_size0 = np.multiply.reduce(self.inputs_shape[0], axis=None)
            inputs_size1 = np.multiply.reduce(self.inputs_shape[1], axis=None)

            if inputs_size0 < inputs_size1:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 1):
                self.byPassOperator()
                return

            if type(self.inputs_buf[1]) is np.ndarray:
                self.inputs_buf[1] = self.inputs_buf[1].squeeze()
                self.inputs_shape[1] = self.inputs_buf[1].shape
            elif self.inputs_buf[1] is None:
                self.type = 'Reshape+Scale'
                self.inter_blob = 'reshape_scale'+str(self.index)
                target_shape = list(np.squeeze(np.random.random(self.inputs_shape[1])).shape)
                self.inputs_shape[1] = target_shape if target_shape != [] else [1]

            self.weight = self.inputs_buf[1] if self.inputs_buf[0] is None else self.inputs_buf[0]
            self.bias = None

            self.scale_param = dict()
            if 'axis' in self.attrs:            
                self.scale_param['axis'] = self.attrs['axis']
            else:
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0

            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False
            self.attrs = self.scale_param

            self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Scale':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        elif self.type == 'Reshape+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [None], [self.inter_blob], reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.inter_blob], self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))

        self.setConverted()

        return layers
