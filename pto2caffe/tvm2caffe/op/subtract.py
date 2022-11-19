import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Subtract(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'subtract')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is not None:
            self.type = 'Bias'

            if self.inputs_buf[0] is not None:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 0):
                self.byPassOperator()
                return

            if type(self.inputs_buf[1]) is np.ndarray:
                self.inputs_buf[1] = self.inputs_buf[1].squeeze()
                self.inputs_shape[1] = self.inputs_buf[1].shape

            self.bias = self.inputs_buf[1] * -1

            self.bias_param = dict()
            if 'axis' in self.attrs:
                self.bias_param['axis'] = self.attrs['axis']
            else:
                self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0

            self.bias_param['num_axes'] = len(self.inputs_shape[1])

            self.attrs = self.bias_param

            self.setParsed()
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.type = 'Scale+Bias'

            self.inter_blob = 'bias_split'+str(self.index)

            self.scale_param = dict()
            self.weight = np.ones(self.inputs_shape[1]).astype(np.float32) * -1

            self.scale_param['axis'] = 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])

            self.attrs = self.bias_param
            self.setParsed()
        elif self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
            self.type = 'Bias+Scale'

            self.inputs.reverse()
            self.inputs_shape.reverse()
            self.inputs_buf.reverse()

            self.weight = np.ones(self.outputs_shape[0]).astype(np.float32) * -1
            self.bias = self.inputs_buf[1] * -1

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])

            self.scale_param = dict()
            self.scale_param['axis'] = 0
            self.scale_param['num_axes'] = len(self.outputs_shape[0])
            self.scale_param['bias_term'] = False

            self.attrs = self.bias_param
            self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Bias':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Scale+Bias':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], self.inputs_buf, [self.inter_blob], self.weight, scale_param=self.scale_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.inter_blob], self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Bias+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[0], self.inter_blob], self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[1]], self.inputs_buf, [self.inter_blob], self.weight, scale_param=self.scale_param))

        self.setConverted()

        return layers
