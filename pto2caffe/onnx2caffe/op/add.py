import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

from util import isShapeCompatible


class Add(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Add')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1 # Caffe Eltwise SUM
            self.attrs = self.eltwise_param
        else:
            self.layer_type = 'Bias'

            inputs_size0 = np.multiply.reduce(self.inputs_shape[0], axis=None)
            inputs_size1 = np.multiply.reduce(self.inputs_shape[1], axis=None)

            if self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()
            elif inputs_size0 < inputs_size1:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 0):
                self.byPassOperator()
                return

            if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]):
                bias_shape = list(np.squeeze(np.random.random(self.inputs_shape[1])).shape)
                if not isShapeCompatible(self.inputs_shape[0], bias_shape):
                    self.model.errorMsg.append('[' + self.node.name + ']: Operator Mul Inputs shape uncompatible for Caffe. ' + str(self.inputs_shape[0]) + ' + ' + str(self.inputs_shape[1]))
                    self.model.unsupport.append(self.operator_code)
                    return
                self.inputs_shape[1] = bias_shape
            else:
                bias_shape = self.inputs_shape[1]

            if self.inputs_buf[1] is not None:
                self.inputs_buf[1] = self.inputs_buf[1].reshape(bias_shape)
            else:
                self.layer_type = 'Reshape+Bias'

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0

            self.bias = self.inputs_buf[1]

            self.attrs = self.bias_param

        self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Bias':
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Reshape+Bias':
            reshape_out = 'reshape'+str(self.index)
            layers.append(caffe_layer('Reshape', 'Reshape'+str(self.index), [self.inputs[1]], [None], [reshape_out], reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))))
            layers.append(caffe_layer('Bias', 'Bais'+str(self.index), [self.inputs[0], reshape_out], self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))

        self.setConverted()

        return layers
