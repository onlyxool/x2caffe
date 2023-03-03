import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

from util import isShapeCompatible


class Mul(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Mul')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.saveConstant(self.node.output[0], self.inputs_buf[0] * self.inputs_buf[1])
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0 # Caffe Eltwise PROD
            self.attrs = self.eltwise_param
            self.setParsed()
        elif (self.inputs_buf[0] is not None or self.inputs_buf[1] is not None) or (self.inputs_shape[0] != self.inputs_shape[1]):
            self.type = 'Scale'

            inputs_size0 = np.multiply.reduce(self.inputs_shape[0], axis=None)
            inputs_size1 = np.multiply.reduce(self.inputs_shape[1], axis=None)

            if self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()
            elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and inputs_size0 < inputs_size1:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 1):
                self.byPassOperator()
                return

            if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]) and self.inputs_buf[1] is None:
                weight_shape = list(np.squeeze(np.random.random(self.inputs_shape[1])).shape)
                if not isShapeCompatible(self.inputs_shape[0], weight_shape):
                    self.unSupported('Inputs shape uncompatible for Caffe. ' + str(self.inputs_shape[0]) + ' x ' + str(self.inputs_shape[1]))
                    return
                self.type = 'Reshape+Scale'
                self.inputs_shape[1] = weight_shape

            self.scale_param = dict()
            if self.model.opset[0] >= 7:
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            else:
                self.scale_param['axis'] = self.attrs['axis']
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False
            self.attrs = self.scale_param

            self.weight = self.inputs_buf[1]
            self.bias = None

            self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Scale':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        elif self.type == 'Reshape+Scale':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [None], self.interblob, reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.interblob[0]], self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))

        self.setConverted()

        return layers
