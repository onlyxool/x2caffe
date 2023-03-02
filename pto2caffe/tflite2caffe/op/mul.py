import tflite
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

from util import isShapeCompatible


class Mul(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'MUL')
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):

        op_opt = self.op.BuiltinOptions()
        opt = tflite.MulOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        self.parseInputOutput()

        # Attributes
        if self.inputs_shape[0] != self.inputs_shape[1] or self.inputs_buf[1] is not None:
            # Scale Layer
            self.type = 'Scale'

            if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]):
                weight_shape = list(np.squeeze(np.random.random(self.inputs_shape[1])).shape)
                if not isShapeCompatible(self.inputs_shape[0], weight_shape):
                    self.model.errorMsg.append('Error: Operator Mul Inputs shape uncompatible for Caffe. ' + str(self.inputs_shape[0]) + ' x ' + str(self.inputs_shape[1]))
                    self.model.unsupport.append(self.operator_code)
                    return
            else:
                weight_shape = self.inputs_shape[1]

            if weight_shape != self.inputs_shape[1]:
                self.inputs_shape[1] = weight_shape
                if self.inputs_buf[1] is not None:
                    self.inputs_buf[1] = self.inputs_buf[1].reshape(weight_shape)
                else:
                    self.type = 'Reshape+Scale'
                    self.reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))

            self.weight = self.inputs_buf[1]
            self.bias = None

            self.scale_param = dict()
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False

            self.attrs = self.scale_param
        else:
            # Eltwise Layer
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 0
            self.attrs = self.eltwise_param

        self.activ_type_code = opt.FusedActivationFunction()

        self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Scale':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))
        elif self.type == 'Reshape+Scale':
            layers.append(caffe_layer(self.layer_type[0], 'Reshape'+str(self.index), [self.inputs[1]], [None], self.interblob, reshape_param=self.reshape_param))
            layers.append(caffe_layer(self.layer_type[1], 'Scale'+str(self.index), [self.inputs[0], self.interblob[0]], self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param))

        self.setConverted()

        return layers
