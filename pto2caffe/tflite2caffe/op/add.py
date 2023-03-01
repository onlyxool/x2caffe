import tflite
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

from util import isShapeCompatible


class Add(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator_code == 'ADD')
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.parseInputOutput()

        op_opt = self.op.BuiltinOptions()
        opt = tflite.AddOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1
            self.attrs = self.eltwise_param
        elif self.inputs_shape[0] != self.inputs_shape[1]:
            self.type = 'Bias'

            if self.inputs_buf[0] is not None and self.inputs_buf[1] is None:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            backup = self.inputs_shape[1]
            if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]):
                self.inputs_shape[1] = list(np.squeeze(np.random.random(self.inputs_shape[1])).shape)
                if not isShapeCompatible(self.inputs_shape[0], self.inputs_shape[1]):
                    self.model.errorMsg.append('Error: Operator ADD Inputs shape uncompatible for Caffe. ' + str(self.inputs_shape[0]) + ' + ' + str(backup))
                    self.model.unsupport.append(self.operator_code)
                    return

            if self.inputs_buf[1] is not None:
                self.inputs_buf[1] = self.inputs_buf[1].reshape(self.inputs_shape[1])
                self.bias = self.inputs_buf[1]
            else:
                self.type = 'Reshape+Bias'
                self.reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))
                self.bias = None

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])
            self.attrs = self.bias_param
        else:
            print(self.inputs_shape, self.inputs_buf)
            raise NotImplementedError

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Bias':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Reshape+Bias':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [None], self.interblob, reshape_param=self.reshape_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.interblob[0]], self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))

        self.setConverted()

        return layers
