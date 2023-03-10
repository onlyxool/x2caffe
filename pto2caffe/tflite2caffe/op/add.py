import tflite
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Add(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator_code == 'ADD')
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        super().__parse__()

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

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 0):
                self.byPassOperator()
                return

            BiasShape = self.inputs_shape[1]
            CompatibleFlag = self.checkShapeCompatible()
            if CompatibleFlag == 'Squeeze':
                self.type = 'Reshape+Bias'
            elif not CompatibleFlag:
                self.unSupported('Error: Operator ADD Inputs shape uncompatible for Caffe. ' + str(self.inputs_shape[0]) + ' + ' + str(BiasShape))
                return

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])
            self.attrs = self.bias_param

            self.bias = self.inputs_buf[1]

        self.activ_type_code = opt.FusedActivationFunction()

        self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Bias':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))
        elif self.type == 'Reshape+Bias':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], [None], self.interblob, reshape_param=dict(shape=dict(dim=self.inputs_shape[1]))))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.interblob[0]], self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param))

        self.setConverted()

        return layers
