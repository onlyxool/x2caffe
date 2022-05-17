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

        op_opt = self.op.BuiltinOptions()
        opt = tflite.AddOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        self.parseInputOutput()

        # Attributes
        if self.inputs_buf[1] is not None:
            # Scale Layer
            self.layer_type = 'Scale'
            self.scale_param = dict()
            self.weight = np.ones(self.inputs_shape[1], dtype=int, order='C')
            self.bias = self.inputs_buf[1]
            if self.inputs_shape[1] != []:
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
            self.scale_param['bias_term'] = True
            self.attrs = self.scale_param
        else:
            # Eltwise Layer
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1
            self.attrs = self.eltwise_param

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Scale':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
