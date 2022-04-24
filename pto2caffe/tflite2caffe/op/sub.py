import copy
import tflite
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

from util import trim_one
from util import compute_scale_axis


class Sub(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        self.setInited()


    def parse(self):
        self.layer_type = 'Scale'

        assert(self.operator == 'SUB')
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Attributes
        op_opt = self.op.BuiltinOptions()
        opt = tflite.SubOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if self.inputs_shape[0] != self.inputs_shape[1] or self.inputs_buf[1] is not None:
            self.scale_param = dict()

            org_shape = copy.deepcopy(self.inputs_shape[1])
            trim = trim_one(org_shape)
            if trim != self.inputs_shape[1]:
                self.pre.append('Reshape')
                self.inputs_shape[1] = trim
                if self.inputs_buf[1] is not None:
                    self.inputs_buf[1] = self.inputs_buf[1].reshape(tuple(trim))

            axis = compute_scale_axis(self.inputs_shape[0], trim)
            if axis is not None:
                self.scale_param['axis'] = axis

            self.weight = np.ones([1], dtype=int, order='C')
            self.scale_param['bias_term'] = True
            self.bias = np.multiply(-1, self.inputs_buf[1])
            self.attrs = self.scale_param
        else:
            raise NotImplementedError(self.operator)


        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        for pre_op in self.pre:
            if pre_op == 'Reshape':
                reshape_param = dict(shape=dict(dim=self.inputs_shape[1]))
                pre_layer = caffe_layer('Reshape', 'Reshape'+str(self.index), [self.inputs[1]], [None], ['reshape'+str(self.index)], reshape_param=reshape_param)
                self.inputs[1] = 'reshape' + str(self.index)

        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
        if len(self.pre) > 0:
            self.setConverted()
            return [pre_layer, layer]

        self.setConverted()

        return [layer]
