import tflite
import logging
import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Binary(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.ADD: 'Add',
        tflite.BuiltinOperator.MUL: 'Mul',
        tflite.BuiltinOperator.DIV: 'Div',
        tflite.BuiltinOperator.SUB: 'Sub',
        tflite.BuiltinOperator.POW: 'Pow',
    }

    OptionMapping = {
        tflite.BuiltinOperator.ADD: tflite.AddOptions,
        tflite.BuiltinOperator.MUL: tflite.MulOptions,
        tflite.BuiltinOperator.DIV: tflite.DivOptions,
        tflite.BuiltinOperator.SUB: tflite.SubOptions,
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()

    @property
    def type(self):
        if self.op_code == tflite.BuiltinOperator.ADD:
            if hasattr(self, 'eltwise_param'):
                return 'Eltwise'
            elif hasattr(self, 'scale_param'):
                return 'Scale'
        elif self.op_code == tflite.BuiltinOperator.SUB:
            return 'TODO:SUB'
        elif self.op_code == tflite.BuiltinOperator.MUL:
            return 'Scale'
        elif self.op_code == tflite.BuiltinOperator.DIV:
            return 'TODO:DIV'
        elif self.op_code == tflite.BuiltinOperator.POW:
            return 'TODO:POW'
        else:
            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in self.TypeMapping)
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        op_opt = self.op.BuiltinOptions()
        if self.op_code == tflite.BuiltinOperator.ADD:
            opt = tflite.AddOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            if self.inputs_buf[1] is not None:
                self.scale_param = dict()
                self.weight = np.ones(self.inputs_shape[1], dtype=int, order='C')
                self.bias = self.inputs_buf[1]
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
                self.scale_param['bias_term'] = True
                self.attrs = self.scale_param
            else:
                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 1
                self.attrs = self.eltwise_param
        elif self.op_code == tflite.BuiltinOperator.SUB:
            opt = tflite.SubOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            raise NotImplementedError('SubOptions')
        elif self.op_code == tflite.BuiltinOperator.MUL:
            opt = tflite.MulOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            if self.inputs_buf[1] is not None:
                self.scale_param = dict()
                self.weight = self.inputs_buf[1]
                self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
                self.scale_param['bias_term'] = False
                self.attrs = self.scale_param
            else:
                raise NotImplementedError
        elif self.op_code == tflite.BuiltinOperator.DIV:
            opt = tflite.DivOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            raise NotImplementedError('DivOptions')
        elif self.op_code == tflite.BuiltinOperator.POW:
            opt = tflite.PowOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            raise NotImplementedError('PowOptions')

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        if self.op_code == tflite.BuiltinOperator.ADD:
            if hasattr(self, 'eltwise_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
            elif hasattr(self, 'scale_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)
        elif self.op_code == tflite.BuiltinOperator.MUL:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
