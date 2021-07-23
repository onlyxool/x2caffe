import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Binary(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.ADD: 'Add',
        tflite.BuiltinOperator.MUL: 'Mul',
        tflite.BuiltinOperator.SUB: 'Sub',
        tflite.BuiltinOperator.POW: 'Pow',
    }

    OptionMapping = {
        tflite.BuiltinOperator.ADD: tflite.AddOptions,
        tflite.BuiltinOperator.MUL: tflite.MulOptions,
        tflite.BuiltinOperator.SUB: tflite.SubOptions,
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()

    @property
    def type(self):
        if self.op_code == tflite.BuiltinOperator.ADD:
            return 'Eltwise'
        elif self.op_code == tflite.BuiltinOperator.SUB:
            return 'TODO'
        elif self.op_code == tflite.BuiltinOperator.MUL:
            return 'TODO'
        elif self.op_code == tflite.BuiltinOperator.POW:
            return 'TODO'
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
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1
            self.attrs = self.eltwise_param
        elif self.op_code == tflite.BuiltinOperator.SUB:
            opt = tflite.SubOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
        elif self.op_code == tflite.BuiltinOperator.MUL:
            opt = tflite.MulOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
        elif self.op_code == tflite.BuiltinOperator.POW:
            opt = tflite.PowOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass

    def convert(self):
        if self.op_code == tflite.BuiltinOperator.ADD:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        self.setConverted()
        return layer
