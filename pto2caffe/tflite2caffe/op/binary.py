import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2onnx')

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

    def __init__(self, tfmodel, tfgraph, tf_op, tf_op_code, index, legacy):
        super().__init__(tfmodel, tfgraph, tf_op, tf_op_code, index, legacy)

        self.setInited()

    @property
    def type(self):
        if self.op_code.BuiltinCode() == tflite.BuiltinOperator.ADD:
            return 'Eltwise'
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.SUB:
            return 'TODO'
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.MUL:
            return 'TODO'
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.POW:
            return 'TODO'
        else:
            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code.BuiltinCode() in self.TypeMapping)
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        if self.op_code.BuiltinCode() == tflite.BuiltinOperator.ADD:
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1

#        op = self.tflite
#        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
#        assert(opcode in self.TypeMapping)
#
#        assert(op.InputsLength() == 2)
#        assert(op.OutputsLength() == 1)
#
#        self.parseInput(0)
#        self.parseInput(1)
#        ot = self.parseOutput(0)
#
#        self.fakeBroadcast()
#
#        # options
#        op_opt = op.BuiltinOptions()
#        if opcode in self.OptionMapping:
#            option = self.OptionMapping[opcode]()
#            option.Init(op_opt.Bytes, op_opt.Pos)
#
#            handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass

    def convert(self):
        if self.op_code.BuiltinCode() == tflite.BuiltinOperator.ADD:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)

        self.setConverted()
        return layer
