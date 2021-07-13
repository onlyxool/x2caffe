import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2onnx')


class Activation(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.LOGISTIC: 'Sigmoid',
        tflite.BuiltinOperator.PRELU: 'PRelu',
        tflite.BuiltinOperator.RELU6: 'ReluX',
        tflite.BuiltinOperator.RELU: 'Relu',
        tflite.BuiltinOperator.LEAKY_RELU: 'Leaky_Relu',
    }

    def __init__(self, tfmodel, tfgraph, tf_op, tf_op_code, index, legacy):
        super().__init__(tfmodel, tfgraph, tf_op, tf_op_code, index, legacy)

        self.setInited()

    @property
    def type(self):
        if self.op_code.BuiltinCode() == tflite.BuiltinOperator.LEAKY_RELU:
            return 'ReLU'
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.LOGISTIC:
            return 'Sigmoid'
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.PRELU:
            return 'PReLU'
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.RELU6:
            return 'ReLUX'
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.RELU:
            return 'ReLU'
        else:
            raise NotImplementedError

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code.BuiltinCode() in self.TypeMapping)
        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        if self.op_code.BuiltinCode() == tflite.BuiltinOperator.LEAKY_RELU:
            op_opt = self.op.BuiltinOptions()
            opt = tflite.LeakyReluOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            self.relu_param = dict()
            self.relu_param['negative_slope'] = opt.Alpha()
            self.attrs = self.relu_param
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.LOGISTIC:
            print('sigmoid')
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.PRELU:
            print('prelu')
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.RELU6:
            print('relu6')
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.RELU:
            print('relu')
        else:
            print('Error', self.op_code.BuiltinCode())

        self.setParsed()


    def propagatableTensors(self):
        return self.inputs + self.outputs


    def transform(self):
        pass


    def convert(self):
        if self.op_code.BuiltinCode() == tflite.BuiltinOperator.LEAKY_RELU:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.LOGISTIC:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.PRELU:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, prelu_param=self.prelu_param)
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.RELU6:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)
        elif self.op_code.BuiltinCode() == tflite.BuiltinOperator.RELU:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        else:
            raise NotImplementedError

        self.setConverted()
        return layer

