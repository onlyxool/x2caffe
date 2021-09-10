import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class Activation(Operator):

    TypeMapping = {
        tflite.BuiltinOperator.LOGISTIC: 'Sigmoid',
        tflite.BuiltinOperator.PRELU: 'PRelu',
        tflite.BuiltinOperator.RELU6: 'ReluX',
        tflite.BuiltinOperator.RELU: 'Relu',
        tflite.BuiltinOperator.LEAKY_RELU: 'Leaky_Relu',
    }


    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()


    @property
    def type(self):
        if self.op_code == tflite.BuiltinOperator.LEAKY_RELU:
            return 'ReLU'
        elif self.op_code == tflite.BuiltinOperator.LOGISTIC:
            return 'Sigmoid'
        elif self.op_code == tflite.BuiltinOperator.PRELU:
            return 'PReLU'
        elif self.op_code == tflite.BuiltinOperator.RELU6:
            return 'ReLUX'
        elif self.op_code == tflite.BuiltinOperator.RELU:
            return 'ReLU'
        else:
            raise NotImplementedError


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in self.TypeMapping)
        if self.op is not None:
            if self.op_code == tflite.BuiltinOperator.PRELU:
                assert(self.op.InputsLength() == 2)
            else:
                assert(self.op.InputsLength() == 1)
            assert(self.op.OutputsLength() == 1)
            self.parseInput()
            self.parseOutput()

        # Option
        if self.op_code == tflite.BuiltinOperator.LEAKY_RELU:
            op_opt = self.op.BuiltinOptions()
            opt = tflite.LeakyReluOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            self.relu_param = dict()
            self.relu_param['negative_slope'] = opt.Alpha()
            self.attrs = self.relu_param
        elif self.op_code == tflite.BuiltinOperator.LOGISTIC:
            self.sigmoid_param = dict()
            self.attrs = self.sigmoid_param
        elif self.op_code == tflite.BuiltinOperator.PRELU:
            self.slope = self.inputs_buf[1].transpose(2, 0, 1)
            self.prelux_param = dict()
            if self.slope.shape[0] == 1:
                self.prelux_param['channel_shared'] = True
            else:
                self.prelux_param['channel_shared'] = False
            self.attrs = self.prelux_param
        elif self.op_code == tflite.BuiltinOperator.RELU6:
            self.relux_param = dict()
            self.relux_param['negative_slope'] = 0
            self.relux_param['x'] = 6
            self.attrs = self.relux_param
        elif self.op_code == tflite.BuiltinOperator.RELU:
            self.relu_param = dict()
            self.relu_param['negative_slope'] = 0
            self.attrs = self.relu_param
        else:
            print('Error', self.op_code)

        self.setParsed()


    def convert(self):
        if self.op_code == tflite.BuiltinOperator.LEAKY_RELU:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        elif self.op_code == tflite.BuiltinOperator.LOGISTIC:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)
        elif self.op_code == tflite.BuiltinOperator.PRELU:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.slope, prelu_param=self.prelu_param)
        elif self.op_code == tflite.BuiltinOperator.RELU6:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)
        elif self.op_code == tflite.BuiltinOperator.RELU:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)
        else:
            raise NotImplementedError

        self.setConverted()

        return [layer]


def handleFusedActivation(preop:Operator):
    if preop.activ_type_code == tflite.ActivationFunctionType.RELU:
        op = Activation(preop.model, None, tflite.BuiltinOperator.RELU, preop.index)
    elif preop.activ_type_code == tflite.ActivationFunctionType.RELU_N1_TO_1:
        raise NotImplementedError('ReluN1To1 is not supported.')
    elif preop.activ_type_code == tflite.ActivationFunctionType.RELU6:
        op = Activation(preop.model, None, tflite.BuiltinOperator.RELU6, preop.index)
    elif preop.activ_type_code == tflite.ActivationFunctionType.TANH:
        op = Activation(preop.model, None, tflite.BuiltinOperator.TANH, preop.index)
    elif preop.activ_type_code == tflite.ActivationFunctionType.SIGN_BIT:
         raise NotImplementedError('SignBits is not supported.')
    else:
        return

    for output in preop.outputs:
        op.outputs.append(output)
        op.inputs.append(output)
    op.inputs_buf.append(None)
    op.parse()

    return op
