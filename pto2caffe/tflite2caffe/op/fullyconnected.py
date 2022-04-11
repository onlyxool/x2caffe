import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class InnerProduct(Operator):
    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.inner_product_param = dict()
        self.attrs = self.inner_product_param
        self.setInited()

    @property
    def type(self):
        return 'InnerProduct'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 3), "TFLite Fullly Connected always has bias"
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()
        for legacy in self.model.legacys:
            if legacy.op_code == tflite.BuiltinOperator.DEQUANTIZE:
                if legacy.outputs[0] == self.inputs[1]:
                    self.inputs_buf[1] = legacy.inputs_buf[0]
                if legacy.outputs[0] == self.inputs[2]:
                    self.inputs_buf[2] = legacy.inputs_buf[0]

        # Weight
        weight = self.inputs_buf[1]
        if weight is not None and len(weight.shape) == 4:
            self.weight = self.inputs_buf[1].transpose(0, 3, 1, 2)
        else:
            self.weight = weight
        self.inputs_buf[1] = self.weight

        # Bias
        bias = self.inputs_buf[2]
        if bias is not None and len(bias.shape) == 4:
            self.bias = bias.transpose(0, 3, 1, 2)
        else:
            self.bias = bias
        self.inputs_buf[2] = self.bias

        # Options
        op_opt = self.op.BuiltinOptions()
        opt = tflite.FullyConnectedOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
#        print(opt.WeightsFormat(), opt.KeepNumDims(), opt.FusedActivationFunction(), opt.AsymmetricQuantizeInputs())
        self.inner_product_param['num_output'] = self.outputs_shape[0][1]
        self.inner_product_param['weight_filler'] = dict()
        self.inner_product_param['weight_filler']['type'] = 'xavier'
        if self.bias is not None:
            self.inner_product_param['bias_term'] = True
            self.inner_product_param['bias_filler'] = dict()
            self.inner_product_param['bias_filler']['type'] = 'constant'
        else:
            self.inner_product_param['bias_term'] = False

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)
        self.setConverted()
        return [layer]
