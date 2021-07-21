import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class InnerProduct(Operator):
    def __init__(self, tfmodel, tfgraph, tf_op, tf_op_code, index, legacys):
        super().__init__(tfmodel, tfgraph, tf_op, tf_op_code, index, legacys)
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

        # Weight
        weight = self.inputs_buf[1]
        if weight is not None and len(weight.shape) == 4:
            self.weight = self.inputs_buf[1].transpose(0, 3, 1, 2)
        else:
            self.weight = weight

        # Bias
        bias = self.inputs_buf[2]
        if bias is not None and len(bias.shape) == 4:
            self.bias = bias.transpose(0, 3, 1, 2)
        else:
            self.bias = bias

        # options
        op_opt = self.op.BuiltinOptions()
        opt = tflite.FullyConnectedOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        self.inner_product_param['num_output'] = self.outputs_shape[0][1]
        self.inner_product_param['weight_filler'] = dict()
        self.inner_product_param['weight_filler']['type'] = 'xavier'
        self.inner_product_param['bias_filler'] = dict()
        self.inner_product_param['bias_filler']['type'] = 'constant'

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            print(__file__, 'TODO: FusedActivationFunction:', activ_type_code)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)
        self.setConverted()
        return layer
