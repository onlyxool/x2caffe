import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from util import handleLegacyPad


class Pooling(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator_code in ('AVERAGE_POOL_2D', 'MAX_POOL_2D'))
        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.type = 'Pooling'

        op_opt = self.op.BuiltinOptions()
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        self.parseInputOutput()

        # Attributes
        self.pooling_param = dict()
        self.pooling_param['pool'] = 1 if self.operator_code == 'AVERAGE_POOL_2D' else 0
        self.pooling_param['kernel_h'] = opt.FilterHeight()
        self.pooling_param['kernel_w'] = opt.FilterWidth()
        self.pooling_param['stride_h'] = opt.StrideH()
        self.pooling_param['stride_w'] = opt.StrideW()
        self.pooling_param['ceil_mode'] = True if opt.Padding() == tflite.Padding.SAME else False

        # Padding
        if opt.Padding() == tflite.Padding.VALID:
            padding_mode = 'VALID'
        elif opt.Padding() == tflite.Padding.SAME:
            padding_mode = 'SAME'

        legacy_pad = self.model.pad.get(self.op.Inputs(0), {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        padding = handleLegacyPad(padding_mode, self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.layer_type)
        self.pooling_param.update(padding)

        # FusedActivation
        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
