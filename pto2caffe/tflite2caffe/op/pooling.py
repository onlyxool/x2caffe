import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from util import handleLegacyPad

logger = logging.getLogger('tflite2caffe')


class Pooling(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.pooling_param = dict()
        self.attrs = self.pooling_param
        self.setInited()


    @property
    def type(self):
        return 'Pooling'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in (tflite.BuiltinOperator.AVERAGE_POOL_2D, tflite.BuiltinOperator.MAX_POOL_2D))
        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Attributes
        op_opt = self.op.BuiltinOptions()
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        self.pooling_param['pool'] = 1 if self.op_code == tflite.BuiltinOperator.AVERAGE_POOL_2D else 0
        self.pooling_param['kernel_h'] = opt.FilterHeight()
        self.pooling_param['kernel_w'] = opt.FilterWidth()
        self.pooling_param['stride_h'] = opt.StrideH()
        self.pooling_param['stride_w'] = opt.StrideW()
        self.pooling_param['ceil_mode'] = True if opt.Padding() == tflite.Padding.SAME else False

        # Padding
        legacy_pad = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        for legacy in self.model.legacys:
            if legacy.op_code == tflite.BuiltinOperator.PAD:
                if legacy.outputs[0] == self.inputs[0]:
                    legacy_pad = legacy.pad
                    self.inputs[0] = legacy.inputs[0]
                    self.inputs_shape[0] = legacy.inputs_shape[0]

        if opt.Padding() == tflite.Padding.VALID:
            padding_mode = 'VALID'
        elif opt.Padding() == tflite.Padding.SAME:
            padding_mode = 'SAME'

        padding = handleLegacyPad(padding_mode, self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.type)
        if len(padding) == 2:
            self.pooling_param['pad_w'] = padding[0]
            self.pooling_param['pad_h'] = padding[1]
        elif len(padding) == 4:
            self.pooling_param['pad_l'] = padding[0]
            self.pooling_param['pad_r'] = padding[1]
            self.pooling_param['pad_t'] = padding[2]
            self.pooling_param['pad_b'] = padding[3]

        # FusedActivation
        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
