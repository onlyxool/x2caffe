import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from util import handleLegacyPad

logger = logging.getLogger('tflite2caffe')


class Deconvolution(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.convolution_param = dict()
#        self.convolution_param['group'] = 1
        self.attrs = self.convolution_param
        self.setInited()


    @property
    def type(self):
        return 'Deconvolution'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 3), "TFLite Conv always has bias"
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
        self.weight = self.inputs_buf[1].transpose(0, 3, 1, 2)
        self.inputs_buf[1] = self.weight
        self.inputs_shape[1] = list(self.inputs_buf[1].shape)

        # Option
        op_opt = self.op.BuiltinOptions()
        opt =tflite.TransposeConvOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        self.convolution_param['num_output'] = self.outputs_shape[0][1]
        self.convolution_param['stride_h'] = opt.StrideH()
        self.convolution_param['stride_w'] = opt.StrideW()
        self.convolution_param['dilation'] = [1, 1]
#        self.convolution_param['group'] = self.inputs_shape[2][1]
        self.convolution_param['kernel_h'] = self.weight.shape[2]
        self.convolution_param['kernel_w'] = self.weight.shape[3]
        self.convolution_param['bias_term'] = False

        # Padding
        legacy_pad = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        for legacy in self.model.legacys:
            if legacy.op_code == tflite.BuiltinOperator.PAD:
                if legacy.outputs[0] == self.inputs[0]:
                    legacy_pad = legacy.pad
                    self.inputs[0] = legacy.inputs[0]
                    self.inputs_shape[0] = legacy.inputs_shape[0]

        if opt.Padding() == tflite.Padding.VALID:
            padding_mode == 'VALID'
        elif opt.Padding() == tflite.Padding.SAME:
            padding_mode == 'SAME'

        padding = handleLegacyPad(padding_mode, self.inputs_shape[2], self.outputs_shape[0], self.convolution_param, legacy_pad, self.type)
        if len(padding) == 2:
            self.convolution_param['pad_w'] = padding[0]
            self.convolution_param['pad_h'] = padding[1]
        elif len(padding) == 4:
            self.convolution_param['pad_l'] = padding[0]
            self.convolution_param['pad_r'] = padding[1]
            self.convolution_param['pad_t'] = padding[2]
            self.convolution_param['pad_b'] = padding[3]

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]