import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from util import handleLegacyPad

logger = logging.getLogger('tflite2caffe')

class Convolution(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.convolution_param = dict()
        self.convolution_param['dilation'] = []
        self.convolution_param['group'] = 1
        self.attrs = self.convolution_param
        self.setInited()


    @property
    def type(self):
        return 'ConvolutionDepthwise' if self.isDepthwise else 'Convolution'


    @property
    def isDepthwise(self):
        return (self.op_code is tflite.BuiltinOperator.DEPTHWISE_CONV_2D)


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 3), "TFLite Conv always has bias"
        assert(self.op.OutputsLength() == 1)

        # Input & OutPut
        self.parseInput()
        self.parseOutput()
        for legacy in self.model.legacys:
            if legacy.op_code == tflite.BuiltinOperator.DEQUANTIZE:
                if legacy.outputs[0] == self.inputs[1]:
                    self.inputs_buf[1] = legacy.inputs_buf[0]
                if legacy.outputs[0] == self.inputs[2]:
                    self.inputs_buf[2] = legacy.inputs_buf[0]

        # Weight
        self.weight = self.inputs_buf[1].transpose(3, 0, 1, 2) if self.isDepthwise else self.inputs_buf[1].transpose(0, 3, 1, 2)
        self.inputs_buf[1] = self.weight

        # Bias
        bias = self.inputs_buf[2]
        if bias is not None and len(bias.shape) == 4:
            self.bias = bias.transpose(3, 0, 1, 2) if self.isDepthwise else bias.transpose(0, 3, 1, 2)
        else:
            self.bias = bias
        self.inputs_buf[2] = self.bias

        # Option
        op_opt = self.op.BuiltinOptions()
        opt = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        self.convolution_param['num_output'] = self.outputs_shape[0][1]
        self.convolution_param['stride_h'] = opt.StrideH()
        self.convolution_param['stride_w'] = opt.StrideW()
        self.convolution_param['dilation'] = [opt.DilationHFactor(), opt.DilationWFactor()]
        self.convolution_param['group'] = self.inputs_shape[0][1] if self.isDepthwise else 1
        self.convolution_param['kernel_h'] = self.weight.shape[2]
        self.convolution_param['kernel_w'] = self.weight.shape[3]
        self.convolution_param['bias_term'] = True if self.bias is not None else False

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

        padding = handleLegacyPad(padding_mode, self.inputs_shape[0], self.outputs_shape[0], self.convolution_param, legacy_pad, self.type)
        if len(padding) == 2:
            self.convolution_param['pad_w'] = padding[0]
            self.convolution_param['pad_h'] = padding[1]
        elif len(padding) == 4:
            self.convolution_param['pad_l'] = padding[0]
            self.convolution_param['pad_r'] = padding[1]
            self.convolution_param['pad_t'] = padding[2]
            self.convolution_param['pad_b'] = padding[3]

        # Fused Activation
        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
