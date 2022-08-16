import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from util import handleLegacyPad


class Convolution(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator_code in ('CONV_2D', 'DEPTHWISE_CONV_2D'))
        assert(self.op.InputsLength() == 3), "TFLite Conv always has bias"
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    @property
    def isDepthwise(self):
        return (self.operator_code == 'DEPTHWISE_CONV_2D')


    def parse(self):
        self.layer_type = 'ConvolutionDepthwise' if self.isDepthwise else 'Convolution'

        op_opt = self.op.BuiltinOptions()
        opt = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        # Input & OutPut
        self.parseInputOutput()

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

        # Attributes
        self.convolution_param = dict()
        self.convolution_param['num_output'] = self.outputs_shape[0][1]
        self.convolution_param['stride_h'] = opt.StrideH()
        self.convolution_param['stride_w'] = opt.StrideW()
        self.convolution_param['dilation'] = [opt.DilationHFactor(), opt.DilationWFactor()]
        self.convolution_param['group'] = self.inputs_shape[0][1] if self.isDepthwise else 1
        self.convolution_param['kernel_h'] = self.weight.shape[2]
        self.convolution_param['kernel_w'] = self.weight.shape[3]
        self.convolution_param['bias_term'] = True if self.bias is not None else False

        # Padding
        if opt.Padding() == tflite.Padding.VALID:
            padding_mode = 'VALID'
        elif opt.Padding() == tflite.Padding.SAME:
            padding_mode = 'SAME'

        legacy_pad = self.model.pad.get(self.op.Inputs(0), {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        padding = handleLegacyPad(padding_mode, self.inputs_shape[0], self.outputs_shape[0], self.convolution_param, legacy_pad, self.type)
        self.convolution_param.update(padding)

        # Fused Activation
        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
