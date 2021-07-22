import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from tflite2caffe.op.pad import computePaddingSize

logger = logging.getLogger('tflite2caffe')

class Convolution(Operator):
    def __init__(self, tfmodel, tfgraph, tf_op, tf_op_code, index, legacys):
        super().__init__(tfmodel, tfgraph, tf_op, tf_op_code, index, legacys)
        self.convolution_param = dict()
        self.convolution_param['dilation'] = []
        self.convolution_param['group'] = 1
        self.attrs = self.convolution_param
        self.setInited()

    @property
    def isDepthwise(self):
        return (self.op_code is tflite.BuiltinOperator.DEPTHWISE_CONV_2D)

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 3), "TFLite Conv always has bias"
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Weight
        self.weight = self.inputs_buf[1].transpose(3, 0, 1, 2) if self.isDepthwise else self.inputs_buf[1].transpose(0, 3, 1, 2)

        # Bias
        bias = self.inputs_buf[2]
        if bias is not None and len(bias.shape) == 4:
            self.bias = bias.transpose(3, 0, 1, 2) if self.isDepthwise else bias.transpose(0, 3, 1, 2)
        else:
            self.bias = bias

        # Option
        op_opt = self.op.BuiltinOptions()
        opt = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        self.convolution_param['num_output'] = self.outputs_shape[0][1]#self.graph.Tensors(self.outputs[0]).Shape(3)
        self.convolution_param['stride_h'] = opt.StrideH()
        self.convolution_param['stride_w'] = opt.StrideW()
        self.convolution_param['dilation'] = [opt.DilationHFactor(), opt.DilationWFactor()]
        self.convolution_param['group'] = self.inputs_shape[0][1] if self.isDepthwise else 1
        self.convolution_param['kernel_h'] = self.weight.shape[2]
        self.convolution_param['kernel_w'] = self.weight.shape[3]
        self.convolution_param['bias_term'] = True if self.bias is not None else False
        if self.isDepthwise is True:
            self.convolution_param['engine'] = 1

        legacy_pad = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        for legacy in self.legacys:
            if legacy.outputs[0] == self.inputs[0]:
                legacy_pad = legacy.pad
                self.inputs[0] = legacy.inputs[0]
        padding = computePaddingSize(opt.Padding(), self.inputs_shape[0], self.outputs_shape[0], self.convolution_param, legacy_pad)
        if len(padding) == 2:
            self.convolution_param['pad_w'] = padding[0]
            self.convolution_param['pad_h'] = padding[1]
        elif len(padding) == 4:
            self.convolution_param['pad_l'] = padding[0]
            self.convolution_param['pad_r'] = padding[1]
            self.convolution_param['pad_t'] = padding[2]
            self.convolution_param['pad_b'] = padding[3]
            if self.isDepthwise is True:
                raise NotImplementedError("Depthwise Convolution not support asymmetric padding")

        activ_type_code = opt.FusedActivationFunction()
        if activ_type_code is not tflite.ActivationFunctionType.NONE:
            self.activ_type_code = activ_type_code

        self.setParsed()
        
    @property
    def type(self):
        return 'ConvolutionDepthwise' if self.isDepthwise else 'Convolution'

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)
        self.setConverted()
        return layer
