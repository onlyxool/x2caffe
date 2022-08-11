import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

from util import handleLegacyPad


class Deconvolution(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)

        assert(self.operator_code == 'TRANSPOSE_CONV')
        assert(self.op.InputsLength() == 3), "TFLite Conv always has bias"
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.layer_type = 'Deconvolution'

        op_opt = self.op.BuiltinOptions()
        opt = tflite.TransposeConvOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        self.parseInputOutput()

        # Weight
        self.weight = self.inputs_buf[1].transpose(0, 3, 1, 2)
        self.inputs_buf[1] = self.weight
        self.inputs_shape[1] = list(self.inputs_buf[1].shape)

        # Attributes
        self.convolution_param = dict()
        self.convolution_param['num_output'] = self.outputs_shape[0][1]
        self.convolution_param['stride_h'] = opt.StrideH()
        self.convolution_param['stride_w'] = opt.StrideW()
#        self.convolution_param['dilation'] = [1, 1]
#        self.convolution_param['group'] = 1
        self.convolution_param['kernel_h'] = self.weight.shape[2]
        self.convolution_param['kernel_w'] = self.weight.shape[3]
        self.convolution_param['bias_term'] = False

        # Padding
        if opt.Padding() == tflite.Padding.VALID:
            padding_mode = 'VALID'
        elif opt.Padding() == tflite.Padding.SAME:
            padding_mode = 'SAME'

        legacy_pad = self.model.pad.get(self.inputs[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        padding = handleLegacyPad(padding_mode, self.inputs_shape[2], self.outputs_shape[0], self.convolution_param, legacy_pad, self.type)
        self.convolution_param.update(padding)

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
