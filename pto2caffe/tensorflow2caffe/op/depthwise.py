from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import handleLegacyPad


class ConvolutionDepthwise(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'DepthwiseConv2dNative')
        self.setInited()


    def parse(self):
        self.layer_type = 'ConvolutionDepthwise'
        super().__parse__()

        # Weight HWIO -> OIHW
        self.weight = self.inputs_buf[1].transpose(2, 3, 0, 1)

        self.inputs_buf[1] = self.weight
        self.inputs_shape[1] = self.weight.shape

        # Bias
        if len(self.inputs) >= 3:
            self.bias = self.inputs_buf[2]
            self.inputs_buf[2] = self.bias
        else:
            self.bias = None

        # Attribute
        self.convolution_param = dict()
        self.convolution_param['num_output'] = self.outputs_shape[0][1]
        self.convolution_param['stride_h'] = self.attrs['strides'][self.ndim('H')]
        self.convolution_param['stride_w'] = self.attrs['strides'][self.ndim('W')]
        self.convolution_param['dilation'] = [self.attrs['dilations'][self.ndim('H')], self.attrs['dilations'][self.ndim('W')]]
        self.convolution_param['group'] = int(self.inputs_shape[0][1] / self.weight.shape[1])
        self.convolution_param['kernel_h'] = self.weight.shape[2]
        self.convolution_param['kernel_w'] = self.weight.shape[3]
        self.convolution_param['bias_term'] = True if self.bias is not None else False

        # Padding
        legacy_pad = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        for legacy in self.model.legacys:
            if legacy.operator == 'Pad':
                if legacy.outputs[0] == self.inputs[0]:
                    legacy_pad = legacy.pad
                    self.inputs[0] = legacy.inputs[0]
                    self.inputs_shape[0] = legacy.inputs_shape[0]

        padding = handleLegacyPad(self.attrs['padding'], self.inputs_shape[0], self.outputs_shape[0], self.convolution_param, legacy_pad, self.type)
        if len(padding) == 2:
            self.convolution_param['pad_w'] = padding[0]
            self.convolution_param['pad_h'] = padding[1]
        elif len(padding) == 4:
            self.convolution_param['pad_l'] = padding[0]
            self.convolution_param['pad_r'] = padding[1]
            self.convolution_param['pad_t'] = padding[2]
            self.convolution_param['pad_b'] = padding[3]

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
