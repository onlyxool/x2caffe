from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Deconvolution(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'nn.ConvTranspose2d')
        self.setInited()


    def parse(self):
        self.layer_type = 'Deconvolution'
        super().__parse__()

        # Attributes
        self.convolution_param = dict()
        self.convolution_param['num_output'] = self.attrs['out_channels']
        self.convolution_param['stride_h'] = stride = self.attrs.get('stride', [1, 1])[0]
        self.convolution_param['stride_w'] = stride = self.attrs.get('stride', [1, 1])[1]
        self.convolution_param['dilation'] = self.attrs.get('dilation', [1, 1]) 
        self.convolution_param['group'] = self.attrs.get('groups', 1)
        self.convolution_param['kernel_size'] = self.attrs['kernel_size']
        self.convolution_param['bias_term'] = self.attrs.get('bias', False)
        #self.convolution_param['ceil_mode'] = False
        #self.attrs['in_channels']

        output_padding = self.attrs.get('output_padding', [0, 0])
        if output_padding == [0, 0]:
            self.convolution_param['pad_h'] = self.attrs.get('padding', [0,0])[0]
            self.convolution_param['pad_w'] = self.attrs.get('padding', [0,0])[1]
        else:
            self.convolution_param['pad_t'] = self.attrs.get('padding', [0,0])[0]
            self.convolution_param['pad_b'] = self.attrs.get('padding', [0,0])[0] - output_padding[0]
            self.convolution_param['pad_l'] = self.attrs.get('padding', [0,0])[1]
            self.convolution_param['pad_r'] = self.attrs.get('padding', [0,0])[1] - output_padding[1]


        # Weight
        self.weight = self.inputs_buf[self.inputs.index('weight')]

        # Bias
        self.bias = self.inputs_buf[self.inputs.index('bias')] if self.convolution_param['bias_term'] else None

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
