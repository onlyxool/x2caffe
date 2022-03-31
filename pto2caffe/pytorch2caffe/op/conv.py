import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Convolution(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.convolution_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Convolution'


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.parseInput()
        self.parseOutput()

        # Weight
        self.weight = self.inputs_buf[self.inputs.index('weight')]

        # Attributes
        self.parseAttributes()
        self.convolution_param['num_output'] = self.attrs['out_channels']#self.weight.shape[0]
        self.convolution_param['stride'] = stride = self.attrs.get('stride', [1, 1]) 
        self.convolution_param['dilation'] = self.attrs.get('dilation', [1, 1]) 
        self.convolution_param['group'] = self.attrs.get('groups', 1)
        self.convolution_param['kernel_size'] = self.attrs['kernel_size']
        self.convolution_param['bias_term'] = self.attrs.get('bias', False)
        self.convolution_param['pad_h'] = self.attrs.get('padding', [0,0])[0]
        self.convolution_param['pad_w'] = self.attrs.get('padding', [0,0])[1]
        #self.attrs['in_channels']

        # Bias
        self.bias = self.inputs_buf[self.inputs.index('bias')] if self.convolution_param['bias_term'] else None

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
