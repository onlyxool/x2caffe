import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Pooling(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.pooling_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        if self.operator == 'nn.MaxPool2d':
            self.pooling_param['pool'] = 0 
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_size'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_size'][1]
        elif self.operator == 'nn.AvgPool2d':
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_size'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_size'][1]
        else:
            raise NotImplementedError(self.operator)

        if 'dilations' in self.attrs and self.attrs['dilations'] != [1, 1]:
            errorMsg = 'Caffe Pooling don\'t support dilation' + self.attrs['dilations']
            raise NotImplementedError(errorMsg)

        strides = self.attrs.get('stride', [1, 1])
        self.pooling_param['stride_h'] = strides[0]
        self.pooling_param['stride_w'] = strides[1]

        self.pooling_param['ceil_mode'] = self.attrs.get('ceil_mode', False)
        self.pooling_param['pad_h'] = self.attrs.get('padding', [0,0])[0]
        self.pooling_param['pad_w'] = self.attrs.get('padding', [0,0])[1]

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
