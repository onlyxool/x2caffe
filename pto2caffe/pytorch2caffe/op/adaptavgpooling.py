import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class AdaptiveAvgPooling(Operator):

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

        input_h = self.inputs_shape[0][2]
        input_w = self.inputs_shape[0][3]

        output_h = self.attrs['output_size'][0]
        output_w = self.attrs['output_size'][1]

        self.pooling_param['pool'] = 1

        self.pooling_param['stride_h'] = stride_h = input_h // output_h
        self.pooling_param['stride_w'] = stride_w = input_w // output_w

        self.pooling_param['kernel_h'] = input_h - (output_h - 1) * stride_h
        self.pooling_param['kernel_w'] = input_w - (output_w - 1) * stride_w

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
