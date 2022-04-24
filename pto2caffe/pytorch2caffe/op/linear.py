import logging

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator

logger = logging.getLogger('Pytorch2Caffe')


class Linear(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.inner_product_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'InnerProduct'
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        self.weight = self.inputs_buf[self.inputs.index('weight')]
        self.bias = self.inputs_buf[self.inputs.index('bias')] if self.attrs['bias'] else None

        self.inner_product_param['num_output'] = self.attrs['out_features']
        self.inner_product_param['axis'] = self.inputs_shape[0].index(self.weight.shape[1])
        self.inner_product_param['transpose'] = False

        self.inner_product_param['weight_filler'] = dict()
        self.inner_product_param['weight_filler']['type'] = 'constant'

        if self.attrs['bias']:
            self.inner_product_param['bias_term'] = True
            self.inner_product_param['bias_filler'] = dict()
            self.inner_product_param['bias_filler']['type'] = 'constant'
        else:
            self.inner_product_param['bias_term'] = False

        #self.attrs['in_features']

        self.attrs = self.inner_product_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)

        self.setConverted()

        return [layer]
