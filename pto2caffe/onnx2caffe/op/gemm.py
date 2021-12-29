import logging

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class InnerProduct(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.inner_product_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'InnerProduct'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2] if len(self.inputs_buf) == 3 else None

        # Options
        self.parseAttributes()        
        if self.attrs['transB'] != 1:
            raise NotImplementedError(self.name, 'Gemm is supported only for inner_product layer')
        if len(self.inputs_shape[1]) != 2 or (self.bias is not None and len(self.bias.shape) != 1):
            raise NotImplementedError(self.name, 'Gemm is supported only for inner_product layer')

        self.inner_product_param['num_output'] = self.inputs_shape[1][0]
        self.inner_product_param['weight_filler'] = dict()
        self.inner_product_param['weight_filler']['type'] = 'constant'#'xavier'
        if self.bias is not None:
            self.inner_product_param['bias_term'] = True
            self.inner_product_param['bias_filler'] = dict()
            self.inner_product_param['bias_filler']['type'] = 'constant'
            if self.inputs_shape[1][0] != self.bias.shape[0]:
                raise NotImplementedError(self.name, 'Gemm is supported only for inner_product layer')
        else:
            self.inner_product_param['bias_term'] = False

        self.attrs = self.inner_product_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)
        self.setConverted()
        return [layer]
