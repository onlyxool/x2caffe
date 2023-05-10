from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Linear(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'linear')
        self.setInited()


    def parse(self):
        self.type = 'InnerProduct'
        super().__parse__()

        self.weight = self.inputBuf_byName('weight')
        self.bias = self.inputBuf_byName('bias')

        self.inner_product_param = dict()
        self.inner_product_param['num_output'] = self.weight.shape[0]
        self.inner_product_param['axis'] = self.inputs_shape[0].index(self.weight.shape[1])
        self.inner_product_param['transpose'] = False

        self.inner_product_param['weight_filler'] = dict()
        self.inner_product_param['weight_filler']['type'] = 'constant'

        if self.bias is not None:
            self.inner_product_param['bias_term'] = True
            self.inner_product_param['bias_filler'] = dict()
            self.inner_product_param['bias_filler']['type'] = 'constant'
        else:
            self.inner_product_param['bias_term'] = False

        self.attrs = self.inner_product_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)

        self.setConverted()

        return [layer]
