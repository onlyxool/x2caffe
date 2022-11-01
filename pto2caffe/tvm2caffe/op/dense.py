from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Dense(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'nn.dense')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'InnerProduct'

        self.weight = self.inputs_buf[1]

        self.inner_product_param = dict()
        self.inner_product_param['num_output'] = self.weight.shape[0]
        self.inner_product_param['transpose'] = False

        self.inner_product_param['weight_filler'] = dict()
        self.inner_product_param['weight_filler']['type'] = 'constant'
        self.inner_product_param['bias_term'] = False

        self.attrs = self.inner_product_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, inner_product_param=self.inner_product_param)

        self.setConverted()

        return [layer]
