from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class MatMul(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'MatMul')
        self.setInited()


    def parse(self):
        self.layer_type = 'InnerProduct'
        super().__parse__()

        if len(self.inputs_shape[0]) != 2 or len(self.inputs_shape[1]) != 2:
            raise NotImplementedError('MatMul don\'t support input dimentions > 2')

        # Weight
        self.weight = self.inputs_buf[1].transpose(1,0)

        # Bias
        self.bias = None

        self.inner_product_param = dict()
        self.inner_product_param['num_output'] = self.weight.shape[0]
        self.inner_product_param['transpose'] = False

        self.inner_product_param['weight_filler'] = dict()
        self.inner_product_param['weight_filler']['type'] = 'constant'

        self.inner_product_param['bias_term'] = False

        self.attrs = self.inner_product_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)

        self.setConverted()

        return [pre_layer, layer, post_layer]
