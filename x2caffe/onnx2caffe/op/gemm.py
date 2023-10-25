from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class InnerProduct(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Gemm')
        self.setInited()


    def parse(self):
        self.type = 'InnerProduct'
        super().__parse__()

        if (isinstance(self.inputs_shape[0], list) and len(self.inputs_shape[0]) != 2) or (isinstance(self.inputs_shape[1], list) and len(self.inputs_shape[1]) != 2):
            self.unSupported('only support input dimentions == 2')
            return

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2] if len(self.inputs_buf) == 3 else None

        # Attributes
        alpha = self.attrs.get('alpha', 1.0)
        if alpha != 1.0:
            self.weight = self.weight * alpha

        beta = self.attrs.get('beta', 1.0)
        if beta != 1.0 and self.bias is not None:
            self.bias = self.bias * beta

        transA = self.attrs.get('transA', 0)
        if transA != 0:
            self.type = 'Permute+InnerProduct'
            self.permute_param = dict(order=[1,0])

        transB = self.attrs.get('transB', 0)
        if transB == 0:
            self.weight = self.weight.transpose(1,0)

        self.inner_product_param = dict()
        self.inner_product_param['num_output'] = self.weight.shape[0]
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
        layers = list()
        if self.type == 'InnerProduct':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param))
        elif self.type == 'Permute+InnerProduct':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[0]], [None], self.interblob, permute_param=self.permute_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param))

        self.setConverted()

        return layers
