from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class InnerProduct(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.inner_product_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'InnerProduct'
        super().__parse__()

        if self.inputs_shape[0] is None or self.inputs_shape[1] is None:
            errorMsg = 'Input shape of Gemm is None. [' + self.name +']'
            raise NotImplementedError(errorMsg)

        if len(self.inputs_shape[1]) != 2 or len(self.inputs_shape[0]) != 2:
            errorMsg = 'Gemm is supported only for inner_product layer. [' + self.name +']'
            raise NotImplementedError(errorMsg)

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
           self.pre_permute_param = dict(order=[1,0])

        transB = self.attrs.get('transB', 0)
        if transB == 0:
            self.weight = self.weight.transpose(1,0)

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
        layers = []
        if hasattr(self, 'pre_permute_param'):
            pre_layer = caffe_layer('Premute', 'Permute'+str(self.index)+'pre', [self.inputs[0]], [None], [self.inputs[0]+'pre'], permute_param=self.pre_permute_param)
            layers.append(pre_layer)
            layer = caffe_layer(self.type, self.name, [self.inputs[0]+'pre'], self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)
        else:
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)
            layers.append(layer)

        self.setConverted()

        return layers
