from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class MatMul(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'aten::matmul')
        self.setInited()


    def parse(self):
        super().__parse__()


        if self.inputs_buf == [None, None]:
            self.type = 'MatMul'

            self.matmul_param = dict()
            self.matmul_param['transpose_a'] = False
            self.matmul_param['transpose_b'] = False
            self.matmul_param['blob_shape'] = self.outputs_shape[0]

            self.attrs = self.matmul_param

            self.setParsed()
        else:
            if len(self.inputs_shape[0]) != 2 or len(self.inputs_shape[1]) != 2:
                raise NotImplementedError('MatMul only support input dimentions == 2')

            self.type = 'InnerProduct'
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
        if self.type == 'InnerProduct':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, inner_product_param=self.inner_product_param)
        elif self.type == 'MatMul':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, matmul_param=self.matmul_param)

        self.setConverted()

        return [layer]
