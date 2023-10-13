from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class MatMul(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'MatMul')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import numpy as np
            self.saveConstant(self.outputs[0], np.matmul(self.inputs_buf[0], self.inputs_buf[1]))
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is not None and (len(self.inputs_shape[0]) == 2 and len(self.inputs_shape[1]) == 2):
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
        else:
            self.type = 'MatMul'
            self.matmul_param = dict()
            self.matmul_param['transpose_a'] = False
            self.matmul_param['transpose_b'] = False

            if self.inputs_buf[1] is not None:
                self.matmul_param['blob_shape'] = list(self.inputs_buf[1].shape)
                self.weight = self.inputs_buf[1]
                self.bias = None

            self.attrs = self.matmul_param
            self.setParsed()


    def convert(self):
        if self.type == 'InnerProduct':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)
        elif self.type == 'MatMul':
            if self.inputs_buf[1] is not None:
                layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight,self.bias, matmul_param=self.matmul_param)
            else:
                layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, matmul_param=self.matmul_param)

        self.setConverted()

        return [layer]
