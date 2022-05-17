from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


def splitkernel(length):
    for i in range(15, 1, -1):
        num_input = length / pow(i, 2)
        if num_input % 1 == 0:
            return i, int(num_input)

    return None


class MatMul(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'MatMul')
        self.setInited()


    def parse(self):
        self.layer_type = 'Convolution'
        super().__parse__()

        if len(self.inputs_shape[0]) == 2 and len(self.inputs_shape[1]) == 2:
            assert(self.inputs_shape[0][1] == self.inputs_shape[1][0])

            weight_size, num_input = splitkernel(self.inputs_shape[1][0])
            if weight_size is None:
                raise NotImplementedError

            # Weight
            weight_shape = [self.inputs_shape[1][1], num_input, weight_size, weight_size]
            self.weight = self.inputs_buf[1].transpose(1,0).reshape(weight_shape)

            # Bias
            self.bias = None

            # Attributes
            self.convolution_param = dict()
            self.convolution_param['num_output'] = self.weight.shape[0]
            self.convolution_param['stride_h'] = weight_size
            self.convolution_param['stride_w'] = weight_size
            self.convolution_param['dilation'] = [1, 1]
            self.convolution_param['group'] = 1
            self.convolution_param['kernel_size'] = [weight_size, weight_size]
            self.convolution_param['bias_term'] = False

            # pre-layer: Reshape
            pre_shape = [self.inputs_shape[0][0], num_input, weight_size, weight_size]
            self.pre_reshape_param = dict(shape=dict(dim=pre_shape))

            # post-layer: Reshape
            self.post_reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
        else:
            raise NotImplementedError('MatMul don\'t support input dimentions > 2')

        self.setParsed()


    def convert(self):
        pre_layer = caffe_layer('Reshape', 'Reshape'+str(self.index)+'pre', [self.inputs[0]], [None], [self.inputs[0]+'pre'], reshape_param=self.pre_reshape_param)

        layer = caffe_layer(self.type, self.name, [self.inputs[0]+'pre', self.inputs[1]], self.inputs_buf, [self.outputs[0]+'post'], self.weight, self.bias, convolution_param=self.convolution_param)

        post_layer = caffe_layer('Reshape', 'Reshape'+str(self.index)+'post', [self.outputs[0]+'post'], [None], self.outputs, reshape_param=self.post_reshape_param)

        self.setConverted()

        return [pre_layer, layer, post_layer]
