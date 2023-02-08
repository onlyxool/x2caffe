import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class InstanceNormalization(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'InstanceNormalization')
        self.setInited()


    def parse(self):
        self.type = 'BatchNorm+Scale'
        super().__parse__()

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2]

        # Gamma
        self.gamma = np.ones(self.inputs_shape[0][1:2]).astype(np.float32)

        # Beta
        self.beta = np.zeros(self.inputs_shape[0][1:2]).astype(np.float32)

        # Attributes
        self.batch_norm_param = dict()
        self.batch_norm_param['eps'] = self.attrs.get('epsilon', 1e-5)
        self.batch_norm_param['use_global_stats'] = False

        self.scale_param = dict()
        self.scale_param['bias_term'] = True

        self.attrs = self.batch_norm_param

        self.setParsed()


    def convert(self):
        layer0 = caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.outputs, self.gamma, self.beta, batch_norm_param=self.batch_norm_param)
        layer1 = caffe_layer(self.layer_type[1], self.name[1], self.outputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer0, layer1]

