from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

from onnx2caffe.utility import computePad


class Reduce(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'ReduceMean')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        # Attributes
        self.pooling_param = dict()
        self.pooling_param['pool'] = 1 # Pooling.AVE
        self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
        self.pooling_param['kernel_w'] = self.inputs_shape[0][3] if len(self.inputs_shape[0]) == 4 else 1
        self.pooling_param['stride_h'] = 1
        self.pooling_param['stride_w'] = 1
        self.pooling_param['ceil_mode'] = False

        kernel_size = [self.pooling_param['kernel_h'], self.pooling_param['kernel_w']]

        # Padding
        legacy_pad = self.model.pad.get(self.node.input[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        padding = computePad(self.type, self.attrs, self.inputs_shape[0], self.outputs_shape[0], kernel_size, [1, 1], legacy_pad)
        self.pooling_param.update(padding)

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
