from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class PReLU(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'nn.prelu')
        self.setInited()


    def parse(self):
        self.type = 'PReLU'
        super().__parse__()

        # Slope
        self.slope = self.inputs_buf[1]
        self.inputs_shape[1] = [1] if self.slope.ndim == 0 else self.slope.shape

        self.prelu_param = dict()
        self.prelu_param['channel_shared'] = True if self.inputs_shape[1][0] == 1 else False
        self.attrs = self.prelu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.slope, prelu_param=self.prelu_param)

        self.setConverted()

        return [layer]
