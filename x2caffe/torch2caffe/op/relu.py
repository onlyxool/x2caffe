from torch.nn.functional import relu

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class ReLU(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'relu')
        self.setInited()


    def parse(self):
        self.type = 'ReLU'
        super().__parse__()

        self.relu_param = dict()
        self.relu_param['negative_slope'] = 0

        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return relu(self.model.variable[self.inputs[0]])
