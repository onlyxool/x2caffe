from torch.nn.functional import silu

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Silu(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'silu')
        self.setInited()


    def parse(self):
        self.type = 'Swish'
        super().__parse__()

        self.swish_param = dict()
        self.swish_param['beta'] = 1.0 

        self.attrs = self.swish_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, swish_param=self.swish_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return silu(self.model.variable[self.inputs[0]], inplace=False)
