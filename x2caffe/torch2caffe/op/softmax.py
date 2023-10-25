from torch.nn.functional import softmax

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Softmax(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'softmax')
        self.setInited()


    def parse(self):
        self.type = 'Softmax'
        super().__parse__()

        self.softmax_param = dict()
        self.softmax_param['axis'] = self.inputs_buf[1]

        self.attrs = self.softmax_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs[:1], self.inputs_buf, self.outputs, softmax_param=self.softmax_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return softmax(self.model.variable[self.inputs[0]], dim=self.inputs_buf[1], _stacklevel=3, dtype=None)
