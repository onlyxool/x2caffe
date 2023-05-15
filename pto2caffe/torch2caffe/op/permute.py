import torch

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Permute(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'permute')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'Permute'
        self.permute_param = dict()
        self.permute_param['order'] = self.inputs_buf[1]

        self.attrs = self.permute_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]


    def forward(self):
        output = torch.permute(self.model.variable[self.inputs[0]], dims=self.inputs_buf[1])

        self.model.variable[self.outputs[0]] = output
        self.model.tensor_shape[self.outputs[0]] = list(output.shape)

