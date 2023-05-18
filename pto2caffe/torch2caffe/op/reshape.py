import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'reshape')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'Reshape'

        self.reshape_param = dict(shape=dict(dim=self.inputs_buf[1]))

        self.attrs = self.reshape_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return self.model.variable[self.inputs[0]].reshape(self.inputs_buf[1])
