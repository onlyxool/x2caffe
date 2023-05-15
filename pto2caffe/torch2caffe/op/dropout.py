import torch
from torch.nn.functional import dropout

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Dropout(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'dropout')
        self.setInited()


    def parse(self):
        self.type = 'Dropout'
        super().__parse__()

        self.dropout_param = dict()
        self.dropout_param['dropout_ratio'] = self.inputs_buf[1]
        self.attrs = self.dropout_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, dropout_param=self.dropout_param)

        self.setConverted()

        return [layer]


    def forward(self):
        output = dropout(self.model.variable[self.inputs[0]], p=self.inputs_buf[1], training=False, inplace=self.inputs_buf[2])

        self.model.variable[self.outputs[0]] = output
        self.model.tensor_shape[self.outputs[0]] = list(output.shape)
