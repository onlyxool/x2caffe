import torch

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Transpose(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'transpose')
        self.setInited()


    def parse(self):
        super().__parse__()

        def swap_elements(list, index1, index2):
            list[index1], list[index2] = list[index2], list[index1]
            return list

        self.type = 'Permute'
        self.permute_param = dict()
        self.permute_param['order'] = swap_elements([i for i in range(len(self.inputs_shape[0]))], self.inputs_buf[1], self.inputs_buf[2])

        self.attrs = self.permute_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return torch.transpose(self.model.variable[self.inputs[0]], dim0=self.inputs_buf[1], dim1=self.inputs_buf[2])
