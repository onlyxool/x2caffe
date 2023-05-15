import torch
import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Concat(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'cat')
        self.setInited()


    def parse(self):
        super().__parse__()

        for input_buf in self.inputs_buf[:-1]:
            if input_buf is not None:
                constant = True
            else:
                constant = False
                break

        if constant:
            self.saveConstant(self.outputs[0], np.concatenate(self.inputs_buf[0], axis=self.inputs_buf[-1]))
        else:
            self.type = 'Concat'
            self.concat_param = dict()
            self.concat_param['axis'] = self.inputs_buf[-1]

            self.attrs = self.concat_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs[:-1], self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]


    def forward(self):
        inputs = list()
        for input_name in self.inputs[:-1]:
            inputs.append(self.model.variable[input_name])
        output = torch.cat(inputs, dim=self.inputs_buf[-1], out=None)

        self.model.variable[self.outputs[0]] = output
        self.model.tensor_shape[self.outputs[0]] = list(output.shape)
