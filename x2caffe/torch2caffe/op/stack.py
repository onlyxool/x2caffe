import torch
import numpy as np
from copy import deepcopy

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Stack(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'stack')
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
            self.saveConstant(self.outputs[0], np.stack(self.inputs_buf[:-1], axis=self.inputs_buf[-1])) 
        else:
            self.type = ('Reshape+' * (len(self.inputs[:-1]))) + 'Concat'

            axis = self.inputs_buf[-1] if self.inputs_buf[-1] > 0 else len(self.inputs_shape[0]) + self.inputs_buf[-1] + 1
            inter_shape = deepcopy(self.inputs_shape[0])
            inter_shape.insert(axis, 1)
            self.reshape_param = dict(shape=dict(dim=inter_shape))

            self.concat_param = dict()
            self.concat_param['axis'] = self.inputs_buf[-1]

            self.attrs = self.concat_param

            self.setParsed()


    def convert(self):
        layers = list()
        for i, input_name in enumerate(self.inputs[:-1]):
            layers.append(caffe_layer(self.layer_type[i], self.name[i], [self.inputs[i]], self.inputs_buf, [self.interblob[i]], reshape_param=self.reshape_param))
            index = i + 1

        layers.append(caffe_layer(self.layer_type[index], self.name[index], self.interblob, self.inputs_buf, self.outputs, concat_param=self.concat_param))

        self.setConverted()

        return layers


    def forward(self):
        inputs = list()
        for input_name in self.inputs[:-1]:
            inputs.append(self.model.variable[input_name])

        return torch.stack(inputs, dim=self.inputs_buf[-1])
