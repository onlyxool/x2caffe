import numpy as np
from copy import deepcopy

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Stack(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'stack')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(-1):
            self.unSupported('Illegal Input Shape.' + str(self.inputs_shape))
            return

        self.outputs_shape[0].insert(self.concat_param['axis'], len(self.reshape_param))
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        super().__parse__()
        self.type = ('Reshape+' * (len(self.inputs[:-1]))) + 'Concat'

        axis = self.inputs_buf[-1]

        self.reshape_param = list()
        for index, input_name in enumerate(self.inputs[:-1]):
            inter_shape = deepcopy(self.inputs_shape[index])
            inter_shape.insert(axis, 1)
            self.reshape_param.append(dict(shape=dict(dim=inter_shape)))

        self.concat_param = dict()
        self.concat_param['axis'] = axis

        self.attrs = self.concat_param
        self.compute_output_shape()

        self.setParsed()


    def convert(self):
        layers = list()
        for i, reshap_param in enumerate(self.reshape_param):
            layers.append(self.layer_type[i], self.name[i], self.inputs, self.inputs_buf, self.interblob[i], reshape_param=self.reshape_param[i])
            index = i + 1

        layers.append(caffe_layer(self.layer_type[index], self.name[index], self.interblob, self.inputs_buf, self.outputs, concat_param=self.concat_param))

        self.setConverted()

        return layers
