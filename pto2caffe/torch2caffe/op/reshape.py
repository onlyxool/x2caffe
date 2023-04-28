import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'reshape')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        self.outputs_shape[0] = list(np.zeros(self.inputs_shape[0]).reshape(self.inputs_buf[1]).shape)
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        super().__parse__()

        self.type = 'Reshape'

        self.compute_output_shape()

        self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

        self.attrs = self.reshape_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
