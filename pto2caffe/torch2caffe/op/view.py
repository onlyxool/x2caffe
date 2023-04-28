import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class View(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'view')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        self.outputs_shape[0] = list(np.zeros(self.inputs_shape[0]).reshape(self.inputs_buf[1]).shape)
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import torch
            self.saveConstant(self.outputs[0], torch.Tensor(self.inputs_buf[0]).view(self.inputs_buf[1]).detach().numpy())
        else:
            self.type = 'Reshape'
            self.reshape_param = dict(shape=dict(dim=self.inputs_buf[1]))
            self.attrs = self.reshape_param
    
            self.compute_output_shape()
            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
