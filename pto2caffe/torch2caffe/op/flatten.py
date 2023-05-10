import torch
from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Flatten(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'flatten')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.' + str(self.inputs_shape))
            return

        self.outputs_shape[0] = list(torch.flatten(torch.rand(self.outputs_shape[0]), self.inputs_buf[1], self.inputs_buf[2]).shape)
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        self.type = 'Flatten'
        super().__parse__()
        self.compute_output_shape()

        self.flatten_param = dict()
        self.flatten_param['axis'] = self.inputs_buf[1]
        self.flatten_param['end_axis'] = self.inputs_buf[2]
        self.attrs = self.flatten_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, flatten_param=self.flatten_param)

        self.setConverted()

        return [layer]
