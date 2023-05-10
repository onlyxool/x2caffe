import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Concat(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'cat')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(-1):
            self.unSupported('Illegal Input Shape.' + str(self.inputs_shape))
            return

        self.outputs_shape[0][self.concat_param['axis']] = np.sum([shape[self.concat_param['axis']] for shape in self.inputs_shape[:-1]])
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


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
            self.compute_output_shape()

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs[:-1], self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]
