import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class UpsampleBilinear(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'upsample_bilinear2d')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        self.outputs_shape[0] = self.inputs_shape[0][:2]+self.inputs_buf[1] if self.inputs_buf[1] is not None else list(np.array(self.inputs_shape[0]) * np.array([1,1]+self.inputs_buf[2]))
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        self.type = 'Interp'
        super().__parse__()

        self.compute_output_shape()

        self.interp_param = dict()
        self.interp_param['align_corners'] = self.inputs_buf[2]
        self.interp_param['height'] = self.outputs_shape[0][2]
        self.interp_param['width'] = self.outputs_shape[0][3]
        self.attrs = self.interp_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs[:1], self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]
