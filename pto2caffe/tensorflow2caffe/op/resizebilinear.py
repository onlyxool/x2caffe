import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class ResizeBilinear(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'ResizeBilinear')
        self.setInited()


    def parse(self):
        self.layer_type = 'Interp'
        super().__parse__()

        # Attributes
        self.interp_param = dict()
        self.interp_param['align_corners'] = self.attrs['align_corners']
        # self.attrs['half_pixel_centers']

        self.interp_param['height'] = self.outputs_shape[0][-2]
        self.interp_param['width'] = self.outputs_shape[0][-1]

        self.attrs = self.interp_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]
