import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Split(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Split')
        self.setInited()


    def parse(self):
        self.type = 'Slice'
        super().__parse__()

        self.slice_param = dict()

        # Axis
        self.slice_param['axis'] = self.attrs['axis'] if self.attrs['axis'] > 0 else self.attrs['axis'] + len(self.inputs_shape[0])

        # Slice Point
        if 'split' in self.attrs:
            slice_point = self.attrs['split']
        elif len(self.inputs_buf) > 1 and isinstance(self.inputs_buf[1], np.ndarray):
            slice_point = self.inputs_buf[1].tolist
        else:
            slice_point = np.array(self.outputs_shape)[:, self.slice_param['axis']:self.slice_param['axis']+1]

        self.slice_param['slice_point'] = np.cumsum(slice_point).tolist()[:-1]

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]

