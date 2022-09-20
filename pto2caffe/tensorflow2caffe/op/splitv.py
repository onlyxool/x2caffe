import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class SplitV(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'SplitV')
        self.setInited()


    def parse(self):
        self.type = 'Slice'
        super().__parse__()

        for index, input_name in enumerate(self.inputs):
            if self.inputs_buf[index] is not None:
                if self.inputs_buf[index].size > 1:
                    size_splits = self.inputs_buf[index].tolist()
                if self.inputs_buf[index].size == 1:
                    attr_axis = int(self.inputs_buf[index])

        # Attribute
        self.slice_param = dict()
        if self.layout == 'NCHW':
            self.slice_param['axis'] = attr_axis
        elif self.layout == 'NHWC':
            if self.op.inputs[0].shape.as_list() == self.inputs_shape[0]:
                self.slice_param['axis'] = attr_axis   
            else:
                self.slice_param['axis'] = dim_map_nhwc2nchw[attr_axis]
        else:
            raise NotImplementedError

        slice_point = list()
        size_splits = size_splits[:-1]
        for index, size in enumerate(size_splits):
            slice_point.append(int(np.sum(size_splits[0:index+1])))

        self.slice_param['slice_point'] = slice_point

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
