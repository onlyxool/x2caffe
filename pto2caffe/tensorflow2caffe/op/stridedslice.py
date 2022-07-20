import sys
import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw

class StridedSlice(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'StridedSlice')
        self.setInited()


    def parse(self):
        self.layer_type = 'Slice'
        super().__parse__()

        if self.inputs_buf[3].size == list(self.inputs_buf[3]).count(1):

            self.slice_param = dict()

            axis_index = np.nonzero(self.inputs_buf[2] - self.inputs_buf[1])[0]
            if axis_index.size > 1:
                errorMsg = 'Error[' + self.op.name + ']: Can\'t slice more than one axis'
                sys.exit()
            else:
                axis_nhwc = int(axis_index[0])
                self.slice_param['axis'] = dim_map_nhwc2nchw[axis_nhwc]

            self.slice_param['slice_point'] = list(np.absolute(self.inputs_buf[2] - self.inputs_buf[1]))[axis_nhwc]
            self.outputs.append(self.name+'_useless')
        else:
            errorMsg = 'Error[' + self.op.name + ']: Do not support stride > 1. OP ' + self.op.name + '\'s strides is ' + str(self.inputs_buf[3]) + '\n'
            sys.exit(errorMsg)

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]