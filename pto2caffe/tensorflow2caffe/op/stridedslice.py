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
        super().__parse__()

        if self.inputs_buf[0] is not None:
            if self.inputs_buf[0].ndim == 1:
                begin = int(self.inputs_buf[1])
                end = int(self.inputs_buf[2])
                strides = int(self.inputs_buf[3])
                self.model.constant[self.outputs[0]] = self.inputs_buf[0][begin:end:strides]
            else:
                raise NotImplementedError
        else:
            if self.inputs_buf[3].size == list(self.inputs_buf[3]).count(1):
                axis_index = np.nonzero(self.inputs_buf[2] - self.inputs_buf[1])[0]
                if axis_index.size > 1:
                    errorMsg = 'Error[' + self.op.name + ']: Can\'t slice more than one axis'
                    sys.exit()

                axis = int(axis_index[0]) if self.layout == 'NHWC' else dim_map_nhwc2nchw[int(axis_index[0])]

                if self.inputs_buf[2][axis] == self.op.inputs[0].shape[axis]:
                    self.model.indentity[self.op.outputs[0].name] = self.model.indentity.get(self.op.inputs[0].name, self.op.inputs[0].name)
                else:
                    self.layer_type = 'Slice'
                    self.slice_param = dict()
                    self.slice_param['axis'] = axis
                    self.slice_param['slice_point'] = list(np.absolute(self.inputs_buf[2] - self.inputs_buf[1]))[axis]
                    self.outputs.append(self.name+'_useless')
                    self.attrs = self.slice_param
                    self.setParsed()
            else:
                errorMsg = 'Error[' + self.op.name + ']: Do not support stride > 1. OP ' + self.op.name + '\'s strides is ' + str(self.inputs_buf[3]) + '\n'
                sys.exit(errorMsg)




    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
