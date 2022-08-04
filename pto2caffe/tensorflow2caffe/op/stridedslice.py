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
            if self.inputs_buf[1].size == 1:
                begin = int(self.inputs_buf[1])
                end = int(self.inputs_buf[2])
                strides = int(self.inputs_buf[3])
                if self.inputs_buf[0][begin:end:strides].size == 1:
                    self.model.constant[self.outputs[0]] = np.array(self.inputs_buf[0][begin:end:strides].item())
                else:
                    self.model.constant[self.outputs[0]] = self.inputs_buf[0][begin:end:strides]
            elif self.inputs_buf[1].size == 2:
                begin = self.inputs_buf[1]
                end = self.inputs_buf[2]
                strides = self.inputs_buf[3]
                self.model.constant[self.outputs[0]] = self.inputs_buf[0][begin[0]:end[0]:strides[0], begin[1]:end[1]:strides[1]]
            elif self.inputs_buf[1].size == 3:
                begin = self.inputs_buf[1]
                end = self.inputs_buf[2]
                strides = self.inputs_buf[3]
                self.model.constant[self.outputs[0]] = self.inputs_buf[0][begin[0]:end[0]:strides[0], begin[1]:end[1]:strides[1], begin[2]:end[2]:strides[2]]
            else:
                raise NotImplementedError(self.op.name)
        else:
            # Skip op if input shape == output shape
            if self.op.inputs[0].shape == self.op.outputs[0].shape: # Skip
                self.model.indentity[self.op.outputs[0].name] = self.model.indentity.get(self.op.inputs[0].name, self.op.inputs[0].name)
            else:
                self.layer_type = 'Slice'

                # Check Stride != 1
                if self.inputs_buf[3].size != list(self.inputs_buf[3]).count(1):
                    errorMsg = 'Error Op' + self.op.name + '(StridedSlice): Do not support stride > 1. OP ' + self.op.name + '\'s strides is ' + str(self.inputs_buf[3]) + '\n'
                    sys.exit(errorMsg)

                axis_index = np.nonzero(self.inputs_buf[2] - self.inputs_buf[1])[0]

                if axis_index.size > 1:
                    errorMsg = 'Error Op ' + self.op.name + '(StridedSlice): Can\'t slice more than one axis'
                    sys.exit(errorMsg)

                start = int(self.inputs_buf[1][axis_index[0]])
                end = int(self.inputs_buf[2][axis_index[0]])

                if start == 0:
                    slice_point = end
                    self.outputs.append(self.name+'_useless')
                elif end == self.op.inputs[0].shape[axis_index[0]]:
                    slice_point = start
                    self.outputs.insert(0, self.name+'_useless')
                else:
                    errorMsg = 'Error Op ' + self.op.name + '(StridedSlice): Can\'t support begin: ' + str(self.inputs_buf[1]) + ' end: ' + str(self.inputs_buf[2])
                    sys.exit(errorMsg)

                self.slice_param = dict()
                self.slice_param['axis'] = int(axis_index[0]) if self.layout == 'NCHW' else dim_map_nhwc2nchw[int(axis_index[0])]
                self.slice_param['slice_point'] = [slice_point]

                self.attrs = self.slice_param

                self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
