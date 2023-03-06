import numpy as np

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator
from util import dim_map_nhwc2nchw

class Slice(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'SLICE')
        assert(self.op.InputsLength() == 3)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
        else:
            self.type = 'Slice'

            self.slice_param = dict()

            if len(np.where(self.inputs_buf[2]>0)) > 1:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (SLICE): Do not support begin:' + str(self.inputs_buf[1]))
                return

            op_axis = (self.inputs_buf[2]>0).tolist().index(True)
            self.slice_param['axis'] = op_axis if len(self.inputs_shape) < 4 else dim_map_nhwc2nchw[op_axis]

            if self.inputs_buf[1][op_axis] == 0:
                slice_points = self.inputs_buf[2][op_axis]
                self.outputs.insert(0, 'intermediate_' + str(self.index))
            elif self.inputs_buf[1][op_axis] > 0:
                slice_points = self.inputs_buf[1][op_axis]
                self.outputs.append('intermediate_' + str(self.index))
            else:
                self.unSupported('Can\'t support axis == ' + op_axis)
                return

            self.slice_param['slice_point'] = slice_points

            self.attrs = self.slice_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
