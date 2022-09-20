import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Slice(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Slice')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            input = tf.constant(self.inputs_buf[0], self.op.inputs[0].dtype)
            begin = tf.constant(self.inputs_buf[1], self.op.inputs[1].dtype)
            size = tf.constant(self.inputs_buf[2], self.op.inputs[2].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Slice(input=input, begin=begin, size=size, name=None).numpy())
        elif self.inputs_shape[0] == self.outputs_shape[0]:
            self.byPassOperator()
        else:
            self.type = 'Slice'

            if len(np.where(self.inputs_buf[2]>0)) > 1:
                self.unSupported('Can\'t Slice more than one axis')
                return

            self.slice_param = dict()
            op_axis = (self.inputs_buf[2]>0).tolist().index(True)
            self.slice_param['axis'] = op_axis if len(self.inputs_shape) < 4 else dim_map_nhwc2nchw[op_axis]

            if self.inputs_buf[1][op_axis] == 0:
                slice_points = self.inputs_buf[2][op_axis]
                self.outputs.insert(0, self.name+'_useless')
            elif self.inputs_buf[1][op_axis] > 0:
                slice_points = self.inputs_buf[1][op_axis]
                self.outputs.append(self.name+'_useless')
            else:
                raise NotImplementedError

            self.slice_param['slice_point'] = slice_points

            self.attrs = self.slice_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
