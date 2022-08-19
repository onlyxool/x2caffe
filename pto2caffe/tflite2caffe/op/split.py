import tflite
import numpy as np
from util import dim_map_nhwc2nchw

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Split(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'SPLIT')
        assert(self.op.InputsLength() == 2 or self.op.InputsLength() == 4)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        self.layer_type = 'Slice'

        self.parseInputOutput()

        op_opt = self.op.BuiltinOptions()
        opt = tflite.SplitOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        # Attributes
        self.slice_param = dict()
        # Axis
        if isinstance(self.inputs_buf[0], np.ndarray):
            self.slice_param['axis'] = dim_map_nhwc2nchw[self.inputs_buf[0][0]]
        elif isinstance(self.inputs_buf[0], int):
            self.slice_param['axis'] = dim_map_nhwc2nchw[self.inputs_buf[0]]

        assert((self.inputs_shape[1][self.slice_param['axis']] / opt.NumSplits()) == (self.outputs_shape[0][self.slice_param['axis']]))

        # Slice Points
        slice_points = []
        for i in range(opt.NumSplits()):
            slice_points.append(self.outputs_shape[i][self.slice_param['axis']])
        self.slice_param['slice_point'] = slice_points

        self.attrs = self.slice_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
