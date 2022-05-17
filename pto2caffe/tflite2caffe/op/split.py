import tflite
import numpy as np
from util import dim_map_nhwc2nchw

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Slice(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        self.slice_param = dict()
        assert(self.operator_code in ('SPLIT', 'STRIDED_SLICE'))
        assert(self.op.InputsLength() == 2 or self.op.InputsLength() == 4)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        self.layer_type = 'Slice'

        self.parseInputOutput()

        if self.operator_code == 'STRIDED_SLICE':
            op_opt = self.op.BuiltinOptions()
            opt = tflite.StridedSliceOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            m_begin = opt.BeginMask()
            m_end = opt.EndMask()

            assert(opt.EllipsisMask() == 0), "EllipsisMask not supported!"
            assert(opt.NewAxisMask() == 0), "NewAxisMask not supported!"
            assert(opt.ShrinkAxisMask() == 0), "ShrinkAxisMask not supported!"
            #print(opt.BeginMask(), opt.EndMask())

            assert(len(self.inputs_buf[1]) == len(self.inputs_shape[0]))

            raise NotImplementedError(self.operator_code)
        elif self.operator_code == 'SPLIT':
            op_opt = self.op.BuiltinOptions()
            opt = tflite.SplitOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)

            if isinstance(self.inputs_buf[0], np.ndarray):
                self.slice_param['axis'] = dim_map_nhwc2nchw[self.inputs_buf[0][0]]
            elif isinstance(self.inputs_buf[0], int):
                self.slice_param['axis'] = dim_map_nhwc2nchw[self.inputs_buf[0]]

            assert((self.inputs_shape[1][self.slice_param['axis']] / opt.NumSplits()) == (self.outputs_shape[0][self.slice_param['axis']]))

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
