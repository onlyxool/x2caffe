import tflite
import numpy as np
from util import dim_map_nhwc2nchw

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class StridedSlice(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'STRIDED_SLICE')
        assert(self.op.InputsLength() == 4)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        self.type = 'Slice'

        self.parseInputOutput()

        op_opt = self.op.BuiltinOptions()
        opt = tflite.StridedSliceOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        if self.inputs_buf[0] is not None:
            import tensorflow as tf
            x = tf.constant(self.inputs_buf[0])
            begin = tf.constant(self.inputs_buf[1])
            end = tf.constant(self.inputs_buf[2])
            strides = tf.constant(self.inputs_buf[3])

            begin_mask = opt.BeginMask()
            end_mask = opt.EndMask()
            ellipsis_mask = opt.EllipsisMask()
            new_axis_mask = opt.NewAxisMask()
            shrink_axis_mask = opt.ShrinkAxisMask()

            self.saveConstant(self.outputs[0], tf.strided_slice(x, begin=begin, end=end, strides=strides,
                    begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask,
                    shrink_axis_mask=shrink_axis_mask).numpy())
        elif self.inputs_shape[0] == self.outputs_shape[0]: # Skip
            self.byPassOperator()
        else:
            # Check Stride != 1
            if self.inputs_buf[3].size != list(self.inputs_buf[3]).count(1):
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (STRIDED_SLICE): Do not support stride > 1. strides is ' + str(self.inputs_buf[3]))
                return

            if len(self.inputs_shape[0]) > 4:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (STRIDED_SLICE): Do not support dimitions > 4.')
                return

            axis_index = np.nonzero(self.inputs_buf[2] - self.inputs_buf[1])[0]

            if axis_index.size > 1:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (STRIDED_SLICE): Can\'t slice more than one axis, begin: '
                        + str(self.inputs_buf[1]) + ' ends: ' + str(self.inputs_buf[2]))
                return
            else:
                axis_index = int(axis_index)

            begin_mask = opt.BeginMask()
            end_mask = opt.EndMask()
            ellipsis_mask = opt.EllipsisMask()
            new_axis_mask = opt.NewAxisMask()
            shrink_axis_mask = opt.ShrinkAxisMask()

            if ellipsis_mask > 0:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (STRIDED_SLICE): Can\'t Support ellipsis_mask > 0')
                return
            if new_axis_mask > 0:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (STRIDED_SLICE): Can\'t Support new_axis_mask > 0')
                return
            if shrink_axis_mask > 0:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (STRIDED_SLICE): Can\'t Support shrink_axis_mask > 0')
                return

            begin_mask = str.zfill('{:b}'.format(begin_mask),len(self.inputs_shape[0]))
            end_mask = str.zfill('{:b}'.format(end_mask),len(self.inputs_shape[0]))
            ellipsis_mask = str.zfill('{:b}'.format(ellipsis_mask),len(self.inputs_shape[0]))
            new_axis_mask = str.zfill('{:b}'.format(new_axis_mask),len(self.inputs_shape[0]))
            shrink_axis_mask = str.zfill('{:b}'.format(shrink_axis_mask),len(self.inputs_shape[0]))

            start = int(self.inputs_buf[1][axis_index])
            if begin_mask[axis_index] == '1':
                start = start if start > 0 else 0

            end = int(self.inputs_buf[2][axis_index])
            if end_mask[axis_index] == '1':
                end = self.inputs_shape[0][dim_map_nhwc2nchw[axis_index]] if self.inputs_shape[0][dim_map_nhwc2nchw[axis_index]] > end else end

            if start == 0:
                slice_point = end
                self.outputs.append('intermediate_' + str(self.index))
            elif end == self.inputs_shape[0][dim_map_nhwc2nchw[axis_index]]:
                slice_point = start
                self.outputs.insert(0, 'intermediate_' + str(self.index))
            else:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error Op (STRIDED_SLICE): Can\'t support begin: ' + str(self.inputs_buf[1]) + ' end: ' + str(self.inputs_buf[2]))
                print(errorMsg)
                return

            self.slice_param = dict()
            self.slice_param['axis'] = axis_index if self.layout == 'NCHW' else dim_map_nhwc2nchw[axis_index]
            self.slice_param['slice_point'] = [slice_point]

            self.attrs = self.slice_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
