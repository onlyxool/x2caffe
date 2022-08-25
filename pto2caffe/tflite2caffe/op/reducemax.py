import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

from util import handleLegacyPad, dim_map_nhwc2nchw


class ReduceMax(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'REDUCE_MAX')
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        self.parseInputOutput()

        op_opt = self.op.BuiltinOptions()
        opt = tflite.ReducerOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        if self.inputs_buf[1].ndim == 0:
            axis = int(self.inputs_buf[1]) if int(self.inputs_buf[1]) > 0 else len(self.inputs_shape[0]) + int(self.inputs_buf[1])
            axis = [dim_map_nhwc2nchw[axis]] if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4 else [axis]
        elif self.inputs_buf[1].size >= 1:
            axis = [dim_map_nhwc2nchw[dim] for dim in self.inputs_buf[1]]

        if opt.KeepDims() and axis == [2,3] and len(self.inputs_shape[0]) == 4:
            self.layer_type = 'Pooling'

            self.pooling_param = dict()
            self.pooling_param['pool'] = 0
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
            self.pooling_param['stride'] = 1
            self.pooling_param['ceil_mode'] = False

            # Padding
            legacy_pad = self.model.pad.get(self.op.Inputs(0), {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
            padding = handleLegacyPad('VALID', self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.type)
            self.pooling_param.update(padding)

            self.attrs = self.pooling_param
        else:
            self.layer_type = 'Reduction'

            if axis != [i for i in range(len(self.inputs_shape[0]))][-len(axis):]:
                self.model.unsupport.append(self.operator_code)
                self.model.errorMsg.append('Error: Op (REDUCE_MAX): Do not support axis:' + str(axis))
                return

            self.reduction_param = dict()
            self.reduction_param['operation'] = 5
            self.reduction_param['axis'] = axis[0]
            self.reduction_param['coeff'] = 1.0

            self.attrs = self.reduction_param

        self.setParsed()


    def convert(self):
        if self.type == 'Pooling':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)
        elif self.type == 'Reduction':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reduction_param=self.reduction_param)

        self.setConverted()

        return [layer]
