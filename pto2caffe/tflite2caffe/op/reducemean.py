import sys
import tflite
from util import dim_map_nhwc2nchw

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class ReduceMean(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'MEAN')
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'

        op_opt = self.op.BuiltinOptions()
        opt = tflite.ReducerOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.KeepDims():
            sys.exit('Reduce: Attributes KeepDims Not Supported.\n')

        self.parseInputOutput()
        assert(len(self.inputs_shape[0]) == 4)

        axis = list()
        for dim in self.inputs_buf[1]:
            axis.append(dim_map_nhwc2nchw[dim])

        if axis != [2,3]:
            errorMsg = 'ReduceMean\'s axis: ' + axis + ' Not support'
            sys.exit(errorMsg)

        # Attributes
        self.pooling_param = dict()
        self.pooling_param['pool'] = 1
        self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
        self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
        self.pooling_param['stride'] = 1
        self.pooling_param['ceil_mode'] = False

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]