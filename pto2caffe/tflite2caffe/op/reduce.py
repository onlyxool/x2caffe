import sys
import tflite
import logging
from util import dim_map_nhwc2nchw

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class Reduce(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)

        self.pooling_param = dict()

        self.setInited()


    @property
    def type(self):
        return 'Pooling'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in (tflite.BuiltinOperator.MEAN, tflite.BuiltinOperator.REDUCE_MAX))
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()
        assert(len(self.inputs_shape[0]) == 4)

        # Attributes
        op_opt = self.op.BuiltinOptions()
        opt = tflite.ReducerOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.KeepDims():
            sys.exit('Reduce: Attributes KeepDims Not Supported.\n')

        axis = []
        for i in range(len(self.inputs_buf[1])):
            axis.append(dim_map_nhwc2nchw[self.inputs_buf[1][i]])

        if self.op_code == tflite.BuiltinOperator.REDUCE_MAX:
            if axis == [2,3]:
                self.pooling_param['pool'] = 0
                self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
                self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
                self.pooling_param['stride'] = 1
                self.pooling_param['ceil_mode'] = False
                self.attrs = self.pooling_param
            else:
                errorMsg = 'ReduceMax\'s axis: ' + axis + ' Not support'
                sys.exit(errorMsg)
        elif self.op_code == tflite.BuiltinOperator.MEAN:
            if axis == [2,3]:
                self.pooling_param['pool'] = 1
                self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
                self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
                self.pooling_param['stride'] = 1
                self.pooling_param['ceil_mode'] = False
                self.attrs = self.pooling_param
            else:
                errorMsg = 'ReduceMean\'s axis: ' + axis + ' Not support'
                sys.exit(errorMsg)

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
