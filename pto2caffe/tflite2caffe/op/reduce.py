import tflite
import logging
from util import dim_map_nhwc2nchw

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class Reduce(Operator):

    TypeMapping = {
        tflite.BuiltinOperator.MEAN: 'ReduceMean',
        tflite.BuiltinOperator.REDUCE_MAX: 'ReduceMax',
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)

        self.pooling_param = dict()

        self.setInited()


    @property
    def type(self):
        return 'Pooling'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()
        assert(len(self.inputs_shape[0]) == 4)

        # Option
        op_opt = self.op.BuiltinOptions()
        opt = tflite.ReducerOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        assert(not opt.KeepDims()), 'Reduce: KeepDims Not Supported.'

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
                print('ReduceMax\'s axis:', axis, 'Not support')
                raise NotImplementedError
        elif self.op_code == tflite.BuiltinOperator.MEAN:
            if axis == [2,3]:
                self.pooling_param['pool'] = 1
                self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
                self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
                self.pooling_param['stride'] = 1
                self.pooling_param['ceil_mode'] = False
                self.attrs = self.pooling_param
            else:
                print('ReduceMean\'s axis:', axis, 'Not support')
                raise NotImplementedError

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
