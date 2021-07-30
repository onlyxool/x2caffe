import tflite
import logging

from util import *
from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Concat(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.CONCATENATION: 'Concat',
    }


    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()


    @property
    def type(self):
        return 'Concat'


    def parse(self):
        logger.debug("Parsing %s...", self.shorty)

        assert(self.op.InputsLength() >= 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        op_opt = self.op.BuiltinOptions()
        opt = tflite.ConcatenationOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        self.concat_param = dict()
        self.concat_param['axis'] = dim_map_nhwc2nchw[opt.Axis()]
        self.attrs = self.concat_param

        self.setParsed()


    def propagatableTensors(self):
        return self.inputs + self.outputs


    def transform(self):
        logger.debug("Transforming %s...", self.shorty)
        layout = self.outputs[0].layout
        if layout is not None:
            axis = self.attrs['axis']
            axis = axis if axis >= 0 else (axis + len(layout.perm))
            self.attrs['axis'] = layout.perm.index(axis)

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)
        self.setConverted()
        return [layer]
