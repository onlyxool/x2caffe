import logging

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import dim_map_nhwc2nchw

logger = logging.getLogger('TensorFlow2Caffe')

class Concat(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.concat_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Concat'


    def parse(self):
        logger.debug('Parsing %s...', self.type)
        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        for index, input_buf in enumerate(self.inputs_buf):
            if input_buf is not None:
                self.concat_param['axis'] = dim_map_nhwc2nchw[self.inputs_buf[index]]

        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]
