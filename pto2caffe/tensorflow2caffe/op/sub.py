import copy
import logging

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import trim_one
from util import compute_scale_axis

logger = logging.getLogger('TensorFlow2Caffe')


class Sub(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()


    @property
    def type(self):
        if hasattr(self, 'eltwise_param'):
            return 'Eltwise'
        elif hasattr(self, 'scale_param'):
            return 'Scale'
        else:
            return 'Sub'


    def parse(self):
        logger.debug('Parsing %s...', self.type)

        self.parseInput()
        self.parseOutput()

        self.parseAttributes()
        if self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
        else:
            self.bias = -self.inputs_buf[1]

            self.scale_param = dict()
            self.scale_param['bias_term'] = True

            # Axis
            if self.bias.shape != () and self.bias.shape != []: 
                self.scale_param['axis'] = self.inputs_shape[0].index(self.bias.shape[0])
                self.scale_param['num_axes'] = len(self.bias.shape)
            else:
                self.scale_param['num_axes'] = 0 

            self.attrs = self.scale_param

        self.setParsed()


    def convert(self):
        if hasattr(self, 'scale_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, None, self.bias, scale_param=self.scale_param)
        elif hasattr(self, 'eltwise_param'):
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)

        self.setConverted()

        return [layer]
