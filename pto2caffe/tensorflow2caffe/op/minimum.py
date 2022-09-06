import numpy as np
import tensorflow as tf

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Minimum(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Minimum')
        self.setInited()


    def parse(self):
        self.layer_type = 'Minimum'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            y = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Minimum(x=x, y=y, name=None).numpy())
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.layer_type = 'Eltwise'

            # Attribute
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 4

            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[1] is not None:
            self.layer_type = 'ReLU'

            # Check weather y == 0
            if np.count_nonzero(self.inputs_buf[1]) > 0:
                self.unSupported('Does not Support y != 0.')
                return

            # Attribute
            self.relu_param = dict()
            self.relu_param['negative_slope'] = 1

            self.attrs = self.relu_param
            self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'ReLU':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
