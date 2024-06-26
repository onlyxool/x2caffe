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
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.type = 'Minimum'
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            y = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Minimum(x=x, y=y, name=None).numpy())
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 4
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[1] is not None:
            if np.count_nonzero(self.inputs_buf[1]) > 0:
                self.type = 'Dummy+Eltwise' # Need Test
                self.dummy_data_param = dict()
                self.dummy_data_param['data_filler'] = dict(type='constant', value=self.inputs_buf[1])
                self.dummy_data_param['shape'] = dict(dim=self.inputs_shape[0])

                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 4

                self.attrs = self.eltwise_param
                self.setParsed()
            else:
                self.type = 'ReLU'
                self.relu_param = dict()
                self.relu_param['negative_slope'] = 1

                self.attrs = self.relu_param
                self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'Eltwise':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'Dummy+Eltwise':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], [self.inputs[1]], self.inputs_buf, self.interblob, self.inputs_buf[1], dummy_data_param=self.dummy_data_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inputs[0], self.interblob[0]], self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'ReLU':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param))

        self.setConverted()

        return layers
