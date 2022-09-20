import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Maximum(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Maximum')
        self.setInited()


    def parse(self):
        self.layer_type = 'Maximum'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import tensorflow as tf
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            y = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Maximum(x=x, y=y, name=None).numpy())
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None:
            self.layer_type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 2
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_buf[1] is not None:
            if np.count_nonzero(self.inputs_buf[1]) > 0:
                self.layer_type = 'Dummy+Eltwise'
                self.dummy_data_param = dict()
                self.dummy_data_param['data_filler'] = dict(type='constant', value=self.inputs_buf[1])
                self.dummy_data_param['shape'] = dict(dim=self.inputs_shape[0])

                self.inter_blob = 'preDummy_split' + str(self.index)

                self.eltwise_param = dict()
                self.eltwise_param['operation'] = 2

                self.attrs = self.eltwise_param
                self.setParsed()
            else:
                self.layer_type = 'ReLU'
                self.relu_param = dict()
                self.relu_param['negative_slope'] = 0

                self.attrs = self.relu_param
                self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Dummy+Eltwise':
            layers.append(caffe_layer('Dummy', 'Dummy'+str(self.index), self.inputs, self.inputs_buf, [self.inter_blob], dummy_data_param=self.dummy_data_param))
            layers.append(caffe_layer('Eltwise', 'Eltwise'+str(self.index), [self.inter_blob], self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param))
        elif self.type == 'ReLU':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
