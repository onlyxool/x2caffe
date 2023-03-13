import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class ExpandDims(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'ExpandDims')
        self.setInited()


    def parse(self):
        self.type = 'Reshape'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], np.expand_dims(self.inputs_buf[0], int(self.inputs_buf[1])))
        else:
            dim = int(self.inputs_buf[1])

            if self.outputs_shape[0] is not None and all(self.outputs_shape[0]):
                target_shape = list(np.expand_dims(np.random.random(self.inputs_shape[0]), dim).shape)
            else:
                self.unSupported('Can\'t Support Output Shape is ' + str(self.outputs_shape[0]))
                return

            self.reshape_param = dict(shape=dict(dim=target_shape))

            self.attrs = self.reshape_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
