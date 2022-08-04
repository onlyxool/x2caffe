import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Tile(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Tile')
        self.setInited()


    def parse(self):
        self.layer_type = 'Tile'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            self.model.constant[self.outputs[0]] = np.tile(self.inputs_buf[0], self.inputs_buf[1])
        else:
            raise NotImplementedError(self.op.name)

            self.tile_param = dict()

            self.attrs = self.tile_param

            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, tile_param=self.tile_param)

        self.setConverted()

        return [layer]
