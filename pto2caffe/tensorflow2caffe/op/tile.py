from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Tile(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Tile')
        self.setInited()


    def parse(self):
        self.type = 'Tile'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import tensorflow as tf
            x = tf.constant(self.inputs_buf[0], self.op.inputs[0].dtype)
            multiples = tf.constant(self.inputs_buf[1], self.op.inputs[1].dtype)
            self.saveConstant(self.outputs[0], tf.tile(x, multiples).numpy())
        else:

            self.tile_param = list()
            for index, axis in enumerate(self.inputs_buf[1]):
                self.tile_param.append(dict())
                self.tile_param[index]['axis'] = index
                self.tile_param[index]['tiles'] = int(self.inputs_buf[1][index])
            self.attrs = self.tile_param

            raise NotImplementedError
            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, tile_param=self.tile_param)

        self.setConverted()

        return [layer]
