import tensorflow as tf
from tensorflow2caffe.op.operator import Operator


class PlaceholderWithDefault(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('PlaceholderWithDefault'))
        self.setInited()


    def parse(self):
        self.layer_type = self.operator_code
        super().__parse__()

        if self.inputs_buf[0] is not None:
            if self.operator_code == 'PlaceholderWithDefault':
                self.model.constant[self.outputs[0]] = tf.raw_ops.PlaceholderWithDefault(tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype), shape=self.op.outputs[0].shape, name=None)
        else:
            raise NotImplementedError(self.op.name)


    def convert(self):
        pass
