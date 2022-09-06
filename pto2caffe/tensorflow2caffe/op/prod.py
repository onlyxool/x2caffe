import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Prod(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Prod')
        self.setInited()


    def parse(self):
        self.layer_type = 'Prod'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            input = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            axis = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Prod(input=input, axis=axis, keep_dims=self.attrs['keep_dims'], name=None).numpy())
        else:
            self.unSupported()


    def convert(self):
        pass
