import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class GatherV2(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'GatherV2')
        self.setInited()


    def parse(self):
        self.type = 'GatherV2'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None and self.inputs_buf[2] is not None:
            params = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            indices = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            axis = tf.constant(self.inputs_buf[2], dtype=self.op.inputs[2].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.GatherV2(params=params, indices=indices, axis=axis, batch_dims=self.attrs['batch_dims'], name=None).numpy())
        elif self.inputs_buf[0] is None:
            self.unSupported('Cant Parse param is None')
        elif self.inputs_buf[2] is None:
            self.unSupported('Cant Parse axis is None')
        else:
            self.unSupported()


    def convert(self):
        pass
