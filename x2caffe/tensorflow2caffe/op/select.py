from tensorflow2caffe.op.operator import Operator


class Select(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'SelectV2')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import tensorflow as tf
            condition = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            t = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            e = tf.constant(self.inputs_buf[2], dtype=self.op.inputs[2].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.SelectV2(condition=condition, t=t, e=e, name=None).numpy())
        else:
            self.unSupported()


    def convert(self):
        pass
