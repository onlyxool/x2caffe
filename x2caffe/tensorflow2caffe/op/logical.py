import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Logical(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('LogicalOr', 'LogicalAnd'))
        self.setInited()


    def parse(self):
        self.type = self.operator_code
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            y = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            if self.operator_code == 'LogicalOr':
                self.saveConstant(self.outputs[0], tf.raw_ops.LogicalOr(x=x, y=y, name=None).numpy())
            elif self.operator_code == 'LogicalAnd':
                self.saveConstant(self.outputs[0], tf.raw_ops.LogicalAnd(x=y, y=y, name=None).numpy())
        else:
            self.unSupported()


    def convert(self):
        pass
