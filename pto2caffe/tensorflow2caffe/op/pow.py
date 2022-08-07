import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Pow(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Pow')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pow'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            y = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.model.constant[self.outputs[0]] = tf.raw_ops.Pow(x=x, y=y, name=None)
        else:
            self.model.unsupport.append(self.operator_code)


    def convert(self):
        pass
