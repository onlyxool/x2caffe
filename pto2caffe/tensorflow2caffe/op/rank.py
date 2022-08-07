import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Rank(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Rank')
        self.setInited()


    def parse(self):
        self.layer_type = 'Rank'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            input = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            self.model.constant[self.outputs[0]] = tf.raw_ops.Rank(input=input, name=None).numpy()
        else:
            self.model.unsupport.append(self.operator_code)


    def convert(self):
        pass
