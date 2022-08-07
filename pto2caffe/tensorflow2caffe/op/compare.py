import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Compare(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('GreaterEqual', 'Greater', 'Less', 'Equal'))
        self.setInited()


    def parse(self):
        self.layer_type = self.operator_code
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            y = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            if self.operator_code == 'GreaterEqual':
                self.model.constant[self.outputs[0]] = tf.math.greater_equal(x, y).numpy()
            elif self.operator_code == 'Greater':
                self.model.constant[self.outputs[0]] = tf.math.greater(x, y).numpy()
            elif self.operator_code == 'Less':
                self.model.constant[self.outputs[0]] = tf.math.less(x, y).numpy()
            elif self.operator_code == 'Equal':
                self.model.constant[self.outputs[0]] = tf.math.equal(x, y).numpy()
        else:
            self.model.unsupport.append(self.operator_code)


    def convert(self):
        pass
