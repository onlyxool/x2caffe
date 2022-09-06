import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Rsqrt(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Rsqrt')
        self.setInited()


    def parse(self):
        self.layer_type = 'Rsqrt'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], tf.math.rsqrt(self.inputs_buf[0], name=None).numpy())
        else:
            self.unSupported()


    def convert(self):
        pass
