import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Floor(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Floor')
        self.setInited()


    def parse(self):
        self.type = 'Floor'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], tf.math.floor(self.inputs_buf[0], name=None).numpy())
        else:
            self.unSupported()


    def convert(self):
        pass
