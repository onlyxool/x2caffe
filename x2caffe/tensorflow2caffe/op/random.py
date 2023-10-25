import tensorflow as tf
from tensorflow2caffe.op.operator import Operator


class Random(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('RandomStandardNormal', 'RandomUniform'))
        self.setInited()


    def parse(self):
        self.type = self.operator_code
        super().__parse__()

        if self.operator_code == 'RandomStandardNormal':
            self.saveConstant(self.outputs[0], tf.random.normal(shape=self.op.outputs[0].shape, dtype=self.attrs['dtype'], seed=None, name=None).numpy())
        elif self.operator_code == 'RandomUniform':
            self.saveConstant(self.outputs[0], tf.random.uniform(shape=self.op.outputs[0].shape, dtype=self.attrs['dtype'], seed=None, name=None).numpy())


    def convert(self):
        pass
