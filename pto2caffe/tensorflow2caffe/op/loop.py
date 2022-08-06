import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Loop(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('LoopCond', 'NextIteration'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            if self.operator_code == 'LoopCond':
                input = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
                self.model.constant[self.outputs[0]] = tf.raw_ops.LoopCond(input=input, name=None).numpy()
            elif self.operator_code == 'NextIteration':
                data = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
                self.model.constant[self.outputs[0]] = tf.raw_ops.NextIteration(data=data, name=None).numpy()
        else:
            import sys
            errorMsg = 'Error: Operator [' + self.operator_code + '] does not Support.\n'
            sys.exit(errorMsg)


    def convert(self):
        pass
