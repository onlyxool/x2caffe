from tensorflow2caffe.op.operator import Operator


class Fill(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Fill')
        self.setInited()


    def parse(self):
        self.layer_type = 'Fill'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            import tensorflow as tf
            dims = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            value = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.model.constant[self.outputs[0]] = tf.raw_ops.Fill(dims=dims, value=value, name=None).numpy()
        else:
            self.model.unsupport.append(self.operator_code)


    def convert(self):
        pass
