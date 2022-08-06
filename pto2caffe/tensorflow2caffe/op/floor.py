from tensorflow2caffe.op.operator import Operator


class Floor(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Floor')
        self.setInited()


    def parse(self):
        self.layer_type = 'Floor'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import tensorflow as tf
            self.model.constant[self.outputs[0]] = tf.math.floor(self.inputs_buf[0], name=None).numpy()
        else:
            raise NotImplementedError(self.op.name)


    def convert(self):
        pass
