from tensorflow2caffe.op.operator import Operator


class GatherV2(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'GatherV2')
        self.setInited()


    def parse(self):
        self.layer_type = 'GatherV2'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            import tensorflow as tf
            params = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            indices = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            axis = tf.constant(self.inputs_buf[2], dtype=self.op.inputs[2].dtype)
            self.model.constant[self.outputs[0]] = tf.raw_ops.GatherV2(params=params, indices=indices, axis=axis, batch_dims=self.attrs['batch_dims'], name=None)
        else:
            import sys
            errorMsg = 'Error: Operator [ GatherV2 ] does not Support.\n'
            sys.exit(errorMsg)


    def convert(self):
        pass
