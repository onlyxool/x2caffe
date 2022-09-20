import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class ByPassOperator(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('Complex', 'Identity', 'IdentityN', 'FakeQuantWithMinMaxVars'))
        self.setInited()


    def parse(self):
        self.layer_type = 'ByPassOperator'
        super().__parse__()

        if self.inputs_buf[0] is not None:
            if self.operator_code == 'FakeQuantWithMinMaxVars':
                self.saveConstant(self.outputs[0], tf.raw_ops.FakeQuantWithMinMaxVars(inputs=self.inputs_buf[0], min=self.inputs_buf[1], max=self.inputs_buf[2],
                        num_bits=self.attrs['num_bits'], narrow_range=self.attrs['narrow_range'], name=None).numpy())
            elif self.operator_code in ('Identity', 'IdentityN', 'Complex'):
                self.saveConstant(self.outputs[0], self.inputs_buf[0])
            elif self.operator_code == 'ComplexAbs':
                self.saveConstant(self.outputs[0], tf.raw_ops.ComplexAbs(x=self.inputs_buf[0], Tout=tf.dtypes.float32, name=None).numpy())
        else:
            self.byPassOperator()


    def convert(self):
        pass
