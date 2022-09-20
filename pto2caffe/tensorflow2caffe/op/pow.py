import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class Pow(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Pow')
        self.setInited()


    def parse(self):
        self.type = 'Pow'
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            y = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Pow(x=x, y=y, name=None).numpy())
        elif self.inputs_buf[1] is not None and self.inputs_buf[1].size == 1:
            self.power_param = dict()

            self.power_param['power'] = int(self.inputs_buf[1])
            self.power_param['scale'] = 1
            self.power_param['shift'] = 0

            self.attrs = self.power_param
            self.setParsed()
        else:
            self.unSupported()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, power_param=self.power_param)

        self.setConverted()

        return [layer]
