import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class TensorList(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('TensorListReserve'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.operator_code == 'TensorListReserve':
            element_shape = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            num_elements = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            element_dtype = self.attrs['element_dtype']
            self.saveConstant(self.outputs[0], tf.raw_ops.TensorListReserve(element_shape=element_shape, num_elements=num_elements, element_dtype=element_dtype, name=None))


    def convert(self):
        pass 
