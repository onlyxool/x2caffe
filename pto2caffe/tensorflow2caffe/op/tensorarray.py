import tensorflow as tf

from tensorflow2caffe.op.operator import Operator


class TensorArray(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('TensorArrayV3', 'TensorArraySizeV3', 'TensorArrayWriteV3', 'TensorArrayReadV3', 'TensorArrayScatterV3'))
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.operator_code == 'TensorArrayV3':
            size = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            dtype = self.attrs['dtype']
            element_shape = self.attrs['element_shape']
            dynamic_size = self.attrs['dynamic_size']
            clear_after_read = self.attrs['clear_after_read']
            identical_element_shapes = self.attrs['identical_element_shapes']
            tensor_array_name = self.attrs['tensor_array_name']

            outputs = tf.raw_ops.TensorArrayV3(size=size, dtype=dtype, element_shape=element_shape, dynamic_size=dynamic_size, clear_after_read=clear_after_read,
                identical_element_shapes=identical_element_shapes, tensor_array_name=tensor_array_name, name=None)

            self.saveConstant(self.outputs[0], outputs[0])
            self.saveConstant(self.outputs[1], outputs[1])
        elif self.operator_code == 'TensorArraySizeV3':
            if self.inputs_buf[1] is None:
                self.unSupported()
                return

            handle = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            flow_in = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.TensorArraySizeV3(handle=handle, flow_in=flow_in, name=None))
        elif self.operator_code == 'TensorArrayWriteV3':
            if self.inputs_buf[2] is None:
                self.unSupported()
                return

            handle = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            index = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            value = tf.constant(self.inputs_buf[2], dtype=self.op.inputs[2].dtype)
            flow_in = tf.constant(self.inputs_buf[3], dtype=self.op.inputs[3].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.TensorArrayWriteV3(handle=handle, index=index, value=value, flow_in=flow_in, name=None))
        elif self.operator_code == 'TensorArrayReadV3':
            if self.inputs_buf[2] is None:
                self.unSupported()
                return

            handle = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            index = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            flow_in = tf.constant(self.inputs_buf[2], dtype=self.op.inputs[2].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.TensorArrayReadV3(handle=handle, index=index, flow_in=flow_in, dtype=self.attrs['dtype'], name=None))
        elif self.operator_code == 'TensorArrayScatterV3':
            if self.inputs_buf[2] is None:
                self.unSupported()
                return

            handle = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            indices = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            value = tf.constant(self.inputs_buf[2], dtype=self.op.inputs[2].dtype)
            flow_in = tf.constant(self.inputs_buf[3], dtype=self.op.inputs[3].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.TensorArrayScatterV3(handle=handle, indices=indices, value=value, flow_in=flow_in, name=None))


    def convert(self):
        pass 
