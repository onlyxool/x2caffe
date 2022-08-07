from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Sum(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Sum')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None and self.inputs_buf[1] is not None:
            # Handle Constant OP
            import tensorflow as tf
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            axis = tf.constant(self.inputs_buf[1], dtype=self.op.inputs[1].dtype)
            self.model.constant[self.outputs[0]] = tf.raw_ops.Sum(x, axis=axis, keep_dims=self.attrs['keep_dims'], name=None)
        elif self.inputs_buf[1] is not None:
            axis = self.inputs_buf[1]

            if axis.size == 1 and int(axis) == len(self.inputs_shape) - 1:
                self.layer_type = 'Reduction'
                self.reduction_param = dict()
                self.reduction_param['operation'] = 1
                self.reduction_param['axis'] = int(axis)
                self.attrs = self.reduction_param
                self.setParsed()
            else:
                raise NotImplementedError(self.op.name)
        else:
            self.model.unsupport.append(self.operator_code)
            errorMsg = 'Error: Op Max (' + self.op.name + '): can\'t support axis == None'
            print(errorMsg)


    def convert(self):
        layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reduction_param=self.reduction_param))

        self.setConverted()

        return [layer]
