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
            self.saveConstant(self.outputs[0], tf.raw_ops.Sum(x, axis=axis, keep_dims=self.attrs['keep_dims'], name=None).numpy())
        elif self.inputs_buf[1] is not None:
            axes = [self.inputs_buf[1].tolist()] if isinstance(self.inputs_buf[1].tolist(), int) else self.inputs_buf[1].tolist()
            if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4:
                for index, axis in enumerate(axes):
                    axes[index] = dim_map_nhwc2nchw[axis]

            if len(axes) == 1 and int(axes[0]) == len(self.inputs_shape) - 1:
                if 'keepdims' not in self.attrs or not self.attrs['keepdims']:
                    self.type = 'Reduction'
                else:
                    self.type = 'Reduction+Reshape'

                    self.type = 'Reduction'
                    self.reduction_param = dict()
                    self.reduction_param['operation'] = 1
                    self.reduction_param['axis'] = int(axes[0])
                    self.attrs = self.reduction_param
                    self.setParsed()
            else:
                self.unSupported('Can\'t support axis == ' + str(axes))
        else:
            unSupported('Can\'t support axis == None')


    def convert(self):
        layers = list()
        if self.type == 'Reduction':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reduction_param=self.reduction_param))
        elif self.type == 'Reduction+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, self.interblob, reduction_param=self.reduction_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], self.interblob, [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))

        self.setConverted()

        return layers
