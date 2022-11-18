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
            axes = self.inputs_buf[1].tolist()
            if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4:
                for index, axis in enumerate(axes):
                    axes[index] = dim_map_nhwc2nchw[axis]

            if len(axes) == 1 and int(axis[0]) == len(self.inputs_shape) - 1:
            if not self.attrs['keepdims']:
                self.type = 'Reduction'
            else:
                self.type = 'Reduction+Reshape'
                self.inter_blob = 'reduction_reshape_split'+str(self.index)

                self.type = 'Reduction'
                self.reduction_param = dict()
                self.reduction_param['operation'] = 1
                self.reduction_param['axis'] = int(axis)
                self.attrs = self.reduction_param
                self.setParsed()
            else:
                raise NotImplementedError(self.op.name)
        else:
            unSupported('Can\'t support axis == None')


    def convert(self):
        layers = list()
        if self.type == 'Reduction':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, reduction_param=self.reduction_param))
        elif self.type == 'Reduction+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, [self.inter_blob], reduction_param=self.reduction_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.inter_blob], [None], self.outputs, reshape_param=dict(shape=dict(dim=self.outputs_shape[0]))))

        self.setConverted()

        return layers
