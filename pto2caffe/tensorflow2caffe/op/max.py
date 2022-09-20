from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import handleLegacyPad
from util import dim_map_nhwc2nchw

class Max(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Max')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[1] is None:
            self.unSupported('Can\'t support axis == None')
            return
        else:
            axis = self.inputs_buf[1]

        if self.inputs_buf[0] is not None:
            import tensorflow as tf
            x = tf.constant(self.inputs_buf[0], dtype=self.op.inputs[0].dtype)
            self.saveConstant(self.outputs[0], tf.raw_ops.Max(input=x, axis=axis, keep_dims=self.attrs['keep_dims'], name=None).numpy())
        else:
            if axis.size == 1:
                self.type = 'ArgMax' # Remove

                self.argmax_param = dict()
                self.argmax_param['out_max_val'] = True
                self.argmax_param['top_k'] = 1
                self.argmax_param['axis'] = dim_map_nhwc2nchw[int(axis)]

                self.attrs = self.argmax_param
            elif (axis.tolist() == [2, 3] and self.layout == 'NCHW') or axis.tolist() == [1, 2] and self.layout == 'NHWC':
                self.type = 'Pooling'

                self.pooling_param = dict()
                self.pooling_param['pool'] = 0
                self.pooling_param['kernel_h'] = self.op.inputs[0].shape[self.ndim('H')]
                self.pooling_param['kernel_w'] = self.op.inputs[0].shape[self.ndim('W')]
                self.pooling_param['stride'] = 1
                self.pooling_param['ceil_mode'] = False

                if not self.attrs['keep_dims']:
                    self.type = 'Pooling+Reshape'
                    self.keep_dims = False
                    self.reshape = 'Max_' + self.op.name + '_split'
                    self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
                else:
                    self.keep_dims = True

                # Padding
                legacy_pad = self.model.pad.get(self.op.inputs[0].name, {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
                padding = handleLegacyPad('VALID', self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.layer_type)
                self.pooling_param.update(padding)

                self.attrs = self.pooling_param
            else:
                raise NotImplementedError(self.op.name)

            self.setParsed()


    def convert(self):
        layers = list()
        if self.type == 'ArgMax':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, argmax_param=self.argmax_param))
        elif self.type == 'Pooling':
            layers.append(caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        elif self.type == 'Pooling+Reshape':
            layers.append(caffe_layer(self.layer_type[0], self.name[0], self.inputs, self.inputs_buf, [self.reshape], pooling_param=self.pooling_param))
            layers.append(caffe_layer(self.layer_type[1], self.name[1], [self.reshape], [None], self.outputs, reshape_param=self.reshape_param))

        self.setConverted()

        return layers
