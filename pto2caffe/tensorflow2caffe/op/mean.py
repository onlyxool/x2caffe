from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import handleLegacyPad


class Mean(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Mean')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        axis = self.inputs_buf[1]
        if (axis.tolist() == [2, 3] and self.layout == 'NCHW') or axis.tolist() == [1, 2] and self.layout == 'NHWC':
            self.pooling_param = dict()
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.inputs_shape[0][2]
            self.pooling_param['kernel_w'] = self.inputs_shape[0][3]
            self.pooling_param['stride'] = 1
            self.pooling_param['ceil_mode'] = False

            if self.attrs['keep_dims']:
                self.keep_dims = True
            else:
                self.keep_dims = False
                self.reshape = 'Mean_' + self.op.name + '_split'
                self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

            # Padding
            legacy_pad = self.model.pad.get(self.op.inputs[0].name, {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
            padding = handleLegacyPad('VALID', self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.type)
            self.pooling_param.update(padding)

            self.attrs = self.pooling_param

            self.setParsed()
        else:
            raise NotImplementedError(self.op.name)


    def convert(self):
        layers = list()
        if self.keep_dims:
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param))
        else:
            layers.append(caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, [self.reshape], pooling_param=self.pooling_param))
            layers.append(caffe_layer('Reshape', self.reshape, [self.reshape], [None], self.outputs, reshape_param=self.reshape_param))

        self.setConverted()

        return layers
